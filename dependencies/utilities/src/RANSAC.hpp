#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <educelab/core/utils/Math.hpp>

namespace ransac {
template <class Value, class T> struct RANSACResult {
  T fitness{0.};
  std::vector<Value> inliers;
  T inlier_rmse{educelab::INF<T>};
};

template <class Xs, class FitFunc, class EvalFunc>
auto RANSAC(const Xs &x, const FitFunc &fit, const EvalFunc &eval,
            double threshold, std::size_t nSamples, std::size_t nIters = 1000,
            const std::optional<std::mt19937::result_type> seed = std::nullopt,
            double probability = 0.99999999) {
  using ValueType = typename Xs::value_type;
  using Model = typename std::invoke_result_t<FitFunc, const Xs &>::second_type;
  using Result = RANSACResult<ValueType, double>;

  Result bestResult;
  Model bestModel;

  // Set up rng
  static std::mt19937 rng{std::random_device{}()};
  if (seed) {
    rng.seed(seed.value());
  }

  // Iterate. nIters is a hard bound; breakIter is the adaptive early exit and
  // is never allowed to exceed it.
  std::size_t breakIter{nIters};
  for (std::size_t i = 0; i < nIters and i < breakIter; ++i) {
    // Randomly sample
    Xs samples;
    std::sample(std::begin(x), std::end(x), std::back_inserter(samples),
                nSamples, rng);

    // Fit and evaluate the model
    auto [success, model] = fit(samples);
    if (not success) {
      continue;
    }

    Result result;
    double sum_error = 0.0;
    for (const auto &ro : x) {
      auto err = eval(ro, model);
      if (err < threshold) {
        sum_error += err * err;
        result.inliers.push_back(ro);
      }
    }

    if (!result.inliers.empty()) {
      result.fitness = static_cast<double>(result.inliers.size()) /
                       static_cast<double>(x.size());
      result.inlier_rmse =
          std::sqrt(sum_error / static_cast<double>(result.inliers.size()));
    }

    // Update our best models
    auto improvedFitness = result.fitness > bestResult.fitness;
    auto improvedRMSE = result.fitness == bestResult.fitness and
                        result.inlier_rmse < bestResult.inlier_rmse;
    if (improvedFitness or improvedRMSE) {
      bestResult = result;
      bestModel = model;
      if (result.fitness < 1.) {
        // Break early based on fitness/rmse: stop once the chance of not yet
        // having drawn an all-inlier sample has fallen below 1 - probability.
        // The denominator underflows to zero for a small fitness and a large
        // nSamples (fitness^nSamples < 2^-53), which makes the quotient -inf
        // and its conversion to breakIter undefined, so test it before
        // dividing and fall back to the hard bound when it is unusable.
        const auto denom = std::log(1. - std::pow(result.fitness, nSamples));
        if (std::isfinite(denom) and denom < 0.) {
          breakIter = static_cast<std::size_t>(
              std::min(static_cast<double>(nIters),
                       std::log(1. - probability) / denom));
        } else {
          breakIter = nIters;
        }
      } else {
        break;
      }
    } // if improved
  } // for nIters

  if (bestResult.inliers.empty()) {
    // If we never found a usable model, report failure rather than refitting on
    // an empty inlier set with an uninitialized model (which is undefined
    // behavior in the fit function).
    return std::make_pair(false, bestModel);
  }

  // Finalize the best model by refitting it on the whole consensus set. A refit
  // that fails does not invalidate the consensus we already have, so keep the
  // model the consensus set was found with rather than failing the call.
  auto [refit, refitModel] = fit(bestResult.inliers);
  if (refit) {
    return std::make_pair(true, refitModel);
  }
  return std::make_pair(true, bestModel);
}
} // namespace ransac
