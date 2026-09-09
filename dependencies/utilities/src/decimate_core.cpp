#include "decimate_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <vcg/complex/append.h>
#include <vcg/complex/algorithms/clean.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/point_sampling.h>
#include <vcg/complex/algorithms/stat.h>
#include <vcg/complex/algorithms/update/bounding.h>
#include <vcg/complex/algorithms/update/curvature.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/quality.h>
#include <vcg/complex/algorithms/update/topology.h>

#include <educelab/core/io/MeshIO.hpp>
#include <educelab/core/types/Mesh.hpp>
#include <educelab/core/types/UVMap.hpp>
#include <educelab/core/utils/String.hpp>

namespace fs = std::filesystem;

namespace pgs::decimate
{
namespace {

/// vcglib's "radius variance" for the curvature-weighted pass. See `oneSided`.
constexpr float kCurvatureVariance = 2.0F;

/// Prior exponent of the deviation-versus-threshold power law, which is very
/// flat -- of order `0.1` in practice. One measured round replaces it.
constexpr double kAssumedExponent = 0.1;

/// Bounds a fitted exponent is held inside: outside them the fit is too flat or
/// too steep to have come from two rounds, and is noise being extrapolated.
constexpr double kMinExponent = 0.02;
constexpr double kMaxExponent = 2.0;

/// Widest factor one extrapolated step may move the threshold: an exponent near
/// the floor above can otherwise ask for twenty orders of magnitude.
constexpr double kMaxStep = 1e6;

/// Prior slope of *face count* against threshold in log-log space, and the
/// bounds a measured one is held inside. Unlike the deviation, this curve does
/// not step: every threshold collapses a slightly different set of edges, so
/// two rounds fit it usefully even when they measured the same deviation. That
/// is what makes it the right thing to steer an unbracketed descent by.
constexpr double kAssumedFaceExponent = 0.4;
constexpr double kMinFaceExponent = 0.1;
constexpr double kMaxFaceExponent = 2.0;

/// Widest factor by which one unbracketed probe may grow the candidate, and so
/// the round's price: a round costs ten samples per face of the *coarser* mesh,
/// which is always the candidate. Growing geometrically bounds the whole
/// descent at `g/(g-1)` of its last round and the overshoot past the answer at
/// `g`, while still crossing decades of threshold in a handful of rounds.
constexpr double kCostGrowth = 2.5;

/// Two rounds whose measured deviations differ by less than this, in log space,
/// are on one tread of the staircase. The fit cannot see the riser from there:
/// it reads the flat as "nearly arrived" and asks for inches, which land on the
/// same tread and ask for inches again. A repeated deviation is therefore not a
/// datum but the absence of one, and the stride has to come from elsewhere.
constexpr double kPlateauTol = 0.01;

/// Largest share of the input a probe may be predicted to leave standing. A
/// candidate that is the input in all but name is the most expensive round on
/// offer and the least informative: it brackets, which a probe a decade coarser
/// would also have done for a fraction of the price.
///
/// This is a bias against a first-choice probe and emphatically **not** a wall.
/// A descent that reaches it asymptotes into it, buying a full measurement per
/// round to move a percent, which is the very failure the trust region it
/// replaced was retired for. `kMinUsefulGrowth` is what stops that.
constexpr double kMaxCandidateShare = 0.9;

/// Least growth that makes an unbracketed probe worth buying. Below it the cost
/// model can only offer a re-measurement of the round just finished, and the
/// question left -- whether anything below is feasible at all -- is settled by
/// one probe at the bottom of the range instead. That probe is at most the whole
/// input, which by then is barely more than the crawling one would have cost.
constexpr double kMinUsefulGrowth = 1.25;

/// How far a bracketed probe is held off the bracket's ends, a probe landing on
/// one being a re-measurement. A fixed factor: a share of a log-width that can
/// span orders would turn away probes sitting comfortably inside.
constexpr double kEndpointMargin = 1.02;

/// How far inside the budget the last round aims when nothing is feasible yet:
/// just missing it delivers nothing at all, which is worth more than faces.
constexpr double kSafetyAim = 0.5;

/// A deviation this close to the budget ends the search: the rounds after it pay
/// full measurement cost for a few percent. Stopping early forgoes only
/// coarseness, never the bound.
constexpr double kCloseEnough = 0.95;

/// Empirical prefactor on the `budget^2 * mean face area` seed. That is the
/// right shape but no closed form relates threshold to budget, so the seed is a
/// guess the fit only needs somewhere to start from.
constexpr double kSeedScale = 100.0;

/// A one-sided sampled distance from `from`'s surface to `to`'s: a uniform
/// Montecarlo pass (whose mean and RMS are the reported ones, comparable to
/// `metro`), a curvature-weighted pass at half the samples, then every vertex --
/// last, because `AddVert` overwrites the quality the weighted pass reads.
auto oneSided(Mesh& from, Mesh& to, int sampleNum, bool weighted,
              double distBound) -> DirectionStats
{
  using Sampler = vcg::tri::HausdorffSampler<Mesh>;
  using Sampling = vcg::tri::SurfaceSampling<Mesh, Sampler>;

  vcg::tri::UpdateBounding<Mesh>::Box(to);
  vcg::tri::UpdateNormal<Mesh>::PerVertexNormalizedPerFaceNormalized(from);

  // Built by hand: `HausdorffSampler(&to)` would build a grid against a box we
  // cannot pass it, which the margin below then replaces -- seconds of wasted
  // work per direction per round. `init(nullptr)` still resets the accumulators.
  Sampler sampler(nullptr);
  sampler.m = &to;
  vcg::tri::UpdateNormal<Mesh>::PerFaceNormalized(to);
  // `read` refuses a mesh with no faces and the collapse never empties one, so
  // the face grid is always the one that gets queried.
  sampler.useVertexSampling = false;
  sampler.markerFunctor.SetMesh(&to);
  sampler.hist.SetRange(0.0, to.bbox.Diag() / 100.0, 100);

  // `Set(begin, end)` inflates by `Diag()/nfaces`, which on a large mesh falls
  // below one float ULP -- no margin, so an extreme vertex indexes the grid out
  // of bounds. `Diag()/100` is vcg's own `SetBBox` margin.
  auto grown = to.bbox;
  grown.Offset(to.bbox.Diag() / 100.0F);
  sampler.unifGridFace.Set(to.face.begin(), to.face.end(), grown);

  // Samples beyond this are discarded outright -- not counted, not folded
  // into the max -- so `measure` derives it from both meshes together.
  sampler.dist_upper_bound = distBound;

  DirectionStats stats;
  Sampling::Montecarlo(from, sampler, sampleNum);
  stats.uniformSamples = static_cast<std::size_t>(sampler.n_total_samples);
  stats.mean = sampler.getMeanDist();
  stats.rms = sampler.getRMSDist();

  if (weighted and sampleNum > 1) {
    vcg::tri::UpdateTopology<Mesh>::VertexFace(from);
    vcg::tri::UpdateCurvature<Mesh>::PerVertexAbsoluteMeanAndGaussian(from);
    vcg::tri::UpdateQuality<Mesh>::VertexAbsoluteCurvatureFromHGAttribute(from);
    // Negated on purpose: WeightedMontecarlo reads quality as a sampling
    // *radius*, so high curvature would otherwise draw *fewer* samples.
    for (auto& v : from.vert) {
      if (not v.IsD()) {
        v.Q() = -v.Q();
      }
    }
    const auto before = sampler.n_total_samples;
    Sampling::WeightedMontecarlo(from, sampler, sampleNum / 2,
                                 kCurvatureVariance);
    stats.weightedSamples =
        static_cast<std::size_t>(sampler.n_total_samples - before);
  }

  const auto before = sampler.n_total_samples;
  Sampling::AllVertex(from, sampler);
  stats.vertexSamples =
      static_cast<std::size_t>(sampler.n_total_samples - before);

  stats.max = sampler.getMaxDist();
  return stats;
}

/// Whether a result sits *on* its face budget rather than merely under it. A
/// collapse drops one or two faces at a time, so two below is the tolerance;
/// further below and something else stopped it, which the report should say.
auto atFaceCap(std::size_t faces, std::size_t maxFaces) -> bool
{
  return maxFaces > 0 and faces <= maxFaces and maxFaces - faces <= 2;
}

/// The mean face area, as a scale for the quadric seed.
auto meanFaceArea(Mesh& mesh) -> double
{
  if (mesh.fn <= 0) {
    return 0.0;
  }
  return vcg::tri::Stat<Mesh>::ComputeMeshArea(mesh) / double(mesh.fn);
}

}  // namespace

auto to_string(Bound bound) -> std::string
{
  switch (bound) {
    case Bound::MaxError:
      return "max_error";
    case Bound::MaxFaces:
      return "max_faces";
    case Bound::QuadricError:
      return "quadric_error";
    case Bound::None:
    default:
      return "none";
  }
}

void read(const fs::path& path, Mesh& mesh, bool& uvsPresent)
{
  educelab::Mesh3f src;
  educelab::UVMap<> uvs;
  educelab::read_mesh(path, src, uvs);
  uvsPresent = not uvs.empty();

  if (src.num_faces() == 0) {
    throw std::runtime_error("mesh has no faces: " + path.string());
  }

  auto vi = vcg::tri::Allocator<Mesh>::AddVertices(
      mesh, static_cast<int>(src.num_vertices()));
  for (std::size_t i = 0; i < src.num_vertices(); ++i, ++vi) {
    const auto& v = src.vertex(i);
    vi->P() = Mesh::CoordType(v[0], v[1], v[2]);
  }

  auto fi = vcg::tri::Allocator<Mesh>::AddFaces(
      mesh, static_cast<int>(src.num_faces()));
  for (std::size_t i = 0; i < src.num_faces(); ++i, ++fi) {
    const auto& f = src.face(i);
    if (f.size() != 3) {
      // Hard error rather than a fan: a fan is a surface the caller never
      // asked about the deviation of, and it would be measured as though they had.
      throw NotTriangular(
          "mesh is not triangular: face " + std::to_string(i) + " of " +
          path.string() + " has " + std::to_string(f.size()) +
          " vertices. Triangulate it first.");
    }
    for (int c = 0; c < 3; ++c) {
      fi->V(c) = &mesh.vert[f[static_cast<std::size_t>(c)]];
    }
  }
}

void write(const fs::path& path, Mesh& mesh)
{
  vcg::tri::Allocator<Mesh>::CompactEveryVector(mesh);

  educelab::Mesh3f dst;
  for (const auto& v : mesh.vert) {
    dst.insert_vertex(v.cP()[0], v.cP()[1], v.cP()[2]);
  }
  for (const auto& f : mesh.face) {
    dst.insert_face(
        static_cast<std::size_t>(vcg::tri::Index(mesh, f.cV(0))),
        static_cast<std::size_t>(vcg::tri::Index(mesh, f.cV(1))),
        static_cast<std::size_t>(vcg::tri::Index(mesh, f.cV(2))));
  }
  // Binary PLY: roughly half the size, losing nothing on a single-precision
  // mesh, and the search rewrites its winner on every improving round.
  // `write_mesh` dispatches on extension only (educelab/libcore#26).
  if (educelab::to_lower(path.extension().string()) == ".ply") {
    educelab::write_ply(path, dst, educelab::PLYFormat::Binary);
  } else {
    educelab::write_mesh(path, dst);
  }
}

auto condition(Mesh& mesh) -> CleanCounts
{
  CleanCounts counts;
  // RemoveDuplicateVertex reports the duplicates it merged; the degenerate
  // faces that merge leaves behind are counted by the pass after it.
  counts.duplicateVertices =
      vcg::tri::Clean<Mesh>::RemoveDuplicateVertex(mesh, false);
  counts.degenerateFaces = vcg::tri::Clean<Mesh>::RemoveDegenerateFace(mesh);
  counts.unreferencedVertices =
      vcg::tri::Clean<Mesh>::RemoveUnreferencedVertex(mesh);
  vcg::tri::Allocator<Mesh>::CompactEveryVector(mesh);
  return counts;
}

auto countNonManifold(Mesh& mesh) -> ManifoldCounts
{
  ManifoldCounts counts;
  vcg::tri::UpdateTopology<Mesh>::FaceFace(mesh);
  counts.nonManifoldEdges = vcg::tri::Clean<Mesh>::CountNonManifoldEdgeFF(mesh);
  counts.nonManifoldVertices =
      vcg::tri::Clean<Mesh>::CountNonManifoldVertexFF(mesh);
  return counts;
}

auto collapse(Mesh& mesh, double quadricError, std::size_t maxFaces,
              const Geometry& geometry) -> std::size_t
{
  vcg::tri::UpdateBounding<Mesh>::Box(mesh);

  vcg::tri::TriEdgeCollapseQuadricParameter params;
  params.QualityThr = geometry.qualityThreshold;
  params.PreserveBoundary = geometry.preserveBoundary;
  params.PreserveTopology = geometry.preserveTopology;
  params.NormalCheck = geometry.normalCheck;
  params.OptimalPlacement = geometry.optimalPlacement;
  // Off, and not a knob: the normalization would make the threshold in the
  // report incomparable between meshes. See Geometry.
  params.ScaleIndependent = false;
  // Not exposed: measured rather than predicted, these only shape collapse
  // order, and sane face aspect ratios are worth more than a knob.
  params.QualityQuadric = true;

  vcg::LocalOptimization<Mesh> session(mesh, &params);
  session.Init<TriEdgeCollapse>();
  if (maxFaces > 0) {
    session.SetTargetSimplices(static_cast<int>(maxFaces));
  }
  if (quadricError > 0.0) {
    session.SetTargetMetric(quadricError);
  }
  // Bounded so the loop below gets to tick a progress bar between batches.
  session.SetTimeBudget(0.5F);

  // `DoOptimization` returns "the heap is not empty", not "there is more to do",
  // so a reached goal must be re-checked here or it spins forever.
  while (session.DoOptimization()
         and (maxFaces == 0 or mesh.fn > static_cast<int>(maxFaces))
         and (quadricError <= 0.0 or session.currMetric < quadricError)) {
  }

  vcg::tri::Allocator<Mesh>::CompactEveryVector(mesh);
  return static_cast<std::size_t>(mesh.fn);
}

auto measure(Mesh& original, Mesh& decimated, const Measurement& measurement)
    -> Deviation
{
  // Per face of the *coarser* mesh: the fine mesh's face count says nothing
  // about how much surface there is to miss, so scaling with it only costs.
  const auto coarser = std::min(original.fn, decimated.fn);
  const auto wanted = std::max<long long>(
      static_cast<long long>(kMinSamples),
      static_cast<long long>(measurement.samplesPerFace) * coarser);
  const auto sampleNum = static_cast<int>(
      std::min<long long>(wanted, std::numeric_limits<int>::max()));

  // The cutoff past which a sample is thrown away, spanning both meshes. The
  // target's own diagonal is not enough: decimation that deletes a detached
  // fragment shrinks the target's box, and the samples stranded on the deleted
  // geometry -- the ones proving the budget was blown -- would be the ones
  // discarded, leaving a max that says the target was met.
  vcg::tri::UpdateBounding<Mesh>::Box(original);
  vcg::tri::UpdateBounding<Mesh>::Box(decimated);
  auto span = original.bbox;
  span.Add(decimated.bbox);
  const double distBound = span.Diag();

  Deviation deviation;
  // Both directions, because the asymmetry is the point: an erased ridge leaves
  // every decimated point close to the original, and only shows up the other way.
  deviation.originalToDecimated = oneSided(
      original, decimated, sampleNum, measurement.curvatureWeighted, distBound);
  deviation.decimatedToOriginal = oneSided(
      decimated, original, sampleNum, measurement.curvatureWeighted, distBound);
  deviation.max = std::max(deviation.originalToDecimated.max,
                           deviation.decimatedToOriginal.max);
  return deviation;
}

void coarsen(Mesh& mesh, const fs::path& output, const Targets& targets,
             const Geometry& geometry, const Measurement& measurement,
             const Search& search, Report& report, const RoundCallback& onRound)
{
  vcg::tri::UpdateBounding<Mesh>::Box(mesh);
  report.inputVertices = static_cast<std::size_t>(mesh.vn);
  report.inputFaces = static_cast<std::size_t>(mesh.fn);

  int rounds = 0;
  // The coarsest feasible result seen is what gets kept, not the last tried:
  // vcglib guarantees no monotonicity, so this makes a non-monotonic search a
  // loss of optimality rather than a violated bound.
  bool haveBest = false;
  // The budget is under the measurement's own noise floor: even a round that
  // collapses nothing measures over it, so no threshold can satisfy it and the
  // search is finished. The *other* way a budget proves too tight -- the
  // cheapest collapse costing more than an ordinary budget -- is read off
  // `haveBest` at the end, where the reason is written.
  bool hitFloor = false;
  // Whether the deviation search ran at all. The single-round modes -- a raw
  // quadric threshold, or a face budget on its own -- have no rounds to run out
  // of, so a pass-through there means the target itself collapsed nothing.
  bool searched = false;
  Attempt best;
  Deviation bestDeviation;
  std::size_t bestVertices = 0;

  // Always handed to the collapse: the target under `--prefer faces`, and under
  // `--prefer error` a cap that stops it early rather than letting deviation grow.
  const auto faceCap = targets.maxFaces;

  /// One decimate-and-measure round, from a pristine copy of the input. The
  /// winner is written out as soon as it wins rather than held, so peak memory
  /// stays at the pristine mesh plus one candidate.
  auto attempt = [&](double quadricError, bool gateOnError) {
    Mesh candidate;
    vcg::tri::Append<Mesh, Mesh>::MeshCopy(candidate, mesh);

    Attempt round;
    round.quadricError = quadricError;
    round.faces = collapse(candidate, quadricError, faceCap, geometry);
    const auto deviation = measure(mesh, candidate, measurement);
    round.deviation = deviation.max;
    round.feasible =
        not targets.haveError() or deviation.max <= targets.maxError;

    ++rounds;
    report.attempts.push_back(round);
    if (onRound) {
      onRound(round);
    }

    // Coarsest feasible wins. An ungated round is accepted whatever the
    // deviation, while `feasible` still records the verdict for the report.
    const bool accepted = round.feasible or not gateOnError;
    if (accepted and (not haveBest or round.faces < best.faces)) {
      haveBest = true;
      best = round;
      bestDeviation = deviation;
      bestVertices = static_cast<std::size_t>(candidate.vn);
      write(output, candidate);
    }
    return round;
  };

  if (targets.haveQuadric()) {
    // The escape hatch: a raw threshold turns the search off, so one round runs
    // and the error budget, if any, is only reported against.
    const auto round = attempt(targets.quadricError, false);
    report.bound = atFaceCap(round.faces, targets.maxFaces)
                       ? Bound::MaxFaces
                       : Bound::QuadricError;
  } else if (not targets.haveError()
             or (targets.prefer == Prefer::Faces and targets.haveFaces())) {
    // The face budget is the target, being the only one given or the preferred
    // one. Collapse until the count is met, and say so if a budget did not survive.
    attempt(0.0, false);
    report.bound = Bound::MaxFaces;
  } else {
    // The deviation budget is the target and the threshold is the search
    // variable: seed, then let each measurement aim the next probe.
    searched = true;
    double seed = search.quadricSeed;
    if (not(seed > 0.0)) {
      const double area = meanFaceArea(mesh);
      seed = kSeedScale * targets.maxError * targets.maxError * area;
      if (not(seed > 0.0)) {
        // Every face zero-area: the bounding box is the only scale left.
        const double diag = mesh.bbox.Diag();
        seed = kSeedScale * targets.maxError * targets.maxError * diag * diag;
      }
    }

    double feasibleAt = 0.0;    // largest threshold known to be within budget
    double infeasibleAt = 0.0;  // smallest threshold known to be over it
    // The candidate at that infeasible end. Faces fall as the threshold rises
    // and the answer is inside the bracket, so this is a hard floor on the
    // coarsest feasible mesh any further round can find.
    std::size_t infeasibleFaces = 0;
    // The last two rounds, which are the two points the fits below run through.
    double lastThr = 0.0;
    double lastDev = 0.0;
    double lastFaces = 0.0;
    double prevThr = 0.0;
    double prevDev = 0.0;
    double prevFaces = 0.0;
    // Which way the descent is pointing: over budget means go cheaper still.
    bool lastFeasible = false;

    auto record = [&](double thr, const Attempt& r) {
      prevThr = lastThr;
      prevDev = lastDev;
      prevFaces = lastFaces;
      lastThr = thr;
      lastDev = r.deviation;
      lastFaces = static_cast<double>(r.faces);
      lastFeasible = r.feasible;
      if (r.feasible) {
        feasibleAt = std::max(feasibleAt, thr);
      } else if (infeasibleAt == 0.0 or thr < infeasibleAt) {
        infeasibleAt = thr;
        infeasibleFaces = r.faces;
      }
    };

    /// The local slope of deviation against threshold, in log-log space.
    auto exponent = [&]() {
      if (not(prevThr > 0.0) or not(prevDev > 0.0) or not(lastDev > 0.0)
          or prevThr == lastThr) {
        return kAssumedExponent;
      }
      const double p =
          std::log(lastDev / prevDev) / std::log(lastThr / prevThr);
      // A flat, backwards or non-finite fit is noise, not a curve; the prior
      // beats extrapolating it.
      if (not std::isfinite(p) or not(p > 0.0)) {
        return kAssumedExponent;
      }
      return std::clamp(p, kMinExponent, kMaxExponent);
    };

    /// The same slope for the face count, as a positive number: faces fall as
    /// the threshold rises. This curve is the one that stays informative on the
    /// staircase -- two rounds that measured one deviation still collapsed
    /// different numbers of edges -- which is why the descent is steered by it.
    auto faceExponent = [&]() {
      if (not(prevThr > 0.0) or not(prevFaces > 0.0) or not(lastFaces > 0.0)
          or prevThr == lastThr) {
        return kAssumedFaceExponent;
      }
      const double k =
          -std::log(lastFaces / prevFaces) / std::log(lastThr / prevThr);
      if (not std::isfinite(k) or not(k > 0.0)) {
        return kAssumedFaceExponent;
      }
      return std::clamp(k, kMinFaceExponent, kMaxFaceExponent);
    };

    /// The coarsest candidate an unbracketed probe may buy, in faces: it may
    /// not grow the last one by more than `kCostGrowth`, nor leave more than
    /// `kMaxCandidateShare` of the input standing.
    auto affordableFaces = [&](bool capGrowth) {
      double cap = kMaxCandidateShare * static_cast<double>(report.inputFaces);
      if (capGrowth) {
        cap = std::min(cap, kCostGrowth * lastFaces);
      }
      return cap;
    };

    /// The threshold the face-count fit puts a given candidate size at.
    auto thresholdFor = [&](double faces) {
      return lastThr * std::pow(lastFaces / faces, 1.0 / faceExponent());
    };

    /// The threshold the fitted curve puts a deviation of `aim` at: a secant
    /// step along the power law, held off the candidates it cannot afford while
    /// the answer is unbracketed, and confined to the bracket -- regula falsi,
    /// bisecting when the fit points at an end -- once one exists.
    auto nextProbe = [&](double aim, bool costAware) {
      // Two rounds that measured one deviation are on one tread of the
      // staircase, which means the last stride moved nothing and the fit is
      // reading a flat. It is the absence of a datum, not a datum, and both
      // branches below need to know: bracketed or not, the stride has to come
      // from somewhere other than the fit.
      const bool plateau =
          prevDev > 0.0 and lastDev > 0.0
          and std::abs(std::log(lastDev / prevDev)) < kPlateauTol;
      double next = lastThr;
      if (not(lastDev > 0.0)) {
        // No deviation at all: nothing was collapsed, so there is no curve
        // through it. Grow.
        next = lastThr * kMaxStep;
      } else {
        const double step = std::pow(aim / lastDev, 1.0 / exponent());
        if (std::isfinite(step) and step > 0.0) {
          next = lastThr * std::clamp(step, 1.0 / kMaxStep, kMaxStep);
        }
      }
      if (feasibleAt > 0.0 and infeasibleAt > 0.0) {
        // Bracketed: both ends are measured, so the bracket already bounds
        // both the answer and the price, and a trust region on top of it is
        // redundant.
        //
        // Bisect when the fit has nothing to offer. That is the case a fit
        // pointing outside the bracket makes obvious -- there is nothing left
        // to interpolate -- but the *plateau* is the same case wearing a
        // disguise, and it is the one that costs runs. A flat fit points
        // comfortably **inside** the bracket, a few percent along, so regula
        // falsi accepts it and re-measures the same tread; on `PHerc0006Cr05`
        // that read 0.101311 three rounds running, crawled the threshold by
        // 0.903 a round, and was still crawling when the scheduler killed it.
        // Regula falsi degenerating into a crawl is the classic failure of the
        // method and bisection is the classic answer: it closes the bracket
        // geometrically whatever the fit believes.
        const double lo = std::min(feasibleAt, infeasibleAt) * kEndpointMargin;
        const double hi = std::max(feasibleAt, infeasibleAt) / kEndpointMargin;
        if (plateau or not(lo < hi) or not(next > lo) or not(next < hi)) {
          next = std::sqrt(feasibleAt * infeasibleAt);
        }
      } else {
        // Unbracketed, and the deviation fit is all there is. What the descent
        // is held to is therefore price rather than log-distance: the candidate
        // at the bottom of an over-long step is a pass-through, the most
        // expensive mesh there is to measure and the least informative, and a
        // probe that lands over budget brackets just as well for a fraction of
        // it. Unlike a trust region built from two thresholds, a bound on price
        // does not shrink to nothing as the probes converge.
        const double cap = affordableFaces(costAware);
        if (not lastFeasible and lastFaces > 0.0
            and cap < kMinUsefulGrowth * lastFaces) {
          // The share is in reach and every probe the cost model would still
          // allow is a re-measurement of the round just finished. Whether
          // *anything* below is feasible is the only question left, and one
          // probe at the bottom answers it: either it collapses nothing and is
          // still over budget, which ends the search with the right reason, or
          // it is feasible and there is a bracket to interpolate inside.
          next = lastThr / kMaxStep;
        } else if (lastFaces > 0.0 and cap > lastFaces) {
          const double floorThr = thresholdFor(cap);
          if (plateau and not lastFeasible) {
            // Descending on one tread. The fit is measuring the flat and will
            // creep along it a few percent at a time -- rounds that each buy a
            // full measurement and re-read the same number -- so take the whole
            // stride the cost model can afford instead of the fit's inches.
            next = floorThr;
          } else {
            next = std::max(next, floorThr);
          }
        }
        if (plateau and lastFeasible and prevThr > 0.0) {
          // Climbing back toward the budget on one tread, where going coarser
          // only makes the next round cheaper. Nothing to price, so double the
          // stride that just failed to move the deviation.
          next = std::max(next, lastThr * std::pow(lastThr / prevThr, 2.0));
        }
      }
      return next;
    };

    // A round that collapsed nothing and is *still* over budget: no smaller
    // threshold can do better, so the search is finished. Only a budget below
    // the measurement's own floor gets here -- two copies of one mesh sample to
    // about 1e-7 of their scale apart, not to zero -- and reaching Decision 4's
    // answer for that case in two rounds is what this is for.
    auto atFloor = [&](const Attempt& r) {
      return not r.feasible and r.faces == report.inputFaces;
    };

    // Nothing left to probe: the bracket has closed to inside the endpoint
    // margin, so `nextProbe` can only bisect it to a threshold already
    // measured. This is how the ordinary too-tight budget ends -- it brackets
    // the staircase step and then re-measures it -- and it stops any converged
    // search from paying full measurement cost for a threshold that cannot move
    // the answer.
    auto exhausted = [&]() {
      if (not(feasibleAt > 0.0) or not(infeasibleAt > 0.0)) {
        return false;
      }
      const double lo = std::min(feasibleAt, infeasibleAt) * kEndpointMargin;
      const double hi = std::max(feasibleAt, infeasibleAt) / kEndpointMargin;
      return not(lo < hi);
    };

    /// What another round could still win, as a share of the best feasible
    /// result's faces -- and an upper bound on it, not an estimate, because the
    /// candidate at the infeasible end is coarser than any feasible mesh left
    /// to find. One is returned while there is no bracket, since nothing is
    /// then known to be forgone.
    auto remainingGain = [&]() {
      if (not haveBest or infeasibleFaces == 0 or best.faces == 0) {
        return 1.0;
      }
      if (infeasibleFaces >= best.faces) {
        return 0.0;
      }
      return static_cast<double>(best.faces - infeasibleFaces)
             / static_cast<double>(best.faces);
    };

    double probe = seed;
    auto round = attempt(probe, true);
    record(probe, round);
    hitFloor = atFloor(round);

    while (rounds < search.maxRounds and not hitFloor and not exhausted()) {
      if (haveBest and best.deviation >= kCloseEnough * targets.maxError) {
        report.stop = "the result came within a few percent of the budget";
        break;
      }
      // The deviation cannot always get there. It is a staircase in the
      // threshold, so a budget landing in a gap between two steps leaves the
      // test above unreachable by construction, and the search then pays full
      // price per round for whatever face count is left to win. What is left is
      // bounded by the bracket, so price *that* instead: below --min-gain, a
      // further measurement of a multi-million-face mesh buys a few percent of
      // it. Stopping here forgoes coarseness, never the bound.
      if (search.minGain > 0.0 and remainingGain() < search.minGain) {
        report.stop =
            "the bracket had less than --min-gain of the faces left to win";
        break;
      }
      // Last round with nothing feasible yet: aim inside the budget, since
      // landing just over delivers nothing at all. The cost model stands aside
      // for it -- a mesh is worth more than the round that bought it -- bar the
      // guard against buying a pass-through, which would deliver nothing either.
      const bool lastChance =
          rounds == search.maxRounds - 1 and feasibleAt == 0.0;
      probe = nextProbe(targets.maxError * (lastChance ? kSafetyAim : 1.0),
                        not lastChance);
      round = attempt(probe, true);
      record(probe, round);
      hitFloor = atFloor(round);
    }
    if (report.stop.empty()) {
      if (hitFloor) {
        report.stop = "no threshold collapses anything inside the budget";
      } else if (exhausted()) {
        report.stop = "the bracket closed on a single threshold";
      } else {
        report.stop = "the round cap was reached";
      }
    }
    // Which target the result is up against: the face cap only if the collapse
    // stopped there rather than at the budget.
    report.bound = (haveBest and atFaceCap(best.faces, targets.maxFaces))
                       ? Bound::MaxFaces
                       : Bound::MaxError;
  }

  if (not haveBest) {
    // Every round was over budget. Writing the input through is the honest
    // answer -- its deviation is zero, so the invariant holds -- and the reason
    // below says which of the two ways it got here.
    write(output, mesh);
    best = Attempt{0.0, report.inputFaces, 0.0, true};
    bestVertices = report.inputVertices;
    bestDeviation = Deviation{};
  }

  report.quadricError = best.quadricError;
  report.outputFaces = best.faces;
  report.outputVertices = bestVertices;
  report.facesRemoved = report.inputFaces - std::min(best.faces,
                                                     report.inputFaces);
  report.deviation = bestDeviation;
  report.passThrough = report.facesRemoved == 0;

  if (targets.haveError() and report.deviation.max > targets.maxError) {
    report.budgetExceeded = true;
  }
  if (targets.haveFaces() and report.outputFaces > targets.maxFaces) {
    report.faceBudgetMet = false;
  }

  // A searched run that removed no faces got there one of two ways, and only
  // one of them wants more rounds. `hitFloor` is the rarer: a budget under the
  // measurement's noise floor, where even a round that collapses nothing comes
  // back over budget. The ordinary one is a budget below the cheapest
  // collapse's deviation, and `hitFloor` cannot see it -- those rounds collapse
  // nothing, measure about 1e-7 of the mesh's scale, and so are *feasible*.
  // What marks them is that the search kept one: `haveBest` with no faces
  // removed means every threshold cheap enough to stay inside the budget was
  // also too cheap to collapse an edge. Both are answers, not failures, and
  // neither is fixed by raising the round cap.
  const bool tooTight = hitFloor or haveBest;
  if (report.passThrough and targets.haveFaces()
      and targets.maxFaces >= report.inputFaces) {
    // Checked before the deviation cases, which would otherwise blame the error
    // budget: the face cap is handed to every collapse, so one at or above the
    // input's own count reaches the goal before the first collapse and no
    // threshold can move it.
    report.reason =
        "the face budget is already at or above the input's face count, so "
        "nothing needed collapsing; lower --max-faces";
  } else if (report.passThrough and searched and tooTight) {
    // Nothing collapsible inside the budget: the mesh really is as coarse as
    // this budget allows. More rounds cannot help; a larger budget can.
    report.reason =
        "the cheapest available collapse already exceeds the target, so the "
        "input was written through unchanged; raise --max-error to coarsen it";
  } else if (report.passThrough and searched) {
    // The bracket was still open at the round cap; calling the mesh minimal
    // here would send the user to relax a budget that was fine.
    report.reason =
        "the round cap was reached before a threshold inside the target was "
        "found, so the input was written through unchanged; raise --max-rounds";
  } else if (report.passThrough) {
    report.reason =
        "the target collapsed nothing, so the input was written through "
        "unchanged";
  } else if (report.budgetExceeded) {
    report.reason =
        "the face budget was preferred and the deviation budget was exceeded "
        "to meet it";
  } else if (not report.faceBudgetMet and targets.haveError()
             and targets.prefer == Prefer::Error) {
    report.reason =
        "the face budget could not be met inside the deviation budget; relax "
        "one of them or pass --prefer faces";
  } else if (not report.faceBudgetMet) {
    // No budget was in the way, so the collapse itself stopped: the geometry
    // options and this report's non-manifold counts are where to look.
    report.reason =
        "the face budget could not be met: no further collapse was admissible "
        "under the geometry options";
  } else {
    report.reason = "target met";
  }
}

auto selfTest(std::ostream& out) -> bool
{
  // A sphere, because every point's exact distance to it is `|len(p) - r|`, so
  // the measurement can be checked against an answer from outside this code.
  constexpr int kSubdivisions = 6;
  constexpr std::size_t kTargetFaces = 2000;
  constexpr double kTolerance = 0.10;

  Mesh sphere;
  vcg::tri::Sphere<Mesh>(sphere, kSubdivisions);
  vcg::tri::Allocator<Mesh>::CompactEveryVector(sphere);
  vcg::tri::UpdateBounding<Mesh>::Box(sphere);
  out << "self-test: unit sphere, " << sphere.vn << " vertices, " << sphere.fn
      << " faces\n";

  Mesh coarse;
  vcg::tri::Append<Mesh, Mesh>::MeshCopy(coarse, sphere);
  const auto faces = collapse(coarse, 0.0, kTargetFaces, Geometry{});
  out << "self-test: decimated to " << faces << " faces\n";
  if (faces > kTargetFaces) {
    out << "self-test: FAILED -- the face budget was not met\n";
    return false;
  }

  const auto deviation = measure(sphere, coarse, Measurement{});
  out << "self-test: measured max deviation "
      << deviation.decimatedToOriginal.max << " (coarse to fine), "
      << deviation.originalToDecimated.max << " (fine to coarse)\n";

  // The same direction, computed independently of the sampling machinery: a
  // reference sharing its code would agree with it whatever either did.
  constexpr int kLattice = 16;
  double analytic = 0.0;
  for (const auto& f : coarse.face) {
    if (f.IsD()) {
      continue;
    }
    for (int i = 0; i <= kLattice; ++i) {
      for (int j = 0; i + j <= kLattice; ++j) {
        const auto a = float(i) / kLattice;
        const auto b = float(j) / kLattice;
        const auto p = f.cP(0) * a + f.cP(1) * b + f.cP(2) * (1.0F - a - b);
        analytic = std::max(analytic, std::abs(double(p.Norm()) - 1.0));
      }
    }
  }
  out << "self-test: analytic max deviation " << analytic << "\n";

  const auto measured = deviation.decimatedToOriginal.max;
  const auto error = std::abs(measured - analytic) / std::max(analytic, 1e-12);
  if (error > kTolerance) {
    out << "self-test: FAILED -- measured and analytic deviation differ by "
        << (error * 100.0) << "%, over the " << (kTolerance * 100.0)
        << "% tolerance\n";
    return false;
  }

  if (deviation.max + 1e-12 < measured) {
    out << "self-test: FAILED -- the symmetric max is below a one-sided max\n";
    return false;
  }

  out << "self-test: passed (" << (error * 100.0) << "% from analytic)\n";
  return true;
}

}  // namespace pgs::decimate
