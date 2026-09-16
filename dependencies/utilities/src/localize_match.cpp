#include "localize_match.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp>
#include <openMVG/image/image_container.hpp>
#include <openMVG/matching/indMatch.hpp>
#include <openMVG/matching/regions_matcher.hpp>

#include "RANSAC.hpp"

namespace feat = openMVG::features;
namespace matching = openMVG::matching;

using namespace pgs::localize;

namespace
{

/// A 2D similarity: `q = s*R*p + t`, carried as the two rows of the 2x3 matrix.
struct Similarity2D {
  double a{1.0};  // s * cos(theta)
  double b{0.0};  // s * sin(theta)
  double tx{0.0};
  double ty{0.0};

  [[nodiscard]] auto apply(const cv::Point2d& p) const -> cv::Point2d
  {
    return {a * p.x - b * p.y + tx, b * p.x + a * p.y + ty};
  }
  [[nodiscard]] auto scale() const -> double { return std::hypot(a, b); }
  [[nodiscard]] auto rotationDeg() const -> double
  {
    return std::atan2(b, a) * 180.0 / M_PI;
  }
};

/// Closed-form least-squares similarity over any number of pairs. The minimal
/// case is two, which is what RANSAC samples.
auto FitSimilarity(const std::vector<Match2D>& pairs)
    -> std::pair<bool, Similarity2D>
{
  if (pairs.size() < 2) {
    return {false, {}};
  }
  cv::Point2d pBar{0.0, 0.0};
  cv::Point2d qBar{0.0, 0.0};
  for (const auto& m : pairs) {
    pBar += m.render;
    qBar += m.query;
  }
  const auto n = static_cast<double>(pairs.size());
  pBar /= n;
  qBar /= n;

  double sxx{0.0};  // sum of dot products
  double sxy{0.0};  // sum of cross products
  double norm{0.0};
  for (const auto& m : pairs) {
    const cv::Point2d p = m.render - pBar;
    const cv::Point2d q = m.query - qBar;
    sxx += p.x * q.x + p.y * q.y;
    sxy += p.x * q.y - p.y * q.x;
    norm += p.x * p.x + p.y * p.y;
  }
  if (norm <= 0.0) {
    // Every sampled point at one location: no orientation to recover.
    return {false, {}};
  }
  Similarity2D model;
  model.a = sxx / norm;
  model.b = sxy / norm;
  if (not std::isfinite(model.a) or not std::isfinite(model.b) or
      model.scale() <= 0.0) {
    return {false, {}};
  }
  // The rotation/scale part maps pBar somewhere; the translation is whatever
  // carries that to qBar.
  model.tx = qBar.x - (model.a * pBar.x - model.b * pBar.y);
  model.ty = qBar.y - (model.b * pBar.x + model.a * pBar.y);
  return {true, model};
}

auto SimilarityResidual(const Match2D& m, const Similarity2D& model) -> double
{
  const auto d = model.apply(m.render) - m.query;
  return std::hypot(d.x, d.y);
}

/// Wrap a cv::Mat of 8-bit greys as an openMVG image, copying.
auto ToOpenMVG(const cv::Mat& gray8) -> openMVG::image::Image<unsigned char>
{
  openMVG::image::Image<unsigned char> out(gray8.cols, gray8.rows);
  for (int y = 0; y < gray8.rows; ++y) {
    const auto* row = gray8.ptr<unsigned char>(y);
    for (int x = 0; x < gray8.cols; ++x) {
      out(y, x) = row[x];
    }
  }
  return out;
}

}  // namespace

namespace
{

/// Any image as single channel, whatever its channel count.
auto ToSingleChannel(const cv::Mat& src) -> cv::Mat
{
  if (src.channels() == 3) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
  }
  if (src.channels() == 4) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    return gray;
  }
  if (src.channels() == 1) {
    return src;
  }
  throw std::runtime_error("image has " + std::to_string(src.channels()) +
                           " channels; expected 1, 3 or 4");
}

}  // namespace

auto pgs::localize::ToGray8(const cv::Mat& src) -> cv::Mat
{
  if (src.empty()) {
    return {};
  }
  const auto gray = ToSingleChannel(src);
  if (gray.depth() == CV_8U) {
    return gray;
  }
  cv::Mat out;
  if (gray.depth() == CV_16U) {
    gray.convertTo(out, CV_8U, 1.0 / 257.0);
  } else {
    // Float or signed: stretch onto the byte range rather than truncate. Only
    // this branch depends on the image's own extremes, and nothing the pipeline
    // hands in reaches it.
    double lo{0.0};
    double hi{0.0};
    cv::minMaxLoc(gray, &lo, &hi);
    const auto span = (hi > lo) ? (hi - lo) : 1.0;
    gray.convertTo(out, CV_8U, 255.0 / span, -255.0 * lo / span);
  }
  return out;
}

auto pgs::localize::ToGray01(const cv::Mat& src) -> cv::Mat
{
  if (src.empty()) {
    return {};
  }
  const auto gray = ToSingleChannel(src);

  cv::Mat out;
  switch (gray.depth()) {
    case CV_8U:
      gray.convertTo(out, CV_32F, 1.0 / 255.0);
      break;
    case CV_16U:
      gray.convertTo(out, CV_32F, 1.0 / 65535.0);
      break;
    case CV_32F:
    case CV_64F: {
      // Already floating point: only rescale when it is plainly not in [0, 1],
      // so a caller's own normalization is not undone.
      double lo{0.0};
      double hi{0.0};
      cv::minMaxLoc(gray, &lo, &hi);
      if (hi > 1.0 or lo < 0.0) {
        const auto span = (hi > lo) ? (hi - lo) : 1.0;
        gray.convertTo(out, CV_32F, 1.0 / span, -lo / span);
      } else {
        gray.convertTo(out, CV_32F);
      }
      break;
    }
    default:
      gray.convertTo(out, CV_32F, 1.0 / 255.0);
      break;
  }
  return out;
}

auto pgs::localize::LocalNormalize(const cv::Mat& gray01, const double sigma,
                                   const double clip) -> cv::Mat
{
  cv::Mat mean;
  cv::GaussianBlur(gray01, mean, {0, 0}, sigma);
  cv::Mat diff = gray01 - mean;
  cv::Mat var;
  cv::GaussianBlur(diff.mul(diff), var, {0, 0}, sigma);
  cv::Mat sd;
  cv::sqrt(var, sd);
  // The epsilon is in the units of an image scaled to [0, 1]: it keeps a
  // perfectly flat neighbourhood from dividing by zero, and it is why ToGray01
  // rescales rather than leaving raw counts.
  sd += 1e-3;

  cv::Mat norm = diff / sd;
  cv::Mat out;
  // [-clip, clip] onto [0, 255], the way the prototype requantized it.
  norm = cv::min(cv::max(norm, -clip), clip);
  norm.convertTo(out, CV_8U, 255.0 / (2.0 * clip), 255.0 * clip / (2.0 * clip));
  return out;
}

auto pgs::localize::DescribeImage(const cv::Mat& gray8, const cv::Mat& mask,
                                  const feat::EDESCRIBER_PRESET preset)
    -> std::unique_ptr<feat::Regions>
{
  feat::SIFT_Anatomy_Image_describer describer;
  if (not describer.Set_configuration_preset(preset)) {
    throw std::runtime_error("unknown describer preset");
  }
  const auto image = ToOpenMVG(gray8);
  std::unique_ptr<openMVG::image::Image<unsigned char>> maskImage;
  if (not mask.empty()) {
    maskImage = std::make_unique<openMVG::image::Image<unsigned char>>(
        ToOpenMVG(mask));
  }
  auto regions = describer.Describe_SIFT_Anatomy(image, maskImage.get());
  return std::unique_ptr<feat::Regions>(regions.release());
}

auto pgs::localize::LiftPosition(const cv::Mat& position, const cv::Point2d& pt,
                                 Eigen::Vector3d& out) -> bool
{
  const auto x0 = static_cast<int>(std::floor(pt.x));
  const auto y0 = static_cast<int>(std::floor(pt.y));
  const auto x1 = x0 + 1;
  const auto y1 = y0 + 1;
  if (x0 < 0 or y0 < 0 or x1 >= position.cols or y1 >= position.rows) {
    return false;
  }
  const auto p00 = position.at<cv::Vec3f>(y0, x0);
  const auto p10 = position.at<cv::Vec3f>(y0, x1);
  const auto p01 = position.at<cv::Vec3f>(y1, x0);
  const auto p11 = position.at<cv::Vec3f>(y1, x1);
  // All four, or none: a keypoint straddling a silhouette would otherwise be
  // lifted onto a position interpolated across empty space.
  for (const auto& p : {p00, p10, p01, p11}) {
    if (not(std::isfinite(p[0]) and std::isfinite(p[1]) and
            std::isfinite(p[2]))) {
      return false;
    }
  }
  const auto fx = pt.x - x0;
  const auto fy = pt.y - y0;
  for (int k = 0; k < 3; ++k) {
    out[k] = p00[k] * (1 - fx) * (1 - fy) + p10[k] * fx * (1 - fy) +
             p01[k] * (1 - fx) * fy + p11[k] * fx * fy;
  }
  return true;
}

auto pgs::localize::CorrespondFromRender(
    const Render& render, const cv::Mat& queryNorm,
    const feat::Regions& queryRegions, const MatchOptions& opts,
    MatchReport& report) -> std::vector<Correspondence>
{
  const auto renderNorm =
      LocalNormalize(ToGray01(render.gray), opts.normalizeSigma,
                     opts.normalizeClip);
  const auto renderRegions = DescribeImage(renderNorm, {}, opts.preset);

  report.detectedRender = renderRegions->RegionCount();
  report.detectedQuery = queryRegions.RegionCount();
  if (report.detectedRender == 0 or report.detectedQuery == 0) {
    return {};
  }

  matching::IndMatches matches;
  matching::DistanceRatioMatch(static_cast<float>(opts.loweRatio), opts.matcher,
                               *renderRegions, queryRegions, matches);
  report.lowePassed = matches.size();
  if (matches.size() < 4) {
    return {};
  }

  std::vector<Match2D> putative;
  putative.reserve(matches.size());
  for (const auto& m : matches) {
    const auto r = renderRegions->GetRegionPosition(m.i_);
    const auto q = queryRegions.GetRegionPosition(m.j_);
    putative.push_back({{r.x(), r.y()}, {q.x(), q.y()}});
  }

  const auto [ok, model] = ransac::RANSAC(
      putative, FitSimilarity, SimilarityResidual, opts.similarityThreshold, 2,
      static_cast<std::size_t>(opts.similarityIterations), opts.seed);
  if (not ok) {
    return {};
  }
  report.similarityScale = model.scale();
  report.similarityRotationDeg = model.rotationDeg();

  std::vector<Match2D> inliers;
  for (const auto& m : putative) {
    if (SimilarityResidual(m, model) < opts.similarityThreshold) {
      inliers.push_back(m);
    }
  }
  report.similarityInliers = inliers.size();
  if (inliers.size() < 4) {
    return {};
  }

  // Lucas-Kanade, query side only: the render-side location is where the
  // position map is sampled, so moving it would move the 3D point too.
  if (opts.refineMatches) {
    std::vector<cv::Point2f> from;
    std::vector<cv::Point2f> to;
    from.reserve(inliers.size());
    to.reserve(inliers.size());
    for (const auto& m : inliers) {
      from.emplace_back(static_cast<float>(m.render.x),
                        static_cast<float>(m.render.y));
      to.emplace_back(static_cast<float>(m.query.x),
                      static_cast<float>(m.query.y));
    }
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        renderNorm, queryNorm, from, to, status, err,
        {opts.lkWindow, opts.lkWindow}, opts.lkLevels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<Match2D> kept;
    kept.reserve(inliers.size());
    for (std::size_t i = 0; i < inliers.size(); ++i) {
      if (status[i] == 0) {
        continue;
      }
      kept.push_back({inliers[i].render, {to[i].x, to[i].y}});
    }
    inliers = std::move(kept);
  }
  report.lkKept = inliers.size();

  std::vector<Correspondence> out;
  out.reserve(inliers.size());
  for (const auto& m : inliers) {
    Correspondence c;
    if (not LiftPosition(render.position, m.render, c.X)) {
      continue;
    }
    c.uv = Eigen::Vector2d(m.query.x, m.query.y);
    out.push_back(c);
  }
  report.lifted = out.size();
  return out;
}
