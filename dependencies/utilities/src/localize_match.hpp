#pragma once

/**
 * @file
 * @brief Finding 3D<->2D correspondences between a render and a query image,
 *        and the pose solve they feed.
 *
 * The chain is: normalize both images locally, describe both with openMVG's
 * SIFT, match brute-force with a Lowe ratio, filter with a 2D-similarity
 * RANSAC, refine the query-side location with Lucas-Kanade, and lift each
 * surviving render-side keypoint to 3D through the render's own position map.
 *
 * Two steps are worth stating outright because they look optional and are not:
 *
 *   - **Local normalization is non-optional.** It is what makes "same modality"
 *     a weak enough requirement to be useful: the mesh texture and the query
 *     will have had different processing even in the same band (in the measured
 *     case a 15-knot tone curve against a linear stretch), and without this the
 *     match rate collapses.
 *   - **The similarity RANSAC is only an outlier filter.** The true relation
 *     between render and query is perspective-plus-relief and a similarity
 *     cannot express it. What it is for is rejecting the sample square, which
 *     is not rigid with respect to the sample -- the label is a loose insert and
 *     moves between the photogrammetry and the spectral capture -- while being
 *     the most feature-dense object in the frame.
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <openMVG/features/image_describer.hpp>
#include <openMVG/features/regions.hpp>
#include <openMVG/matching/matcher_type.hpp>

#include "localize_camera.hpp"
#include "localize_render.hpp"

namespace pgs::localize
{

/// Convert any image to single-channel 8-bit. A 16-bit sample is requantized
/// linearly as `round(v / 257)`, which maps 65535 onto 255 exactly -- not a
/// stretch, which would make the result depend on the image's own extremes.
auto ToGray8(const cv::Mat& src) -> cv::Mat;

/// Convert any image to single-channel float in [0, 1]. 16-bit input is scaled
/// by 1/65535 and 8-bit by 1/255, so the normalization epsilon below means the
/// same thing whatever the query's depth.
auto ToGray01(const cv::Mat& src) -> cv::Mat;

/**
 * @brief Subtract the local mean, divide by the local SD, clip, and requantize
 *        to 8 bits.
 *
 * @param gray01 single-channel float image in [0, 1]
 * @param sigma  Gaussian sigma, in pixels, of both local statistics
 * @param clip   how many local SDs to keep; the output maps [-clip, clip] onto
 *               [0, 255]
 */
auto LocalNormalize(const cv::Mat& gray01, double sigma, double clip) -> cv::Mat;

/// One putative or surviving 2D match between the render and the query.
struct Match2D {
  cv::Point2d render;
  cv::Point2d query;
};

/// One 3D<->2D correspondence ready for a resection.
struct Correspondence {
  /// World position, lifted from the render's position map.
  Eigen::Vector3d X;
  /// Query pixel, in the query's own (distorted) image coordinates.
  Eigen::Vector2d uv;
};

/// What the matching stage found, and where it lost things.
struct MatchReport {
  std::size_t detectedRender{0};
  std::size_t detectedQuery{0};
  std::size_t lowePassed{0};
  std::size_t similarityInliers{0};
  std::size_t lkKept{0};
  std::size_t lifted{0};
  /// Rotation, in degrees, of the fitted similarity. Not a gate: a large value
  /// on a prior that should be close is a legible symptom, nothing more.
  double similarityRotationDeg{0.0};
  double similarityScale{1.0};
};

/// How to find and filter matches. Every field is a flag on the CLI.
struct MatchOptions {
  double normalizeSigma{15.0};
  double normalizeClip{3.0};
  openMVG::features::EDESCRIBER_PRESET preset{
      openMVG::features::HIGH_PRESET};
  openMVG::matching::EMatcherType matcher{openMVG::matching::BRUTE_FORCE_L2};
  double loweRatio{0.85};
  double similarityThreshold{6.0};
  int similarityIterations{50000};
  bool refineMatches{true};
  int lkWindow{21};
  int lkLevels{3};
  std::optional<unsigned int> seed;
};

/**
 * @brief Describe an 8-bit image with openMVG's SIFT.
 *
 * openMVG's describer and not `cv::SIFT`: the sparse backend has no choice --
 * the query descriptors must be commensurable with the ones in `matches_dir`,
 * whose type is recorded in `image_describer.json` -- and using one describer
 * for both backends is what lets their correspondences go into one solve.
 *
 * @param mask optional 8-bit mask; features are kept where it is non-zero.
 */
auto DescribeImage(const cv::Mat& gray8, const cv::Mat& mask,
                   openMVG::features::EDESCRIBER_PRESET preset)
    -> std::unique_ptr<openMVG::features::Regions>;

/// A reference image -- a render, or any image with a position map --
/// normalized and described once, so it can be matched against many queries.
struct ReferenceFeatures {
  /// Locally normalized, 8-bit; Lucas-Kanade tracks from it.
  cv::Mat norm;
  std::unique_ptr<openMVG::features::Regions> regions;
};

/**
 * @brief Normalize and describe a reference image once.
 *
 * @param mask optional 8-bit mask; features are kept where it is non-zero.
 *        `Render::textured` is the one to pass when the render has untextured
 *        regions (ADR 0013). Empty keeps every feature, which is what
 *        CorrespondFromRender does.
 */
auto DescribeReference(const Render& reference, const MatchOptions& opts,
                       const cv::Mat& mask = {}) -> ReferenceFeatures;

/**
 * @brief Match a described reference against a query and lift the survivors
 *        through the reference's position map.
 *
 * The matching half of CorrespondFromRender, for a caller that matches one
 * reference against several queries, or the same query at several scales,
 * and should describe the reference only once.
 */
auto CorrespondFromReference(const Render& reference,
                             const ReferenceFeatures& features,
                             const cv::Mat& queryNorm,
                             const openMVG::features::Regions& queryRegions,
                             const MatchOptions& opts, MatchReport& report)
    -> std::vector<Correspondence>;

/**
 * @brief Match a render against a query and lift the survivors to 3D.
 *
 * DescribeReference then CorrespondFromReference, with no mask -- what
 * `pgs-localize` measured in ADR 0011.
 *
 * @param render       the render, its position map, and its depth
 * @param queryNorm    the locally normalized query, 8-bit
 * @param queryRegions the query's regions, described once by the caller and
 *                     reused across passes
 * @param opts         matching options
 * @param report       filled with the counts at every stage
 */
auto CorrespondFromRender(const Render& render, const cv::Mat& queryNorm,
                          const openMVG::features::Regions& queryRegions,
                          const MatchOptions& opts, MatchReport& report)
    -> std::vector<Correspondence>;

/// Bilinear sample of a CV_32FC3 position map, rejecting any location whose
/// four neighbours are not all finite. Returns false for a rejected sample.
auto LiftPosition(const cv::Mat& position, const cv::Point2d& pt,
                  Eigen::Vector3d& out) -> bool;

}  // namespace pgs::localize
