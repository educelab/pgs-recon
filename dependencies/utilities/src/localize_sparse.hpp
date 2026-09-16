#pragma once

/**
 * @file
 * @brief The sparse backend: match the query against the SfM structure's own
 *        descriptors.
 *
 * This is `openMVG_main_SfM_Localization`'s correspondence source, in process.
 * Three things are different, and they are the three that were worth
 * reimplementing for:
 *
 *   1. **The query is described in memory.** No `-q <dir>` to scan, so the
 *      mask cannot be featurised as a spurious view of its own -- which is what
 *      made `pgs-calibrate --mask` and `--generate-mask` silent no-ops.
 *   2. **`--sfm-transform` is applied to the scene at load**, not to the pose
 *      on the way out. The resection then happens in the mesh's frame from the
 *      start, which is what puts the sparse structure and the render-lifted
 *      points in one coordinate system and lets them share a solve.
 *   3. **K comes from `--camera` and is never fitted**, when one is given.
 *
 * The query must be described the way `matches_dir` was, so the describer is
 * loaded from the scene's own `image_describer.json` and is never a flag.
 * That is also why the sparse path describes the plain 8-bit query rather than
 * the locally normalized one the mesh path matches against: the database
 * regions were computed from ordinary photographs, and a normalized query's
 * descriptors are not commensurable with them.
 */

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <openMVG/cameras/Camera_Intrinsics.hpp>
#include <openMVG/geometry/pose3.hpp>
#include <openMVG/multiview/solver_resection.hpp>

#include "localize_camera.hpp"
#include "localize_match.hpp"

namespace pgs::localize
{

/// Thrown when the scene or its regions cannot be used, naming why.
class BadScene : public std::runtime_error
{
public:
  explicit BadScene(const std::string& what) : std::runtime_error(what) {}
};

/// A solved scene, its regions and the describer that made them. Opaque on
/// purpose -- openMVG's SfM_Data and Regions_Provider are heavy types and
/// nothing outside the sparse backend wants to see them.
struct SparseScene;

/// What the sparse backend found.
struct SparseReport {
  std::size_t detectedQuery{0};
  std::size_t putative{0};
  std::size_t inliers{0};
  std::size_t views{0};
  std::size_t landmarks{0};
  bool resected{false};
};

/**
 * @brief Load a scene and its regions, optionally changing frame at load.
 *
 * @param transform  4x4 similarity; the scene's landmarks and poses (and its
 *                   pose priors) are moved into that frame in memory.
 */
auto LoadSparseScene(const std::filesystem::path& scenePath,
                     const std::filesystem::path& matchesDir,
                     const std::optional<Eigen::Matrix4d>& transform)
    -> std::shared_ptr<SparseScene>;

/// Describe an 8-bit image with the scene's own describer, so the result is
/// commensurable with what `matches_dir` holds.
auto DescribeWithScene(const SparseScene& scene, const cv::Mat& gray8,
                       const cv::Mat& mask)
    -> std::unique_ptr<openMVG::features::Regions>;

/// What a sparse resection produced.
struct SparseResult {
  bool ok{false};
  openMVG::geometry::Pose3 pose;
  /// The inlier 3D<->2D correspondences, in the scene's (possibly transformed)
  /// frame and the query's own pixel coordinates.
  std::vector<Correspondence> correspondences;
};

/**
 * @brief Match the query against the structure and resect.
 *
 * @param intrinsic K, or nullptr to let the solver recover one
 */
auto LocalizeAgainstScene(
    const SparseScene& scene,
    const openMVG::features::Regions& queryRegions,
    const std::shared_ptr<openMVG::cameras::IntrinsicBase>& intrinsic,
    openMVG::resection::SolverType solver, int width, int height,
    double errorMax, SparseReport& report) -> SparseResult;

}  // namespace pgs::localize
