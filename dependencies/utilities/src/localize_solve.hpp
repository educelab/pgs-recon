#pragma once

/**
 * @file
 * @brief Resecting a camera from 3D<->2D correspondences, scoring the result,
 *        and writing what came out.
 *
 * The solve is openMVG's: a RANSAC resection (`SfM_Localizer::Localize`) to
 * find the pose and its inliers, then one `SfM_Localizer::RefinePose` over
 * them. **One pass.** Re-rendering from the refined pose and solving again was
 * measured over twelve datasets at 1.820 -> 1.821 -> 1.838 px; a second pass is
 * worth keeping only as a guard on a thin first one, which is what
 * `--second-pass-min-inliers` is and why it defaults to off.
 *
 * The query observations stay in the query's own distorted image coordinates
 * all the way through. openMVG undistorts them itself before P3P, and the
 * bundle adjustment behind `RefinePose` models the distortion directly, so
 * undistorting them here would be doing it twice.
 */

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <openMVG/cameras/Camera_Intrinsics.hpp>
#include <openMVG/geometry/pose3.hpp>
#include <openMVG/multiview/solver_resection.hpp>

#include "localize_camera.hpp"
#include "localize_match.hpp"

namespace pgs::localize
{

/// Build the openMVG intrinsic a camera file describes: `pinhole_radial_k3`,
/// which is what `pgs-retexture --calibration` and `repoint_calibration`
/// expect to find in the emitted scene.
auto MakeIntrinsic(const Camera& cam)
    -> std::shared_ptr<openMVG::cameras::IntrinsicBase>;

/**
 * @brief A complaint about an intrinsic that cannot describe a real camera, or
 *        nullopt when it can.
 *
 * Ported from `pgs-calibrate`'s `validate_localized_intrinsic`, which this tool
 * would otherwise have dropped. Only the DLT path can produce one of these: it
 * estimates the focal from the correspondences, and goes degenerate when the
 * scene is near-fronto-parallel or the matches are few or ill-distributed --
 * measured on this rig as a focal of 72459 px against a true 21700 and a
 * principal point at (-65848, 16033) on an 8176 x 6132 image. Such a camera
 * loads fine everywhere downstream and misaligns the texture, so it is refused
 * here rather than emitted.
 */
auto ComplainAboutIntrinsic(const openMVG::cameras::IntrinsicBase& intrinsic)
    -> std::optional<std::string>;

/// A pose and how it was reached.
struct SolveResult {
  bool ok{false};
  openMVG::geometry::Pose3 pose;
  /// Indices into the correspondences handed in.
  std::vector<std::size_t> inliers;
  /// RMS reprojection error over the inliers, in pixels. Not the held-out
  /// statistic and not comparable with the published gates.
  double inlierRmsPx{0.0};
  /// Extent of the inliers across the query frame. A solve constrained to one
  /// corner is a different thing from the same count spread over the frame.
  double spreadX{0.0};
  double spreadY{0.0};
  /// Set when the intrinsic was recovered by DLT rather than supplied.
  std::shared_ptr<openMVG::cameras::IntrinsicBase> fittedIntrinsic;
};

/**
 * @brief Resect a camera from correspondences.
 *
 * @param intrinsic  the known K, or nullptr to let DLT recover one
 * @param model      the intrinsic model DLT fits (ignored when @p intrinsic is
 *                   given)
 * @param solver     absolute pose solver
 * @param errorMax   residual bound in pixels; 0 leaves openMVG's own
 */
auto SolvePose(const std::vector<Correspondence>& corrs,
               const std::shared_ptr<openMVG::cameras::IntrinsicBase>& intrinsic,
               openMVG::cameras::EINTRINSIC model,
               openMVG::resection::SolverType solver, int width, int height,
               double errorMax) -> SolveResult;

/// Reprojection statistics of one pose over a set of correspondences.
struct ScoreStats {
  std::size_t n{0};
  /// RMS over the residuals below their own 98th percentile. Trimmed because
  /// the published gate thresholds were measured that way; the untrimmed tail
  /// is reported separately as @ref p99Px.
  double rmsPx{0.0};
  double p99Px{0.0};
  /// Norm of the *mean* residual vector -- the systematic part, which is what
  /// discriminates a wrong solve from a noisy one. Reported alongside its
  /// components, which say which way it leans.
  double biasPx{0.0};
  double biasXPx{0.0};
  double biasYPx{0.0};
};

auto ScorePose(const openMVG::geometry::Pose3& pose,
               const openMVG::cameras::IntrinsicBase& intrinsic,
               const std::vector<Correspondence>& corrs) -> ScoreStats;

/// What a held-out split measured.
struct HeldoutResult {
  bool ok{false};
  int splitPx{0};
  std::size_t fitCount{0};
  std::size_t testCount{0};
  ScoreStats stats;
};

/**
 * @brief Solve on one colour of a spatial checkerboard and score on the other.
 *
 * The published gate thresholds are measured this way and do not transfer to a
 * plain inlier RMS, which is a different statistic over a set the solve chose.
 * It is called *held-out* and not *independent*: the matches were still found
 * using a render from the pose being scored, so this bounds the fit, not the
 * registration.
 */
auto HeldoutScore(const std::vector<Correspondence>& corrs,
                  const std::shared_ptr<openMVG::cameras::IntrinsicBase>& intrinsic,
                  openMVG::cameras::EINTRINSIC model,
                  openMVG::resection::SolverType solver, int width, int height,
                  double errorMax, int splitPx) -> HeldoutResult;

/**
 * @brief Write the one-view openMVG calibration.
 *
 * Through openMVG's own cereal serializer rather than hand-built JSON, which
 * sidesteps `fix_polymorphic_registration` entirely: the registration bit and
 * the type name land where cereal wants them because cereal wrote them. One
 * view, one intrinsic, one extrinsic, empty structure -- the shape
 * `pgs-retexture --calibration` and `repoint_calibration` read.
 */
void WriteCalibration(const std::filesystem::path& path,
                      const openMVG::cameras::IntrinsicBase& intrinsic,
                      const openMVG::geometry::Pose3& pose,
                      const std::filesystem::path& image);

/// The camera file form of a solved pose, ready for `WriteCameraFile`.
auto CameraFromSolve(const Camera& base,
                     const openMVG::cameras::IntrinsicBase& intrinsic,
                     const openMVG::geometry::Pose3& pose) -> Camera;

/// Parse a resection solver name. "auto" resolves against @p haveK.
auto ParseSolver(const std::string& name, bool haveK)
    -> openMVG::resection::SolverType;

/// Parse an intrinsic model name for the DLT path.
auto ParseCameraModel(const std::string& name) -> openMVG::cameras::EINTRINSIC;

}  // namespace pgs::localize
