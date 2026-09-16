#pragma once

/**
 * @file
 * @brief The flat camera file: a pinhole K, optional radial distortion, and a
 *        world-to-camera pose.
 *
 * One file carries both because the existing format already does, and because
 * this tool's output has to be a valid input to its own next run: a solved
 * camera seeds the next localization as its prior. `pgs-retexture` and
 * `rt_reorder_texture --camera-file` read the same format.
 *
 *     fx 21700.2
 *     fy 21700.2
 *     cx 4000.9
 *     cy 3591.2
 *     width 8176
 *     height 6132
 *     k1 0
 *     k2 0
 *     k3 0
 *     pose r00 r01 r02 t0 r10 ... 0 0 0 1
 *     expected_standoff 154.2
 *     standoff_tolerance 1.0
 *
 * The two standoff keys are optional and describe the *rig*, not this camera:
 * how far the scene should sit down the optical axis, and how far it may drift
 * before the gate complains. They live here for the same reason K does -- they
 * have the same lifetime, they are re-measured when the table is re-set, and
 * `pose`'s own `t_z` is already that distance, so a caller passing the number
 * on a command line would be keeping a second copy of it free to drift. The
 * CLI flags override them.
 *
 * `pose` is 16 row-major floats of the 4x4 **world-to-camera** matrix, so its
 * translation is `t` in `x_cam = R*X + t` and **not** the camera centre. The two
 * reconcile as `C = -R^T t`, and a file whose `t` reads (2.67, -2.75, 154.16)
 * describes a camera at C = (-2.67, -2.75, 154.16). Misreading one for the other
 * is a proper rigid motion that no residual can reveal, which is why
 * `ReadCameraFile` logs the centre it derived.
 */

#include <array>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pgs::localize
{

/// Thrown for a camera file that cannot be used as one, naming why.
class BadCamera : public std::runtime_error
{
public:
  explicit BadCamera(const std::string& what) : std::runtime_error(what) {}
};

/// A pinhole camera with optional radial distortion and a world-to-camera pose.
struct Camera {
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  int width{0};
  int height{0};
  /// Radial k1, k2, k3 in OpenCV/OpenMVG order.
  std::array<double, 3> disto{{0.0, 0.0, 0.0}};
  /// Whether the file carried any distortion key at all, zero or not. A camera
  /// that states k1 = 0 is a measurement; one that states nothing is a gap, and
  /// the emitted calibration's intrinsic type turns on which it was.
  bool haveDisto{false};
  /// Whether @ref R and @ref t came from a file rather than being identity.
  bool havePose{false};
  /// Expected camera-to-scene distance along the optical axis, in the scene's
  /// units, and how far the solve may sit from it before the gate complains.
  /// Zero means the file said nothing; the CLI flags override either way.
  double expectedStandoff{0.0};
  double standoffTolerance{0.0};
  /// World-to-camera rotation: `x_cam = R * X_world + t`.
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t{Eigen::Vector3d::Zero()};

  /// The camera centre in world coordinates, `C = -R^T t`.
  [[nodiscard]] auto center() const -> Eigen::Vector3d
  {
    return -R.transpose() * t;
  }

  /// Set the pose from a camera centre and a world-to-camera rotation.
  void set_pose_from_center(const Eigen::Matrix3d& rot,
                            const Eigen::Vector3d& c)
  {
    R = rot;
    t = -rot * c;
    havePose = true;
  }

  /// Whether any distortion coefficient is non-zero.
  [[nodiscard]] auto distorted() const -> bool
  {
    return disto[0] != 0.0 or disto[1] != 0.0 or disto[2] != 0.0;
  }
};

/**
 * @brief Read a camera file.
 *
 * Requires fx, cx, cy, width and height; fy defaults to fx. `pose`, when
 * present, must be 16 values whose bottom row is (0, 0, 0, 1) and whose
 * rotation block is orthonormal with a positive determinant -- a pose that is
 * not a rigid motion is a silent frame error, so it is refused here.
 * Tangential terms are refused: the model is radial-only.
 */
auto ReadCameraFile(const std::filesystem::path& path) -> Camera;

/// Write a camera file. Distortion keys are written whether zero or not, so a
/// round trip preserves "this camera was measured to have none".
void WriteCameraFile(const std::filesystem::path& path, const Camera& cam);

/**
 * @brief Rescale K to a different image size.
 *
 * A camera file states K at its own `width` x `height`; the query may be a
 * different size. Focal and principal point scale with it, distortion
 * coefficients are dimensionless and ride along unchanged. Refuses a change of
 * aspect ratio, which no scalar rescale can express.
 */
auto RescaleCamera(const Camera& cam, int width, int height) -> Camera;

/// A one-line description of a camera, centre included.
void DescribeCamera(std::ostream& os, const Camera& cam);

}  // namespace pgs::localize
