#include "localize_solve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <openMVG/cameras/Camera_Common.hpp>
#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/cameras/Camera_Pinhole_Brown.hpp>
#include <openMVG/cameras/Camera_Pinhole_Fisheye.hpp>
#include <openMVG/cameras/Camera_Pinhole_Radial.hpp>
#include <openMVG/multiview/projection.hpp>
#include <openMVG/sfm/pipelines/localization/SfM_Localizer.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>

namespace cam = openMVG::cameras;
namespace fs = std::filesystem;
namespace geo = openMVG::geometry;
namespace res = openMVG::resection;
namespace sfm = openMVG::sfm;

using namespace pgs::localize;
using openMVG::Mat;
using openMVG::Mat3;
using openMVG::Mat34;
using openMVG::Vec2;
using openMVG::Vec3;

namespace
{

/// Fill an openMVG resection payload from our correspondences.
void FillMatchData(const std::vector<Correspondence>& corrs,
                   sfm::Image_Localizer_Match_Data& data)
{
  const auto n = static_cast<Eigen::Index>(corrs.size());
  data.pt3D.resize(3, n);
  data.pt2D.resize(2, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const auto& c = corrs[static_cast<std::size_t>(i)];
    data.pt3D.col(i) = c.X;
    data.pt2D.col(i) = c.uv;
  }
}

auto Percentile(std::vector<double> values, const double pct) -> double
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto pos = pct / 100.0 * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = std::min(lo + 1, values.size() - 1);
  const auto frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

}  // namespace

auto pgs::localize::MakeIntrinsic(const Camera& c)
    -> std::shared_ptr<cam::IntrinsicBase>
{
  // Always radial_k3, zeros included: it is the type the downstream readers
  // match on, and a camera measured to have no distortion still has a model.
  return std::make_shared<cam::Pinhole_Intrinsic_Radial_K3>(
      c.width, c.height, c.fx, c.cx, c.cy, c.disto[0], c.disto[1], c.disto[2]);
}

auto pgs::localize::ParseSolver(const std::string& name, const bool haveK)
    -> res::SolverType
{
  if (name == "auto") {
    // P3P when K is known and DLT when it is not. `pgs-calibrate` had to refuse
    // the wrong combination because the stock binary falls back silently; in
    // process there is nothing to fall back from, so just pick correctly.
    return haveK ? res::SolverType::P3P_NORDBERG_ECCV18
                 : res::SolverType::DLT_6POINTS;
  }
  if (name == "dlt") {
    return res::SolverType::DLT_6POINTS;
  }
  if (name == "p3p-ke") {
    return res::SolverType::P3P_KE_CVPR17;
  }
  if (name == "p3p-kneip") {
    return res::SolverType::P3P_KNEIP_CVPR11;
  }
  if (name == "p3p-nordberg") {
    return res::SolverType::P3P_NORDBERG_ECCV18;
  }
  if (name == "p3p-ding") {
    return res::SolverType::P3P_DING_CVPR23;
  }
  if (name == "up2p") {
    return res::SolverType::UP2P_KUKELOVA_ACCV10;
  }
  throw std::runtime_error("unknown --resection-method: " + name);
}

auto pgs::localize::ParseCameraModel(const std::string& name) -> cam::EINTRINSIC
{
  if (name == "pinhole") {
    return cam::PINHOLE_CAMERA;
  }
  if (name == "radial1") {
    return cam::PINHOLE_CAMERA_RADIAL1;
  }
  if (name == "radial3") {
    return cam::PINHOLE_CAMERA_RADIAL3;
  }
  if (name == "radial3tangential") {
    return cam::PINHOLE_CAMERA_BROWN;
  }
  if (name == "fisheye") {
    return cam::PINHOLE_CAMERA_FISHEYE;
  }
  throw std::runtime_error("unknown --camera-model: " + name);
}

namespace
{

auto MakeModel(const cam::EINTRINSIC model, const int w, const int h,
               const double focal, const double ppx, const double ppy)
    -> std::shared_ptr<cam::IntrinsicBase>
{
  switch (model) {
    case cam::PINHOLE_CAMERA:
      return std::make_shared<cam::Pinhole_Intrinsic>(w, h, focal, ppx, ppy);
    case cam::PINHOLE_CAMERA_RADIAL1:
      return std::make_shared<cam::Pinhole_Intrinsic_Radial_K1>(w, h, focal,
                                                                ppx, ppy);
    case cam::PINHOLE_CAMERA_RADIAL3:
      return std::make_shared<cam::Pinhole_Intrinsic_Radial_K3>(w, h, focal,
                                                                ppx, ppy);
    case cam::PINHOLE_CAMERA_BROWN:
      return std::make_shared<cam::Pinhole_Intrinsic_Brown_T2>(w, h, focal, ppx,
                                                               ppy);
    case cam::PINHOLE_CAMERA_FISHEYE:
      return std::make_shared<cam::Pinhole_Intrinsic_Fisheye>(w, h, focal, ppx,
                                                              ppy);
    default:
      throw std::runtime_error("unsupported camera model for DLT resection");
  }
}

}  // namespace

auto pgs::localize::ComplainAboutIntrinsic(const cam::IntrinsicBase& intrinsic)
    -> std::optional<std::string>
{
  const auto params = intrinsic.getParams();
  if (params.size() < 3) {
    return "it carries no focal length and principal point";
  }
  const auto f = params[0];
  const auto cx = params[1];
  const auto cy = params[2];
  const auto w = static_cast<double>(intrinsic.w());
  const auto h = static_cast<double>(intrinsic.h());
  if (not std::isfinite(f) or f <= 0.0) {
    return "its focal length is " + std::to_string(f);
  }
  if (not std::isfinite(cx) or not std::isfinite(cy) or cx < 0.0 or cx > w or
      cy < 0.0 or cy > h) {
    return "its principal point (" + std::to_string(cx) + ", " +
           std::to_string(cy) + ") is outside the " + std::to_string(int(w)) +
           "x" + std::to_string(int(h)) + " image";
  }
  return std::nullopt;
}

auto pgs::localize::SolvePose(
    const std::vector<Correspondence>& corrs,
    const std::shared_ptr<cam::IntrinsicBase>& intrinsic,
    const cam::EINTRINSIC model, const res::SolverType solver, const int width,
    const int height, const double errorMax) -> SolveResult
{
  SolveResult out;
  // Six is DLT's minimum; P3P needs three, but a pose from three points has no
  // consensus to speak of and nothing here wants one.
  if (corrs.size() < 6) {
    return out;
  }

  sfm::Image_Localizer_Match_Data data;
  FillMatchData(corrs, data);
  if (errorMax > 0.0) {
    data.error_max = errorMax;
  }

  geo::Pose3 pose;
  const auto imageSize =
      openMVG::Pair(static_cast<unsigned int>(width),
                    static_cast<unsigned int>(height));
  if (not sfm::SfM_Localizer::Localize(solver, imageSize, intrinsic.get(), data,
                                       pose)) {
    return out;
  }

  auto solved = intrinsic;
  bool refineIntrinsic = false;
  if (not solved) {
    // DLT recovered a projection matrix rather than a pose against a known K.
    // Decompose it, seed the requested model with the focal it found, and let
    // the refinement fit the rest.
    Mat3 K;
    Mat3 R;
    Vec3 t;
    openMVG::KRt_From_P(data.projection_matrix, &K, &R, &t);
    const auto focal = (K(0, 0) + K(1, 1)) / 2.0;
    solved = MakeModel(model, width, height, focal, K(0, 2), K(1, 2));
    refineIntrinsic = true;
    out.fittedIntrinsic = solved;
  }

  // One pass. `b_refine_pose` always; intrinsics only when DLT fitted them.
  if (not sfm::SfM_Localizer::RefinePose(solved.get(), pose, data, true,
                                        refineIntrinsic)) {
    // A refinement that does not improve the RMSE is not a failed solve: the
    // resection's pose still stands, and openMVG returns false for both.
    // Keeping it is what makes a converged-at-the-first-try solve usable.
  }

  out.ok = true;
  out.pose = pose;
  out.inliers.reserve(data.vec_inliers.size());
  for (const auto idx : data.vec_inliers) {
    out.inliers.push_back(static_cast<std::size_t>(idx));
  }

  std::vector<Correspondence> inlierCorrs;
  inlierCorrs.reserve(out.inliers.size());
  for (const auto idx : out.inliers) {
    inlierCorrs.push_back(corrs[idx]);
  }
  const auto stats = ScorePose(pose, *solved, inlierCorrs);
  out.inlierRmsPx = stats.rmsPx;

  double minX = std::numeric_limits<double>::infinity();
  double maxX = -minX;
  double minY = minX;
  double maxY = -minX;
  for (const auto& c : inlierCorrs) {
    minX = std::min(minX, c.uv[0]);
    maxX = std::max(maxX, c.uv[0]);
    minY = std::min(minY, c.uv[1]);
    maxY = std::max(maxY, c.uv[1]);
  }
  if (not inlierCorrs.empty()) {
    out.spreadX = maxX - minX;
    out.spreadY = maxY - minY;
  }
  return out;
}

auto pgs::localize::ScorePose(const geo::Pose3& pose,
                              const cam::IntrinsicBase& intrinsic,
                              const std::vector<Correspondence>& corrs)
    -> ScoreStats
{
  ScoreStats out;
  out.n = corrs.size();
  if (corrs.empty()) {
    return out;
  }

  std::vector<double> errors;
  errors.reserve(corrs.size());
  double sumX{0.0};
  double sumY{0.0};
  for (const auto& c : corrs) {
    const Vec2 proj = intrinsic.project(pose(c.X));
    const auto dx = proj[0] - c.uv[0];
    const auto dy = proj[1] - c.uv[1];
    sumX += dx;
    sumY += dy;
    errors.push_back(std::hypot(dx, dy));
  }
  const auto n = static_cast<double>(corrs.size());
  out.biasXPx = sumX / n;
  out.biasYPx = sumY / n;
  out.biasPx = std::hypot(out.biasXPx, out.biasYPx);
  out.p99Px = Percentile(errors, 99.0);

  // Trimmed at the 98th percentile, matching how the gate thresholds were
  // measured. The untrimmed tail is not discarded -- it is p99Px above.
  const auto cut = Percentile(errors, 98.0);
  double sumSq{0.0};
  std::size_t kept{0};
  for (const auto e : errors) {
    if (e < cut) {
      sumSq += e * e;
      ++kept;
    }
  }
  if (kept < 10) {
    // Too few survived the trim to say anything with: score them all rather
    // than report an RMS over a handful.
    sumSq = 0.0;
    for (const auto e : errors) {
      sumSq += e * e;
    }
    kept = errors.size();
  }
  out.rmsPx = std::sqrt(sumSq / static_cast<double>(kept));
  return out;
}

auto pgs::localize::HeldoutScore(
    const std::vector<Correspondence>& corrs,
    const std::shared_ptr<cam::IntrinsicBase>& intrinsic,
    const cam::EINTRINSIC model, const res::SolverType solver, const int width,
    const int height, const double errorMax, const int splitPx) -> HeldoutResult
{
  HeldoutResult out;
  out.splitPx = splitPx;
  if (splitPx <= 0) {
    return out;
  }

  // A spatial checkerboard, not a random split: neighbouring correspondences
  // share the render patch they came from, so a random half is not held out
  // from the other in any useful sense.
  std::vector<Correspondence> fit;
  std::vector<Correspondence> test;
  for (const auto& c : corrs) {
    const auto bx = static_cast<long long>(std::floor(c.uv[0] / splitPx));
    const auto by = static_cast<long long>(std::floor(c.uv[1] / splitPx));
    ((bx + by) % 2 == 0 ? fit : test).push_back(c);
  }
  out.fitCount = fit.size();
  out.testCount = test.size();
  if (fit.size() < 20 or test.size() < 20) {
    return out;
  }

  const auto solved =
      SolvePose(fit, intrinsic, model, solver, width, height, errorMax);
  if (not solved.ok) {
    return out;
  }
  const auto& used = solved.fittedIntrinsic ? *solved.fittedIntrinsic
                                            : *intrinsic;
  out.stats = ScorePose(solved.pose, used, test);
  out.ok = true;
  return out;
}

void pgs::localize::WriteCalibration(const fs::path& path,
                                     const cam::IntrinsicBase& intrinsic,
                                     const geo::Pose3& pose,
                                     const fs::path& image)
{
  sfm::SfM_Data scene;
  scene.s_root_path = fs::absolute(image).parent_path().string();
  scene.views[0] = std::make_shared<sfm::View>(
      image.filename().string(), 0, 0, 0,
      static_cast<openMVG::IndexT>(intrinsic.w()),
      static_cast<openMVG::IndexT>(intrinsic.h()));
  scene.intrinsics[0] =
      std::shared_ptr<cam::IntrinsicBase>(intrinsic.clone());
  scene.poses[0] = pose;
  // structure and control_points stay empty: the calibration is a camera, not
  // a scene, and every reader of it filters structure out anyway.
  if (not sfm::Save(scene, path.string(), sfm::ESfM_Data(sfm::ALL))) {
    throw std::runtime_error("could not write calibration: " + path.string());
  }
}

auto pgs::localize::CameraFromSolve(const Camera& base,
                                    const cam::IntrinsicBase& intrinsic,
                                    const geo::Pose3& pose) -> Camera
{
  Camera out = base;
  out.width = static_cast<int>(intrinsic.w());
  out.height = static_cast<int>(intrinsic.h());

  const auto params = intrinsic.getParams();
  if (params.size() >= 3) {
    out.fx = out.fy = params[0];
    out.cx = params[1];
    out.cy = params[2];
  }
  // Only the radial models' trailing parameters are k1..k3. A fisheye's are
  // four coefficients of a different series and a Brown's carry two tangential
  // terms after the radial ones; neither is a radial k1..k3, so neither is
  // copied into one.
  out.disto = {{0.0, 0.0, 0.0}};
  const auto type = intrinsic.getType();
  if (type == cam::PINHOLE_CAMERA_RADIAL1 or
      type == cam::PINHOLE_CAMERA_RADIAL3 or
      type == cam::PINHOLE_CAMERA_BROWN) {
    for (std::size_t i = 0; i < 3 and i + 3 < params.size(); ++i) {
      out.disto[i] = params[i + 3];
    }
  }
  out.haveDisto = true;
  out.set_pose_from_center(pose.rotation(), pose.center());
  return out;
}
