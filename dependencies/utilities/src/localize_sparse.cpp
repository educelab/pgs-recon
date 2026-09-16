#include "localize_sparse.hpp"

#include "localize_describer.hpp"

#include <fstream>
#include <iostream>

#include <openMVG/features/image_describer.hpp>
#include <openMVG/geometry/Similarity3.hpp>
#include <openMVG/image/image_container.hpp>
#include <openMVG/sfm/pipelines/localization/SfM_Localizer_Single_3DTrackObservation_Database.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_transform.hpp>

namespace feat = openMVG::features;
namespace fs = std::filesystem;
namespace geo = openMVG::geometry;
namespace sfm = openMVG::sfm;

using namespace pgs::localize;

namespace pgs::localize
{

struct SparseScene {
  sfm::SfM_Data scene;
  std::shared_ptr<sfm::Regions_Provider> regions;
  std::unique_ptr<feat::Image_describer> describer;
  sfm::SfM_Localization_Single_3DTrackObservation_Database localizer;
};

}  // namespace pgs::localize

namespace
{

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

auto pgs::localize::LoadSparseScene(
    const fs::path& scenePath, const fs::path& matchesDir,
    const std::optional<Eigen::Matrix4d>& transform)
    -> std::shared_ptr<SparseScene>
{
  auto out = std::make_shared<SparseScene>();

  if (not sfm::Load(out->scene, scenePath.string(),
                    sfm::ESfM_Data(sfm::ALL))) {
    throw BadScene("could not load the scene: " + scenePath.string());
  }
  if (out->scene.GetPoses().empty() or out->scene.GetLandmarks().empty()) {
    throw BadScene(scenePath.string() +
                   " has no solved poses or no structure; there is nothing to "
                   "localize against");
  }

  if (transform) {
    // sim(X) = s * R * (X - C), so a 4x4 [sR | t] is R = M/s, C = -R^T t / s.
    // Applied to the scene rather than to the pose on the way out: the
    // resection then happens in the target frame from the start, and the
    // structure ends up in the same frame the render-lifted points are in.
    const Eigen::Matrix3d m = transform->block<3, 3>(0, 0);
    const Eigen::Vector3d t = transform->block<3, 1>(0, 3);
    const auto scale = m.col(0).norm();
    if (scale <= 0.0) {
      throw BadScene("--sfm-transform has a degenerate rotation block");
    }
    const Eigen::Matrix3d r = m / scale;
    if ((r.transpose() * r - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
        1e-6) {
      throw BadScene("--sfm-transform is not a similarity: its 3x3 block is "
                     "not a rotation times a uniform scale");
    }
    const Eigen::Vector3d c = -r.transpose() * t / scale;
    // transform_priors = true: pgs-recon scenes carry rig pose priors, and a
    // prior left in the old frame is a constraint pulling the wrong way.
    sfm::ApplySimilarity(geo::Similarity3(geo::Pose3(r, c), scale), out->scene,
                         true);
    std::cout << "  scene moved into the transform's frame (scale " << scale
              << ")\n";
  }

  // The describer that made matches_dir. Loaded from the scene's own record of
  // it and never a flag: the query's descriptors have to be commensurable with
  // the database's, and the database cannot be re-described.
  try {
    out->describer = LoadDescriber(matchesDir / "image_describer.json");
  } catch (const std::exception& e) {
    throw BadScene(e.what());
  }

  auto regionType = out->describer->Allocate();
  out->regions = std::make_shared<sfm::Regions_Provider>();
  if (not out->regions->load(out->scene, matchesDir.string(), regionType)) {
    throw BadScene("could not load the scene's regions from " +
                   matchesDir.string() +
                   "; a published dataset whose intermediate archive was "
                   "pruned of .feat/.desc cannot be localized against by the "
                   "sparse backend");
  }
  if (not out->localizer.Init(out->scene, *out->regions)) {
    throw BadScene("could not build the 3D point descriptor database");
  }
  return out;
}

auto pgs::localize::DescribeWithScene(const SparseScene& scene,
                                      const cv::Mat& gray8, const cv::Mat& mask)
    -> std::unique_ptr<feat::Regions>
{
  const auto image = ToOpenMVG(gray8);
  std::unique_ptr<openMVG::image::Image<unsigned char>> maskImage;
  if (not mask.empty()) {
    maskImage = std::make_unique<openMVG::image::Image<unsigned char>>(
        ToOpenMVG(mask));
  }
  return scene.describer->Describe(image, maskImage.get());
}

auto pgs::localize::LocalizeAgainstScene(
    const SparseScene& scene, const feat::Regions& queryRegions,
    const std::shared_ptr<openMVG::cameras::IntrinsicBase>& intrinsic,
    const openMVG::resection::SolverType solver, const int width,
    const int height, const double errorMax, SparseReport& report)
    -> SparseResult
{
  SparseResult out;
  report.detectedQuery = queryRegions.RegionCount();
  report.views = scene.scene.GetViews().size();
  report.landmarks = scene.scene.GetLandmarks().size();

  sfm::Image_Localizer_Match_Data data;
  if (errorMax > 0.0) {
    data.error_max = errorMax;
  }
  const auto imageSize =
      openMVG::Pair(static_cast<unsigned int>(width),
                    static_cast<unsigned int>(height));
  geo::Pose3 pose;
  if (not scene.localizer.Localize(solver, imageSize, intrinsic.get(),
                                   queryRegions, pose, &data)) {
    report.putative = static_cast<std::size_t>(data.pt2D.cols());
    return out;
  }
  report.putative = static_cast<std::size_t>(data.pt2D.cols());
  report.inliers = data.vec_inliers.size();
  report.resected = true;

  out.ok = true;
  out.pose = pose;
  out.correspondences.reserve(data.vec_inliers.size());
  for (const auto idx : data.vec_inliers) {
    Correspondence c;
    c.X = data.pt3D.col(idx);
    c.uv = data.pt2D.col(idx);
    out.correspondences.push_back(c);
  }
  return out;
}
