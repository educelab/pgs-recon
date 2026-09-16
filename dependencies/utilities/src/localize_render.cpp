#include "localize_render.hpp"

#include "localize_match.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>

#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <educelab/core/io/MeshIO.hpp>

namespace el = educelab;
namespace fs = std::filesystem;

using namespace pgs::localize;

namespace
{

using Scalar = double;
using Vector3 = bvh::v2::Vec<Scalar, 3>;
using BBox = bvh::v2::BBox<Scalar, 3>;
using Ray = bvh::v2::Ray<Scalar, 3>;
using Triangle = bvh::v2::Tri<Scalar, 3>;
using Node = bvh::v2::Node<Scalar, 3>;
using Bvh = bvh::v2::Bvh<Node>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();

struct HitRecord {
  std::size_t primitiveIdx;
  Scalar distance;
  Scalar u;
  Scalar v;
};

/// Nearest-hit traversal. The returned `primitiveIdx` indexes the BVH's own
/// permuted order and must go through `bvh.prim_ids` before it means a face.
auto IntersectRay(Ray ray, const Bvh& bvh,
                  const std::vector<PrecomputedTri>& tris)
    -> std::optional<HitRecord>
{
  static constexpr auto invalidID = std::numeric_limits<std::size_t>::max();
  static constexpr std::size_t stackSize = 64;
  static constexpr bool useRobustTraversal = true;

  auto primId = invalidID;
  Scalar u{0.0};
  Scalar v{0.0};

  bvh::v2::SmallStack<Bvh::Index, stackSize> stack;
  bvh.intersect<false, useRobustTraversal>(
      ray, bvh.get_root().index, stack,
      [&](const auto begin, const auto end) {
        for (auto i = begin; i < end; ++i) {
          if (auto hit = tris[i].intersect(ray)) {
            primId = i;
            u = hit.value().first;
            v = hit.value().second;
          }
        }
        return primId != invalidID;
      });

  if (primId == invalidID) {
    return std::nullopt;
  }
  // `ray.tmax` is shrunk to the nearest hit by the traversal.
  return HitRecord{primId, ray.tmax, u, v};
}

/// The BVH plus its triangles, already permuted into `bvh.prim_ids` order so
/// the traversal above can index them directly.
struct BVHData {
  Bvh bvh;
  std::vector<PrecomputedTri> precompTris;
};

auto BuildBVH(const Mesh& mesh, bvh::v2::ThreadPool& pool) -> BVHData
{
  std::vector<Triangle> tris;
  tris.reserve(mesh.num_faces());
  for (std::size_t f = 0; f < mesh.num_faces(); ++f) {
    const auto& face = mesh.face(f);
    const auto& a = mesh.vertex(face[0]);
    const auto& b = mesh.vertex(face[1]);
    const auto& c = mesh.vertex(face[2]);
    tris.emplace_back(Vector3(a[0], a[1], a[2]), Vector3(b[0], b[1], b[2]),
                      Vector3(c[0], c[1], c[2]));
  }

  bvh::v2::ParallelExecutor executor(pool);
  std::vector<BBox> bboxes(tris.size());
  std::vector<Vector3> centers(tris.size());
  executor.for_each(0, tris.size(), [&](const auto begin, const auto end) {
    for (auto i = begin; i < end; ++i) {
      bboxes[i] = tris[i].get_bbox();
      centers[i] = tris[i].get_center();
    }
  });

  bvh::v2::DefaultBuilder<Node>::Config config;
  config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
  auto bvh = bvh::v2::DefaultBuilder<Node>::build(pool, bboxes, centers, config);

  std::vector<PrecomputedTri> precompTris(tris.size());
  executor.for_each(0, tris.size(), [&](const auto begin, const auto end) {
    for (auto i = begin; i < end; ++i) {
      precompTris[i] = tris[bvh.prim_ids[i]];
    }
  });

  return BVHData{std::move(bvh), std::move(precompTris)};
}

/// Sample the atlas at a hit, as a greyscale byte.
auto SampleSurface(const cv::Mat& img, const UVMap& uvMap,
                   const std::size_t faceId, const double interU,
                   const double interV) -> unsigned char
{
  cv::Vec3d uvPts[3];
  for (std::size_t corner = 0; corner < 3; ++corner) {
    const auto& uv = uvMap.get_coordinate(faceId, corner);
    uvPts[corner] = cv::Vec3d(uv[0], uv[1], 0.0);
  }

  // bvh's barycentrics are relative to the *second* vertex, so the rotated
  // argument order below is load-bearing.
  const cv::Vec3d bary{interU, interV, 1.0 - interU - interV};
  const cv::Vec3d cPoint =
      bary[0] * uvPts[1] + bary[1] * uvPts[2] + bary[2] * uvPts[0];

  const auto px = UVToPixel(cPoint[0], cPoint[1], img);
  // Bilinear, replicate-border. No wrap and no tiling: a UV outside [0, 1]
  // clamps to the chart's edge rather than reappearing on the far side of it.
  cv::Mat patch;
  cv::getRectSubPix(img, {1, 1}, px, patch, CV_8U);
  return patch.at<unsigned char>(0, 0);
}

}  // namespace

auto MeshBundle::texture_for_chart(const std::size_t i) const -> const cv::Mat*
{
  if (i >= textures.size() or textures[i].empty()) {
    return nullptr;
  }
  return &textures[i];
}

auto MeshBundle::has_texture() const -> bool
{
  return std::any_of(textures.begin(), textures.end(),
                     [](const cv::Mat& m) { return not m.empty(); });
}

auto pgs::localize::ReadMeshBundle(const fs::path& path) -> MeshBundle
{
  MeshBundle out;
  std::vector<fs::path> texturePaths;
  el::read_mesh(path, out.mesh, out.uvMap, texturePaths);

  if (out.mesh.num_vertices() == 0) {
    throw BadMesh("mesh has no vertices: " + path.string());
  }
  if (out.mesh.num_faces() == 0) {
    throw BadMesh("mesh has no faces: " + path.string());
  }
  for (std::size_t f = 0; f < out.mesh.num_faces(); ++f) {
    if (out.mesh.face(f).size() != 3) {
      throw BadMesh("mesh is not triangular: " + path.string() + " face " +
                    std::to_string(f) + " has " +
                    std::to_string(out.mesh.face(f).size()) + " corners");
    }
  }

  // libcore reads the file's bottom-left `v`; flip once, here, so in-memory `v`
  // is top-left and `y = v * (rows - 1)` needs no flip at sample time. The
  // transform is its own inverse, and a writer would apply it again.
  for (std::size_t i = 0; i < out.uvMap.size(); ++i) {
    out.uvMap.at(i)[1] = 1.0F - out.uvMap.at(i)[1];
  }

  // chart i <-> textures[i] <-> texturePaths[i], positionally. A material with
  // no map_Kd arrives as an empty path and keeps an empty slot; so does a
  // referenced image that is not on disk. Never skipped.
  out.textures.reserve(texturePaths.size());
  out.texturePaths.reserve(texturePaths.size());
  for (const auto& rel : texturePaths) {
    if (rel.empty()) {
      out.texturePaths.emplace_back();
      out.textures.emplace_back();
      continue;
    }
    const fs::path texPath = path.parent_path() / rel;
    if (not fs::exists(texPath)) {
      std::cerr << "WARNING: referenced texture not found: " << texPath << "\n";
      out.texturePaths.emplace_back();
      out.textures.emplace_back();
      continue;
    }
    auto img = cv::imread(texPath.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) {
      std::cerr << "WARNING: could not decode texture: " << texPath << "\n";
      out.texturePaths.emplace_back();
      out.textures.emplace_back();
      continue;
    }
    out.texturePaths.push_back(texPath);
    // The renderer samples 8-bit greys; a 16-bit atlas is requantized,
    // which is moot after the local normalization the matcher runs.
    out.textures.push_back(ToGray8(img));
  }
  return out;
}

auto pgs::localize::RenderMesh(const MeshBundle& bundle, const Camera& cam,
                               const std::size_t threads) -> Render
{
  if (cam.width <= 0 or cam.height <= 0) {
    throw BadMesh("cannot render to a " + std::to_string(cam.width) + "x" +
                  std::to_string(cam.height) + " image");
  }

  bvh::v2::ThreadPool pool(threads);
  const auto bvhData = BuildBVH(bundle.mesh, pool);
  const auto& bvh = bvhData.bvh;
  const auto& tris = bvhData.precompTris;

  const int cols = cam.width;
  const int rows = cam.height;

  Render out;
  out.gray = cv::Mat::zeros(rows, cols, CV_8UC1);
  out.depth = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(kNaN));
  out.position = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(kNaN, kNaN, kNaN));

  const Eigen::Matrix3d rInv = cam.R.transpose();
  const Eigen::Vector3d camCenter = cam.center();

  // Far clip from the camera-to-scene distance plus the mesh diagonal, so a
  // ray can always reach the far side of the mesh whatever the standoff.
  Eigen::Vector3d bbMin = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  Eigen::Vector3d bbMax = -bbMin;
  for (std::size_t i = 0; i < bundle.mesh.num_vertices(); ++i) {
    const auto& v = bundle.mesh.vertex(i);
    for (int k = 0; k < 3; ++k) {
      bbMin[k] = std::min(bbMin[k], v[k]);
      bbMax[k] = std::max(bbMax[k], v[k]);
    }
  }
  const auto diag = (bbMax - bbMin).norm();
  const auto far = ((camCenter - 0.5 * (bbMin + bbMax)).norm() + diag) * 2.0;

  const bool haveTexture = bundle.has_texture();

  std::mutex missingMutex;
  std::vector<std::size_t> missing;
  std::atomic<std::size_t> hits{0};

  // Per output pixel, so trivially parallel: nothing scatters and there is no
  // depth-test race. Chunked by row.
  bvh::v2::ParallelExecutor executor(pool);
  executor.for_each(0, static_cast<std::size_t>(rows),
                    [&](const auto begin, const auto end) {
    std::vector<std::size_t> localMissing;
    std::size_t localHits{0};
    for (auto row = begin; row < end; ++row) {
      const auto v = static_cast<int>(row);
      auto* grayRow = out.gray.ptr<unsigned char>(v);
      auto* depthRow = out.depth.ptr<float>(v);
      auto* posRow = out.position.ptr<cv::Vec3f>(v);
      for (int u = 0; u < cols; ++u) {
        // The ideal pinhole ray through the pixel centre. Not undistorted:
        // this renders the pinhole camera and the query keypoints are
        // undistorted instead.
        const Eigen::Vector3d dCam((u + 0.5 - cam.cx) / cam.fx,
                                   (v + 0.5 - cam.cy) / cam.fy, 1.0);
        const auto dCamNorm = dCam.norm();
        const Eigen::Vector3d dWorld = rInv * dCam / dCamNorm;

        Ray ray(Vector3(camCenter[0], camCenter[1], camCenter[2]),
                Vector3(dWorld[0], dWorld[1], dWorld[2]), 0.0, far);
        const auto hit = IntersectRay(ray, bvh, tris);
        if (not hit) {
          continue;
        }
        ++localHits;

        const auto dist = hit->distance;
        // Slant range over ray obliquity is perpendicular (optical-axis) depth,
        // which is camera-space Z. RT stores the same.
        depthRow[u] = static_cast<float>(dist / dCamNorm);

        // Exact by construction: no attribute is interpolated to get here.
        const Eigen::Vector3d pos = camCenter + dWorld * dist;
        posRow[u] = cv::Vec3f(static_cast<float>(pos[0]),
                              static_cast<float>(pos[1]),
                              static_cast<float>(pos[2]));

        if (not haveTexture) {
          continue;
        }
        // The traversal's index is into the BVH's permuted order.
        const auto faceId = bvh.prim_ids[hit->primitiveIdx];
        // A face with no UVs still occludes and still lifts a position; it
        // just contributes no colour.
        if (not(bundle.uvMap.has(faceId, 0) and bundle.uvMap.has(faceId, 1) and
                bundle.uvMap.has(faceId, 2))) {
          continue;
        }
        // The face's chart is the one its corner-0 coordinate carries, which is
        // how libcore groups faces by chart on write.
        const auto chart = bundle.uvMap.get_coordinate(faceId, 0).chart;
        const auto* img = bundle.texture_for_chart(chart);
        if (img == nullptr) {
          localMissing.push_back(chart);
          continue;
        }
        grayRow[u] = SampleSurface(*img, bundle.uvMap, faceId, hit->u, hit->v);
      }
    }
    hits += localHits;
    if (not localMissing.empty()) {
      const std::lock_guard<std::mutex> lock{missingMutex};
      missing.insert(missing.end(), localMissing.begin(), localMissing.end());
    }
  });

  out.hits = hits.load();
  // Accumulated and deduped once, at the end. Per-pixel would be millions of
  // lines saying the same thing.
  std::sort(missing.begin(), missing.end());
  missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
  out.missingCharts = std::move(missing);
  return out;
}

namespace
{

/// Write the self-test's scene: a unit plane with a four-quadrant texture, and
/// a small occluder in front of it carrying a second chart.
void WriteSelfTestScene(const fs::path& dir)
{
  fs::create_directories(dir);

  // Four quadrants, top-left / top-right / bottom-left / bottom-right. The
  // values are what the render is checked against, and they differ across both
  // axes so a flip in either one shows up.
  cv::Mat base(256, 256, CV_8UC1);
  base(cv::Rect(0, 0, 128, 128)) = 10;
  base(cv::Rect(128, 0, 128, 128)) = 70;
  base(cv::Rect(0, 128, 128, 128)) = 140;
  base(cv::Rect(128, 128, 128, 128)) = 210;
  cv::imwrite((dir / "base.png").string(), base);

  const cv::Mat front(32, 32, CV_8UC1, cv::Scalar(250));
  cv::imwrite((dir / "front.png").string(), front);

  {
    std::ofstream mtl{dir / "selftest.mtl"};
    mtl << "newmtl material0\nmap_Kd base.png\n";
    mtl << "newmtl material1\nmap_Kd front.png\n";
  }
  {
    std::ofstream obj{dir / "selftest.obj"};
    obj << "mtllib selftest.mtl\n";
    // The plane: z = 0, spanning [-1, 1] in x and y.
    obj << "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n";
    // The occluder: z = 2, spanning [-0.25, 0.25].
    obj << "v -0.25 -0.25 2\nv 0.25 -0.25 2\nv 0.25 0.25 2\nv -0.25 0.25 2\n";
    // Texture coordinates in the file's BOTTOM-LEFT `v` convention: the
    // vertex at world y = -1 gets v = 0. Reading flips them, so the vertex at
    // the top of the image ends up at the top of the texture.
    // One set per material, not one set shared: the chart index rides on the
    // *pool coordinate*, so two materials pointing at the same `vt` would be
    // two charts claiming one coordinate and the later one would win for both.
    obj << "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n";
    obj << "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n";
    obj << "usemtl material0\n";
    obj << "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n";
    obj << "usemtl material1\n";
    obj << "f 5/5 6/6 7/7\nf 5/5 7/7 8/8\n";
  }
}

auto Check(std::ostream& os, const bool ok, const std::string& what) -> bool
{
  os << (ok ? "  ok   " : "  FAIL ") << what << "\n";
  return ok;
}

}  // namespace

auto pgs::localize::SelfTestRender(std::ostream& os) -> bool
{
  const auto dir = fs::temp_directory_path() / "pgs-localize-selftest";
  bool ok = true;
  try {
    WriteSelfTestScene(dir);
    const auto bundle = ReadMeshBundle(dir / "selftest.obj");

    ok &= Check(os, bundle.mesh.num_faces() == 4,
                "4 faces read (got " +
                    std::to_string(bundle.mesh.num_faces()) + ")");
    ok &= Check(os, bundle.textures.size() == 2,
                "2 charts, positionally aligned with their images (got " +
                    std::to_string(bundle.textures.size()) + ")");

    // The UV origin convention, confirmed rather than reasoned about. The file
    // says `vt 0 0` for the corner at world (-1, -1); in memory that must be
    // (0, 1), top-left. If the flip were missing or applied twice this reads
    // (0, 0) and the render below comes out mirrored in v.
    if (not bundle.uvMap.empty()) {
      const auto& c = bundle.uvMap.get_coordinate(0, 0);
      ok &= Check(os,
                  std::abs(c[0] - 0.0F) < 1e-6F and
                      std::abs(c[1] - 1.0F) < 1e-6F,
                  "file `vt 0 0` reads back as top-left (0, 1) (got (" +
                      std::to_string(c[0]) + ", " + std::to_string(c[1]) + "))");
    } else {
      ok &= Check(os, false, "UV map is not empty");
    }

    // Looking straight down at the plane from 10 units up. R = diag(1, -1, -1)
    // is the canonical overhead orientation; with f = 1000 over a 200 px image
    // the plane exactly fills the frame.
    Camera cam;
    cam.fx = cam.fy = 1000.0;
    cam.cx = cam.cy = 100.0;
    cam.width = cam.height = 200;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(1, 1) = -1.0;
    R(2, 2) = -1.0;
    cam.set_pose_from_center(R, Eigen::Vector3d(0.0, 0.0, 10.0));

    const auto render = RenderMesh(bundle, cam, 0);
    ok &= Check(os, render.hits == 200UL * 200UL,
                "every pixel found a surface (got " +
                    std::to_string(render.hits) + " of 40000)");
    ok &= Check(os, render.missingCharts.empty(),
                "every hit face resolved to a chart image");

    // The position map round-trip: project each world position back through
    // the camera it was rendered with and it must land on the pixel centre it
    // came from. This is what says the render and the lift agree -- a frame
    // error here is a rigid motion and moves no residual anywhere downstream.
    double maxReproj = 0.0;
    double maxDepthErr = 0.0;
    for (int v = 0; v < render.position.rows; ++v) {
      for (int u = 0; u < render.position.cols; ++u) {
        const auto& p = render.position.at<cv::Vec3f>(v, u);
        if (not std::isfinite(p[0])) {
          continue;
        }
        const Eigen::Vector3d X(p[0], p[1], p[2]);
        const Eigen::Vector3d xCam = cam.R * X + cam.t;
        const auto pu = cam.fx * xCam[0] / xCam[2] + cam.cx;
        const auto pv = cam.fy * xCam[1] / xCam[2] + cam.cy;
        maxReproj = std::max(maxReproj,
                             std::hypot(pu - (u + 0.5), pv - (v + 0.5)));
        maxDepthErr = std::max(
            maxDepthErr, std::abs(xCam[2] - render.depth.at<float>(v, u)));
      }
    }
    // 1e-3 px, not zero: the position map is CV_32FC3, and a float32 world
    // coordinate reprojects through a 1000 px focal with about 1e-5 px of
    // quantization. Anything that moves this is a convention error, not noise.
    ok &= Check(os, maxReproj < 1e-3,
                "position map reprojects onto its own pixel centres (max " +
                    std::to_string(maxReproj) + " px)");
    ok &= Check(os, maxDepthErr < 1e-4,
                "depth is camera-space Z, not slant range (max error " +
                    std::to_string(maxDepthErr) + ")");

    // The four quadrants, in the render's own orientation. A v flip swaps top
    // for bottom, a u flip swaps left for right, and a transposed sample swaps
    // both diagonally -- each of the three shows up as a different pair here.
    const auto at = [&render](int u, int v) {
      return static_cast<int>(render.gray.at<unsigned char>(v, u));
    };
    ok &= Check(os, at(50, 50) == 10 and at(150, 50) == 70 and
                        at(50, 150) == 140 and at(150, 150) == 210,
                "texture quadrants land where they belong (TL/TR/BL/BR = " +
                    std::to_string(at(50, 50)) + "/" +
                    std::to_string(at(150, 50)) + "/" +
                    std::to_string(at(50, 150)) + "/" +
                    std::to_string(at(150, 150)) + ")");

    // Nearest hit wins, so the occluder is what the centre of the frame sees:
    // its own chart, its own depth, and a surface 2 units nearer.
    ok &= Check(os, at(100, 100) == 250,
                "the nearer surface is the one sampled (got " +
                    std::to_string(at(100, 100)) + ", expected chart 1's 250)");
    ok &= Check(os, std::abs(render.depth.at<float>(100, 100) - 8.0F) < 1e-4F,
                "the occluder's depth is 8, not the plane's 10 (got " +
                    std::to_string(render.depth.at<float>(100, 100)) + ")");
    ok &= Check(os, std::abs(render.depth.at<float>(10, 10) - 10.0F) < 1e-4F,
                "outside the occluder the plane is still at 10 (got " +
                    std::to_string(render.depth.at<float>(10, 10)) + ")");
  } catch (const std::exception& e) {
    os << "  FAIL exception: " << e.what() << "\n";
    ok = false;
  }

  std::error_code ec;
  fs::remove_all(dir, ec);
  os << (ok ? "self-test passed\n" : "self-test FAILED\n");
  return ok;
}
