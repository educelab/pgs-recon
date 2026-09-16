#pragma once

/**
 * @file
 * @brief Loading a textured mesh, and rendering it through a pinhole camera by
 *        ray-cast.
 *
 * The renderer is `rt_reorder_texture --projection camera`'s loop
 * (`ReorderUnorganizedTexture.cpp`, the camera-mode branch), reproduced so the
 * measured numbers behind ADR 0011 -- 1.81-2.53 px, the 861 px prior basin, the
 * 6% LK gain -- describe this renderer and not a different one. Ray-cast rather
 * than scanline for the same reason, and because the position map then falls
 * out exact: `pos = C + d * dist`, with no `1/z` interpolation, no
 * perspective-correct `uv/z`, and no back-projection to get wrong.
 *
 * One deliberate divergence from RT: the ray directions are **not** undistorted
 * (`ReorderUnorganizedTexture.cpp:1122`), so this renders the ideal pinhole
 * camera rather than the distorted camera's image space. Distortion is handled
 * on the other side, by undistorting the query keypoints before the solve.
 * That makes the two renderers bit-comparable only on an undistorted camera --
 * which is every camera this has been measured on, `k1` being zero on the rig.
 */

#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <educelab/core/types/Mesh.hpp>
#include <educelab/core/types/UVMap.hpp>

#include "localize_camera.hpp"

namespace pgs::localize
{

/// The mesh type: doubles, 3D, default vertex traits. No normals and no vertex
/// colours -- the renderer shades nothing, so neither is read or kept.
using Mesh = educelab::Mesh<double, 3>;
/// Per-wedge UVs carrying the atlas chart index the renderer samples through.
using UVMap = educelab::UVMap<float, 2, educelab::traits::WithChart>;

/**
 * @brief Where a UV coordinate lands in its chart image.
 *
 * `x = u * cols - 0.5`: u = 0 and u = 1 fall on the outer edges of the first
 * and last texels, which is what a texture-mapper means by a UV coordinate and
 * what OpenMVS authored the atlas against.
 *
 * `rt_reorder_texture` uses the corner-aligned `u * (cols - 1)` instead, which
 * puts u = 0 and u = 1 on those texels' *centres* -- a half-texel
 * scale-and-shift from this, and a ramp rather than a constant offset. Both
 * were run over twelve datasets and came out indistinguishable (held-out rms
 * 1.9028 against 1.9173, each better on 7 and 5 of 12, the solved standoff
 * 30 um apart on average), so the correct one is simply the one used. See
 * ADR 0011; a render is therefore not bit-comparable with RT's.
 */
inline auto UVToPixel(double u, double v, const cv::Mat& img) -> cv::Point2f
{
  return {static_cast<float>(u * img.cols - 0.5),
          static_cast<float>(v * img.rows - 0.5)};
}

/// Thrown for a mesh that cannot be rendered, naming why.
class BadMesh : public std::runtime_error
{
public:
  explicit BadMesh(const std::string& what) : std::runtime_error(what) {}
};

/**
 * @brief A mesh and everything needed to sample its texture.
 *
 * The invariant worth stating: **chart i <-> textures[i] <-> texturePaths[i],
 * positionally**. A material that declares no `map_Kd`, and one whose image is
 * missing from disk, both keep an empty slot rather than being skipped --
 * skipping would shift every later chart onto the wrong image, which renders as
 * a plausible picture of the wrong thing.
 */
struct MeshBundle {
  Mesh mesh;
  UVMap uvMap;
  std::vector<cv::Mat> textures;
  std::vector<std::filesystem::path> texturePaths;

  /// The image for chart @p i, or nullptr when there is none. The one place
  /// "out of range or empty means no texture" is decided.
  [[nodiscard]] auto texture_for_chart(std::size_t i) const -> const cv::Mat*;

  /// Whether any chart has a usable image.
  [[nodiscard]] auto has_texture() const -> bool;
};

/**
 * @brief Read a mesh, its UV map and its textures.
 *
 * libcore reads and writes bottom-left `v`; this flips it once, here at the I/O
 * boundary, so in-memory `v` is top-left and sampling needs no flip. Texture
 * decode is `cv::imread`, which handles the atlases the toolchain writes.
 *
 * @throws BadMesh when the file has no vertices, or carries a face that is not
 * a triangle -- the renderer's BVH is built of triangles and a fan
 * triangulation would break the face-to-UV correspondence it samples through.
 */
auto ReadMeshBundle(const std::filesystem::path& path) -> MeshBundle;

/// What one render produced.
struct Render {
  /// The rendered texture, 8-bit greyscale at the camera's resolution. Black
  /// where no surface was hit or where the surface has no usable chart.
  cv::Mat gray;
  /// Per-pixel world XYZ, CV_32FC3, NaN where no surface was hit. NaN and not
  /// zero: a valid sample at the origin has to be distinguishable from a miss.
  cv::Mat position;
  /// Per-pixel perpendicular (optical-axis) depth, CV_32FC1, NaN where no
  /// surface was hit. Slant range divided by ray obliquity, as RT stores it.
  cv::Mat depth;
  /// How many pixels found a surface.
  std::size_t hits{0};
  /// Charts referenced by a hit face that had no usable image, deduped.
  std::vector<std::size_t> missingCharts;
};

/**
 * @brief Render @p bundle through @p cam by ray-cast.
 *
 * One ray per output pixel, through the pixel **centre**, nearest BVH hit wins
 * -- so occlusion is handled and backface culling is moot, a hit being the
 * nearest surface regardless of facing. Per-pixel and therefore trivially
 * parallel: there is no scatter and no depth-test race.
 *
 * @param threads worker threads; 0 autodetects.
 */
auto RenderMesh(const MeshBundle& bundle, const Camera& cam,
                std::size_t threads) -> Render;

/// Render a generated mesh from a known pose and check what came back. Writes
/// what it checked to @p os. See `localize_render.cpp` for what it proves.
auto SelfTestRender(std::ostream& os) -> bool;

}  // namespace pgs::localize
