// Center, orient, and scale a mesh using ArUco/ChArUco markers detected in the
// SfM scene images (a replacement for the texture-based pgs-center).
//
// Translation comes from the mesh's oriented-bounding-box center (Eigen PCA);
// orientation and scale come from markers triangulated out of the SfM scene:
//   marker "right" (corner 0->1) -> +X
//   marker "up"    (corner 3->0) -> +Y
//   marker normal  (right x up)  -> +Z
// When markers are unavailable, orientation falls back to the OBB axes
// (--orient-method bbox/auto), mirroring pgs-center's bounding-box calibration.
//
// Outputs (at least one required):
//   -o/--output-mesh   the transformed mesh (requires --input-mesh)
//   --save-transform   a 4x4 float64 .npy similarity transform, compatible with
//                      pgs-center --load-transform and pgs-calibrate/
//                      pgs-retexture --sfm-transform.
//
// The mesh must already be in the SfM coordinate frame.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <educelab/core/io/MeshIO.hpp>
#include <educelab/core/types/Mesh.hpp>
#include <educelab/core/types/UVMap.hpp>
#include <educelab/core/utils/String.hpp>
#include <indicators/progress_bar.hpp>

#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "marker_detection.hpp"
#include "marker_types.hpp"

namespace ar  = cv::aruco;
namespace el  = educelab;
namespace fs  = std::filesystem;
namespace po  = boost::program_options;
using namespace openMVG;
using namespace indicators;
using namespace pgs;

namespace {

// Mesh with per-vertex normals + colors (what we read into so existing normals
// survive the round-trip). libcore's writers omit normals entirely when no
// vertex carries one, so no fabricated zero-normals leak into the output.
struct VTraitsFull : el::traits::WithNormal<double, 3>, el::traits::WithColor {};
using MeshN = el::Mesh<double, 3, VTraitsFull>;
// UV map carrying atlas chart indices so multi-texture meshes round-trip.
using UVMapC = el::UVMap<float, 2, el::traits::WithChart>;

// ---------------------------------------------------------------------------
// Orientation from markers
// ---------------------------------------------------------------------------

struct MarkerOrient {
    bool ok{false};
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
};

// Compute orientation frame (R, mapping world -> canonical) and the centroid of
// all triangulated marker corners. Rows of R are the canonical axes in world
// coords: right -> +X, up -> +Y, normal -> +Z.
auto ComputeMarkerOrientation(const Landmarks &landmarks,
                              const std::set<int> &markerIDs) -> MarkerOrient
{
    auto toEigen = [](const Vec3 &v) { return Eigen::Vector3d(v[0], v[1], v[2]); };
    Eigen::Vector3d right_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d down_sum  = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid  = Eigen::Vector3d::Zero();
    int right_count = 0, down_count = 0, centroid_count = 0;

    for (const auto &mID : markerIDs) {
        std::array<std::optional<Vec3>, 4> c;
        for (int i = 0; i < 4; ++i) {
            auto it = landmarks.find(GetLandmarkID(mID, i));
            if (it != landmarks.end() && it->second.X) {
                c[i] = it->second.X.value();
                centroid += toEigen(*c[i]);
                ++centroid_count;
            }
        }
        // Right: corner 0 (TL) -> 1 (TR), and corner 3 (BL) -> 2 (BR)
        if (c[0] && c[1]) {
            right_sum += (toEigen(*c[1]) - toEigen(*c[0])).normalized();
            ++right_count;
        }
        if (c[3] && c[2]) {
            right_sum += (toEigen(*c[2]) - toEigen(*c[3])).normalized();
            ++right_count;
        }
        // Down: corner 0 (TL) -> 3 (BL), and corner 1 (TR) -> 2 (BR)
        if (c[0] && c[3]) {
            down_sum += (toEigen(*c[3]) - toEigen(*c[0])).normalized();
            ++down_count;
        }
        if (c[1] && c[2]) {
            down_sum += (toEigen(*c[2]) - toEigen(*c[1])).normalized();
            ++down_count;
        }
    }

    MarkerOrient out;
    if (right_count == 0 || down_count == 0) {
        return out;  // ok == false
    }

    const Eigen::Vector3d right_hat = right_sum.normalized();
    const Eigen::Vector3d normal_hat =
        right_hat.cross(-down_sum.normalized()).normalized();
    const Eigen::Vector3d up_hat = normal_hat.cross(right_hat).normalized();

    out.R.row(0) = right_hat;
    out.R.row(1) = up_hat;
    out.R.row(2) = normal_hat;
    if (centroid_count > 0) {
        out.centroid = centroid / centroid_count;
    }
    out.ok = true;
    return out;
}

// ---------------------------------------------------------------------------
// Oriented bounding box (Eigen PCA)
// ---------------------------------------------------------------------------

struct OBB {
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};
    // axes[0..2] are unit vectors sorted by extent (largest first), right-handed
    std::array<Eigen::Vector3d, 3> axes{
        Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()};
    Eigen::Vector3d extents{Eigen::Vector3d::Zero()};  // sorted to match axes
};

template <typename MeshT>
auto ComputeOBB(const MeshT &mesh) -> OBB
{
    const auto n = mesh.num_vertices();
    OBB obb;
    if (n == 0) {
        return obb;
    }

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < n; ++i) {
        const auto &v = mesh.vertex(i);
        mean += Eigen::Vector3d(v[0], v[1], v[2]);
    }
    mean /= static_cast<double>(n);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (std::size_t i = 0; i < n; ++i) {
        const auto &v = mesh.vertex(i);
        const Eigen::Vector3d d = Eigen::Vector3d(v[0], v[1], v[2]) - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(n);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    const Eigen::Matrix3d evec = es.eigenvectors();

    // Project onto eigenvectors to find extents along each principal axis
    Eigen::Vector3d mn = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::max());
    Eigen::Vector3d mx = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::lowest());
    for (std::size_t i = 0; i < n; ++i) {
        const auto &v = mesh.vertex(i);
        const Eigen::Vector3d p =
            evec.transpose() * (Eigen::Vector3d(v[0], v[1], v[2]) - mean);
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }
    const Eigen::Vector3d mid = 0.5 * (mn + mx);
    obb.center = mean + evec * mid;
    const Eigen::Vector3d ext = mx - mn;

    // Sort axes by extent, largest first
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(),
              [&ext](int a, int b) { return ext[a] > ext[b]; });
    for (int i = 0; i < 3; ++i) {
        obb.axes[i] = evec.col(order[i]);
        obb.extents[i] = ext[order[i]];
    }
    // Enforce right-handedness
    if (obb.axes[0].cross(obb.axes[1]).dot(obb.axes[2]) < 0.0) {
        obb.axes[2] = -obb.axes[2];
    }
    return obb;
}

auto BasisVector(char axis) -> Eigen::Vector3d
{
    switch (axis) {
        case 'x': return Eigen::Vector3d::UnitX();
        case 'y': return Eigen::Vector3d::UnitY();
        case 'z': return Eigen::Vector3d::UnitZ();
        default:  return Eigen::Vector3d::UnitX();
    }
}

// Orientation from the OBB axes: map the largest extent to maxDir, the second to
// midDir; resolve the remaining 180-degree ambiguity using surface normals.
auto ComputeBBoxOrientation(const OBB &obb, const MeshN &mesh, char maxDir,
                            char midDir, int flipMax, int flipMid)
    -> Eigen::Matrix3d
{
    const Eigen::Vector3d maxT = flipMax * BasisVector(maxDir);
    const Eigen::Vector3d midT = flipMid * BasisVector(midDir);
    const Eigen::Vector3d minT = maxT.cross(midT);  // right-handed third axis

    Eigen::Matrix3d A;
    A.col(0) = obb.axes[0];
    A.col(1) = obb.axes[1];
    A.col(2) = obb.axes[2];
    Eigen::Matrix3d B;
    B.col(0) = maxT;
    B.col(1) = midT;
    B.col(2) = minT;
    // R maps OBB axis i -> target i (A, B orthonormal & right-handed -> proper)
    Eigen::Matrix3d R = B * A.transpose();

    // 180-degree disambiguation: most surface should face +minT. Needs faces.
    if (mesh.num_faces() == 0) {
        std::cout << "WARNING: mesh has no faces; skipping bounding-box flip "
                     "test (orientation may be ambiguous up to 180 degrees)\n";
        return R;
    }

    const auto n = mesh.num_vertices();
    const std::size_t numSamples = std::min<std::size_t>(n, 1000);
    const std::size_t step = std::max<std::size_t>(1, n / numSamples);
    std::size_t sampled = 0, agree = 0;
    for (std::size_t i = 0; i < n; i += step) {
        if (mesh.vertex_faces(i).empty()) {
            continue;
        }
        const auto nrm = el::vertex_normal(mesh, i);
        const Eigen::Vector3d rn =
            R * Eigen::Vector3d(nrm[0], nrm[1], nrm[2]);
        ++sampled;
        if (rn.dot(minT) > 0.0) {
            ++agree;
        }
    }
    if (sampled > 0 &&
        static_cast<double>(agree) <= 0.4 * static_cast<double>(sampled)) {
        // Flip 180 degrees about the mid axis (negates max & min target dirs)
        const Eigen::Matrix3d flip =
            2.0 * midT * midT.transpose() - Eigen::Matrix3d::Identity();
        R = flip * R;
    }
    return R;
}

// ---------------------------------------------------------------------------
// Transform assembly + IO
// ---------------------------------------------------------------------------

// Build a 4x4 similarity transform matching pgs-center --save-transform:
//   tfm = scale @ rot @ trans   (centers on `center`, rotates, then scales)
auto BuildTransform(const Eigen::Matrix3d &R, const Eigen::Vector3d &center,
                    double scale) -> Eigen::Matrix4d
{
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    trans.block<3, 1>(0, 3) = -center;
    Eigen::Matrix4d rot = Eigen::Matrix4d::Identity();
    rot.block<3, 3>(0, 0) = R;
    Eigen::Matrix4d scl = Eigen::Matrix4d::Identity();
    scl.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    return scl * rot * trans;
}

// Write a 4x4 float64 matrix as NumPy .npy v1.0, C-order, little-endian.
void WriteNpy(const fs::path &path, const Eigen::Matrix4d &mat)
{
    const std::string dictStr =
        "{'descr': '<f8', 'fortran_order': False, 'shape': (4, 4), }";
    const std::size_t minLen    = dictStr.size() + 1;  // +1 for \n
    const std::size_t headerLen = ((minLen + 10 + 63) / 64) * 64 - 10;

    std::string header(headerLen, ' ');
    std::copy(dictStr.begin(), dictStr.end(), header.begin());
    header[headerLen - 1] = '\n';

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write: " + path.string());
    }
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    file.write(reinterpret_cast<const char *>(magic), 8);
    const auto hlen = static_cast<uint16_t>(headerLen);
    file.write(reinterpret_cast<const char *>(&hlen), 2);
    file.write(header.data(), static_cast<std::streamsize>(headerLen));
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const double v = mat(r, c);
            file.write(reinterpret_cast<const char *>(&v), 8);
        }
    }
}

// Apply the 4x4 transform to vertices and rotate normals (if present).
template <typename MeshT>
void ApplyTransform(MeshT &mesh, const Eigen::Matrix4d &tfm,
                    const Eigen::Matrix3d &rot)
{
    using Vertex = typename MeshT::Vertex;
    for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
        auto &v = mesh.vertex(i);
        const Eigen::Vector4d p = tfm * Eigen::Vector4d(v[0], v[1], v[2], 1.0);
        v[0] = p[0];
        v[1] = p[1];
        v[2] = p[2];
        if constexpr (el::traits::has_normal<Vertex>::value) {
            if (v.normal) {
                const auto &nv = v.normal.value();
                const Eigen::Vector3d rn =
                    rot * Eigen::Vector3d(nv[0], nv[1], nv[2]);
                (*v.normal)[0] = rn[0];
                (*v.normal)[1] = rn[1];
                (*v.normal)[2] = rn[2];
            }
        }
    }
}

// Copy the per-chart input textures next to the output mesh, renaming each to
// match the output mesh stem, and return the relative (basename) paths to
// reference from the written mesh/MTL. Naming follows the pipeline convention
// ({stem}.obj -> {stem}.mtl -> {stem}.{ext}): the first texture takes the bare
// stem, subsequent ones get an index suffix ("<stem>1.<ext>", ...), matching
// duplicate_mtllib in wavefront.py.
auto CopyTextures(const std::vector<fs::path> &texPaths, const fs::path &inDir,
                  const fs::path &outDir, const std::string &outStem)
    -> std::vector<fs::path>
{
    std::vector<fs::path> out;
    out.reserve(texPaths.size());
    for (std::size_t i = 0; i < texPaths.size(); ++i) {
        const auto &tex = texPaths[i];
        const fs::path src = inDir / tex;
        if (!fs::exists(src)) {
            std::cout << "WARNING: texture file not found, reference kept as-is: "
                      << src << "\n";
            out.push_back(tex);  // keep the original reference
            continue;
        }
        const std::string dstStem =
            i == 0 ? outStem : outStem + std::to_string(i);
        const fs::path dstName = dstStem + tex.extension().string();
        const fs::path dst = outDir / dstName;
        if (fs::weakly_canonical(src) != fs::weakly_canonical(dst)) {
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
        }
        out.push_back(dstName);
    }
    return out;
}

// Write the (already transformed) mesh, dispatching on extension and texture
// count.
template <typename MeshT>
void WriteOrientedMesh(const fs::path &path, const MeshT &mesh,
                       const UVMapC &uvmap,
                       const std::vector<fs::path> &texPaths)
{
    const auto ext = el::to_lower_copy(path.extension().string());
    const bool hasUV = uvmap.size() > 0;
    if (ext == ".obj") {
        if (!hasUV) {
            el::write_obj(path, mesh);
        } else if (texPaths.empty()) {
            el::write_obj(path, mesh, uvmap);
        } else if (texPaths.size() == 1) {
            el::write_obj(path, mesh, uvmap, texPaths.front());
        } else {
            el::write_obj(path, mesh, uvmap, texPaths);
        }
    } else if (ext == ".ply") {
        if (!hasUV) {
            el::write_ply(path, mesh);
        } else if (texPaths.empty()) {
            el::write_ply(path, mesh, uvmap);
        } else {
            if (texPaths.size() > 1) {
                std::cout << "WARNING: PLY supports a single texture; writing "
                             "only the first of " << texPaths.size() << "\n";
            }
            el::write_ply(path, mesh, uvmap, texPaths.front());
        }
    } else {
        throw std::runtime_error("Unsupported output mesh extension: " + ext);
    }
}

enum EXIT_CODE {
    SUCCESS   = 0,
    HELP      = 1,
    BAD_ARG   = 2,
    NO_VIEWS  = 3,
    NO_LDMS   = 4,
    NO_ORIENT = 5,
    NO_MESH   = 6,
};

} // namespace

auto main(int argc, char *argv[]) -> int
{
    // clang-format off
    po::options_description parser("options");
    parser.add_options()
        ("help,h", "print help message")
        ("input-scene,i",   po::value<std::string>()->required(),
                            "input SfM scene file")
        ("input-mesh",      po::value<std::string>(),
                            "input mesh (obj/ply) in the SfM coordinate frame; "
                            "enables OBB-center translation and bbox orientation")
        ("output-mesh,o",   po::value<std::string>(),
                            "write the transformed mesh (requires --input-mesh)")
        ("save-transform",  po::value<std::string>(),
                            "write the 4x4 similarity transform as a .npy file")
        ("marker-size,s",   po::value<double>(),
                            "ArUco marker size in desired world units (required "
                            "unless --no-scale, --orient-method bbox, or "
                            "--detection-method sample-square, which has a known "
                            "fixed size)")
        ("detection-method,m", po::value<std::string>()->default_value("markers"),
                            "marker detection method: markers, sample-square")
        ("orient-method",   po::value<std::string>()->default_value("auto"),
                            "orientation source: auto, aruco, bbox")
        ("scale-method",    po::value<std::string>()->default_value("umeyama"),
                            R"(scale method: "umeyama" (weighted-median) or )"
                            R"("edge" (median of edge lengths))")
        ("no-scale",        po::bool_switch(),
                            "skip scale computation (scale = 1)")
        ("max-dir",         po::value<std::string>()->default_value("x"),
                            "bbox: axis for the largest OBB edge (x, y, z)")
        ("mid-dir",         po::value<std::string>()->default_value("y"),
                            "bbox: axis for the 2nd largest OBB edge (x, y, z)")
        ("flip-max",        po::bool_switch(),
                            "bbox: invert the largest-edge target axis")
        ("flip-mid",        po::bool_switch(),
                            "bbox: invert the 2nd-largest-edge target axis")
        ("sfm-root",        po::value<std::string>(),
                            "override SfM root path for images")
        ("undistort-images", po::bool_switch(),
                            "undistort images before detection")
        ("detect-inverted", po::bool_switch(),
                            "attempt to detect inverted markers")
        ("min-marker-pix",  po::value<int>()->default_value(32),
                            "minimum marker size in pixels")
        ("no-ransac",       po::bool_switch(), "disable RANSAC triangulation")
        ("save-debug-images", po::value<std::string>(),
                            "save marker detection debug images to this directory")
        ("progress,p",      po::bool_switch(), "show progress bar")
    ;
    // clang-format on

    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, parser), args);
    if (argc == 1 || args.count("help")) {
        std::cout << parser << "\n";
        return HELP;
    }
    po::notify(args);

    const bool noScale      = args["no-scale"].as<bool>();
    const bool undistort    = args["undistort-images"].as<bool>();
    const bool useRansac    = !args["no-ransac"].as<bool>();
    const bool showProgress = args["progress"].as<bool>();
    const bool saveDebug    = args.count("save-debug-images") > 0;

    const bool haveMesh   = args.count("input-mesh") > 0;
    const bool wantMesh   = args.count("output-mesh") > 0;
    const bool wantXform  = args.count("save-transform") > 0;

    // Need at least one output
    if (!wantMesh && !wantXform) {
        std::cerr << "ERROR: nothing to do; pass --output-mesh and/or "
                     "--save-transform\n";
        return BAD_ARG;
    }
    if (wantMesh && !haveMesh) {
        std::cerr << "ERROR: --output-mesh requires --input-mesh\n";
        return BAD_ARG;
    }

    const auto orientMethod =
        el::to_lower_copy(args["orient-method"].as<std::string>());
    if (orientMethod != "auto" && orientMethod != "aruco" &&
        orientMethod != "bbox") {
        std::cerr << "ERROR: --orient-method must be auto, aruco, or bbox\n";
        return BAD_ARG;
    }
    if (orientMethod == "bbox" && !haveMesh) {
        std::cerr << "ERROR: --orient-method bbox requires --input-mesh\n";
        return BAD_ARG;
    }

    auto axisOk = [](const std::string &s) {
        return s == "x" || s == "y" || s == "z";
    };
    const auto maxDirStr = el::to_lower_copy(args["max-dir"].as<std::string>());
    const auto midDirStr = el::to_lower_copy(args["mid-dir"].as<std::string>());
    if (!axisOk(maxDirStr) || !axisOk(midDirStr)) {
        std::cerr << "ERROR: --max-dir/--mid-dir must be x, y, or z\n";
        return BAD_ARG;
    }
    if (maxDirStr == midDirStr) {
        std::cerr << "ERROR: --max-dir and --mid-dir must differ\n";
        return BAD_ARG;
    }
    const char maxDir = maxDirStr[0];
    const char midDir = midDirStr[0];
    const int flipMax = args["flip-max"].as<bool>() ? -1 : 1;
    const int flipMid = args["flip-mid"].as<bool>() ? -1 : 1;

    const auto methodStr =
        el::to_lower_copy(args["detection-method"].as<std::string>());
    std::function<DetectionResult(const cv::Mat &, const ar::DetectorParameters &)> detect;
    if (methodStr == "markers") {
        detect = DetectMarkers;
    } else if (methodStr == "sample-square") {
        detect = DetectSampleSquare;
    } else {
        std::cerr << "ERROR: Unknown detection method: " << methodStr << "\n";
        return BAD_ARG;
    }

    // The PGS sample square is a fixed-size physical target, so its edge length
    // is known a priori; marker-size only needs to be supplied for loose markers.
    static constexpr double kSampleSquareSize = 0.47;
    const bool sampleSquare = methodStr == "sample-square";

    // Marker size is only needed when scale is derived from loose markers
    const bool scaleFromMarkers = !noScale && orientMethod != "bbox";
    if (scaleFromMarkers && !sampleSquare && !args.count("marker-size")) {
        std::cerr << "ERROR: --marker-size is required unless --no-scale, "
                     "--orient-method bbox, or --detection-method sample-square\n";
        return BAD_ARG;
    }
    const double markerSize = args.count("marker-size")
                                  ? args["marker-size"].as<double>()
                                  : (sampleSquare ? kSampleSquareSize : 1.0);
    if (args.count("marker-size") && markerSize <= 0.0) {
        std::cerr << "ERROR: --marker-size must be a positive value\n";
        return BAD_ARG;
    }

    const auto scaleMethod =
        el::to_lower_copy(args["scale-method"].as<std::string>());
    if (scaleMethod != "umeyama" && scaleMethod != "edge") {
        std::cerr << "ERROR: --scale-method must be \"umeyama\" or \"edge\"\n";
        return BAD_ARG;
    }

    ar::DetectorParameters params;
    params.useAruco3Detection     = true;
    params.detectInvertedMarker   = args["detect-inverted"].as<bool>();
    params.cornerRefinementMethod = ar::CORNER_REFINE_SUBPIX;
    const auto minMarkerSize = static_cast<float>(args["min-marker-pix"].as<int>());

    // Load SfM scene
    const fs::path sfmPath = args["input-scene"].as<std::string>();
    sfm::SfM_Data sfmData;
    sfm::Load(sfmData, sfmPath.string(), sfm::ALL);
    std::cout << "Loaded SfM scene: " << sfmData.GetViews().size() << " views, "
              << sfmData.GetPoses().size() << " poses, "
              << sfmData.GetIntrinsics().size() << " intrinsics\n";

    fs::path sfmRoot = sfmData.s_root_path;
    if (args.count("sfm-root")) {
        sfmRoot = args["sfm-root"].as<std::string>();
        std::cout << "Using custom SfM root: " << sfmRoot << "\n";
    }

    fs::path debugDir;
    if (saveDebug) {
        debugDir = args["save-debug-images"].as<std::string>();
        fs::create_directories(debugDir);
    }

    // ---- Marker detection + triangulation (skipped for forced bbox) ----
    std::set<int> markerIDs;
    Landmarks landmarks;
    if (orientMethod != "bbox") {
        sfm::Views views;
        std::copy_if(sfmData.GetViews().begin(), sfmData.GetViews().end(),
                     std::inserter(views, views.end()),
                     [&sfmData](const auto &pair) {
                         return sfmData.IsPoseAndIntrinsicDefined(pair.second.get());
                     });
        if (views.empty()) {
            std::cerr << "ERROR: No views with pose and intrinsic!\n";
            return NO_VIEWS;
        }

        std::size_t viewsWithLandmarks = 0;
        const std::size_t numIters = views.size();
        std::size_t iter = 0;
        std::cout << "Detecting landmarks in " << numIters << " views\n";
        std::unique_ptr<ProgressBar> bar;
        if (showProgress) {
            bar = std::make_unique<ProgressBar>(
                option::BarWidth{50}, option::Start{" ["},
                option::ForegroundColor{Color::unspecified},
                option::MaxProgress{numIters});
        }

        for (const auto &[viewID, view] : views) {
            const fs::path fullPath = sfmRoot / view->s_Img_path;
            auto image = cv::imread(fullPath);
            if (image.empty()) {
                std::cout << "WARNING: Could not load image: " << fullPath << "\n";
                if (bar) bar->tick();
                continue;
            }
            if (undistort) {
                auto cam = sfmData.intrinsics.at(view->id_intrinsic);
                image = UndistortImage(image, cam.get());
            }

            params.minMarkerLengthRatioOriginalImg =
                minMarkerSize / static_cast<float>(std::max(image.rows, image.cols));
            const auto res = detect(image, params);
            if (!res.markerIds.empty()) ++viewsWithLandmarks;

            if (saveDebug && !res.markerIds.empty()) {
                auto dbgImg = image.clone();
                ar::drawDetectedMarkers(dbgImg, res.markerCorners, res.markerIds);
                const auto outFile =
                    debugDir / fs::path(view->s_Img_path).replace_extension("jpg");
                fs::create_directories(outFile.parent_path());
                cv::imwrite(outFile, dbgImg);
            }

            for (std::size_t i = 0; i < res.markerIds.size(); ++i) {
                const int mID = res.markerIds[i];
                markerIDs.insert(mID);
                for (std::size_t k = 0; k < res.markerCorners[i].size(); ++k) {
                    const auto cID = GetLandmarkID(mID, static_cast<int>(k));
                    if (!landmarks.count(cID)) landmarks.insert({cID, {}});
                    auto &ldm = landmarks.at(cID);
                    const auto &pt = res.markerCorners[i][k];
                    ldm.obs.emplace_back(viewID, Vec2{pt.x, pt.y});
                }
            }
            ++iter;
            if (bar) {
                bar->set_option(option::PostfixText{
                    std::to_string(iter) + "/" + std::to_string(numIters)});
                bar->tick();
            }
        }
        std::cout << "Detected " << landmarks.size() << " marker landmarks in "
                  << viewsWithLandmarks << " of " << numIters << " views\n";

        std::size_t numTriangulated = 0;
        double reprojSumSq = 0.0;
        std::size_t inlierObs = 0, totalObs = 0;
        std::cout << "Triangulating landmarks" << (useRansac ? " w/RANSAC" : "") << "\n";
        for (auto &[ldmID, ldm] : landmarks) {
            if (ldm.obs.size() < 3) {
                continue;
            }
            std::vector<RansacObservation> x;
            x.reserve(ldm.obs.size());
            for (const auto &[vid, o] : ldm.obs) {
                const auto view = sfmData.views.at(vid);
                const auto cam  = sfmData.intrinsics.at(view->id_intrinsic);
                const Vec3 pt   = undistort ? (*cam)(o) : (*cam)(cam->get_ud_pixel(o));
                x.emplace_back(o, pt, cam, sfmData.GetPoseOrDie(view.get()));
            }
            auto [ok, X] = useRansac ? TriangulateRansac(x) : Triangulate(x);
            if (!ok) {
                continue;
            }
            ldm.X = X;
            ++numTriangulated;
            for (const auto &ro : x) {
                const double err = ro.cam->residual(ro.pose(X), ro.obs).norm();
                ++totalObs;
                if (err < kReprojThresholdPx) {
                    reprojSumSq += err * err;
                    ++inlierObs;
                }
            }
        }
        std::cout << "Triangulated " << numTriangulated << " of "
                  << landmarks.size() << " landmarks\n";
        if (inlierObs > 0) {
            const double rmse =
                std::sqrt(reprojSumSq / static_cast<double>(inlierObs));
            std::cout << "Reprojection error (RMSE): " << rmse << " px over "
                      << inlierObs << "/" << totalObs << " inlier observations\n";
        }
    }

    // ---- Marker orientation (if applicable) ----
    MarkerOrient mo;
    if (orientMethod != "bbox") {
        mo = ComputeMarkerOrientation(landmarks, markerIDs);
    }

    // ---- Load mesh + compute OBB (if a mesh was given) ----
    MeshN mesh;
    UVMapC uvmap;
    std::vector<fs::path> texPaths;
    OBB obb;
    fs::path meshInPath;
    if (haveMesh) {
        meshInPath = args["input-mesh"].as<std::string>();
        std::cout << "Loading mesh: " << meshInPath << "\n";
        el::read_mesh(meshInPath, mesh, uvmap, texPaths);
        if (mesh.num_vertices() == 0) {
            std::cerr << "ERROR: mesh has no vertices: " << meshInPath << "\n";
            return NO_MESH;
        }
        std::cout << "Mesh: " << mesh.num_vertices() << " vertices, "
                  << mesh.num_faces() << " faces, " << texPaths.size()
                  << " textures\n";
        obb = ComputeOBB(mesh);
    }

    // ---- Resolve orientation method ----
    bool useAruco = false;
    if (orientMethod == "aruco") {
        if (!mo.ok) {
            std::cerr << "ERROR: --orient-method aruco but no usable markers "
                         "were detected/triangulated\n";
            return NO_ORIENT;
        }
        useAruco = true;
    } else if (orientMethod == "bbox") {
        useAruco = false;
    } else {  // auto
        if (mo.ok) {
            useAruco = true;
        } else if (haveMesh) {
            std::cout << "No usable markers; falling back to bounding-box "
                         "orientation\n";
            useAruco = false;
        } else {
            std::cerr << "ERROR: no usable markers and no --input-mesh for "
                         "bounding-box fallback\n";
            return NO_ORIENT;
        }
    }

    // ---- Orientation, scale, center ----
    Eigen::Matrix3d R;
    double scale = 1.0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();

    if (useAruco) {
        R = mo.R;
        if (!noScale) {
            ScaleStats stats = (scaleMethod == "edge")
                ? ComputeEdgeScaleStats(landmarks, markerIDs, markerSize)
                : ComputeUmeyamaScaleStats(landmarks, markerIDs, markerSize);
            if (stats.scales.empty()) {
                std::cerr << "ERROR: Could not estimate scale from markers\n";
                return NO_ORIENT;
            }
            scale = stats.summary;
            std::cout << "Scale factor (" << scaleMethod << "): " << scale << "\n";
        }
        center = haveMesh ? obb.center : mo.centroid;
    } else {
        R = ComputeBBoxOrientation(obb, mesh, maxDir, midDir, flipMax, flipMid);
        center = obb.center;
        if (!noScale) {
            std::cout << "WARNING: scale cannot be derived from a bounding box; "
                         "using scale = 1.0 (pass --no-scale to silence)\n";
        }
    }
    std::cout << "Center: [" << center.transpose() << "]\n";

    const Eigen::Matrix4d tfm = BuildTransform(R, center, scale);
    std::cout << "Transform:\n" << tfm << "\n";

    // ---- Outputs ----
    if (wantXform) {
        const fs::path outPath = args["save-transform"].as<std::string>();
        fs::create_directories(fs::weakly_canonical(outPath).parent_path());
        WriteNpy(outPath, tfm);
        std::cout << "Saved transform: " << outPath << "\n";
    }

    if (wantMesh) {
        const fs::path outPath = args["output-mesh"].as<std::string>();
        fs::create_directories(fs::weakly_canonical(outPath).parent_path());
        const auto outTex = CopyTextures(
            texPaths, meshInPath.parent_path(), outPath.parent_path(),
            outPath.stem().string());
        ApplyTransform(mesh, tfm, R);
        WriteOrientedMesh(outPath, mesh, uvmap, outTex);
        std::cout << "Saved mesh: " << outPath << "\n";
    }

    return SUCCESS;
}
