/**
 * @file
 * @brief `pgs-localize-joint`: the in-process half of localizing a hand-held
 *        view set (ADR 0013).
 *
 * Three subcommands, one per thing that has to happen where openMVG, OpenCV and
 * the renderer live:
 *
 *   - **position-map**: render the mesh from the overhead view's pose once,
 *     keeping the world position per pixel. Every hand-held view's matches
 *     against the overhead image lift through it (Decision 2, item 2).
 *   - **view**: localize one hand-held image. Resect against the overhead
 *     image for a prior, then render-and-match against the IR reference mesh
 *     from that prior -- ADR 0011's chained mode with the overhead view in the
 *     sparse backend's place. Emits the mesh-only calibration and the inlier
 *     correspondences.
 *   - **adjust**: one shared intrinsic, every pose, mesh points fixed, tie
 *     points free (Decision 4), with the focal cost curve and the per-view
 *     divergence that checks the no-movement assumption (Decision 5).
 *   - **qa**: how far each view's image sits from where its calibration says,
 *     measured against the other views, the overhead, and the reference
 *     texture, each labelled for how independent it is (Decision 6).
 *
 * Everything here is built from `localize_core`, the code `pgs-localize` runs
 * and ADR 0011 measured; this file is orchestration over it. The command line
 * is mirrored by `pgs_recon/viewset/tools.py`; the two are edited together.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <openMVG/cameras/Camera_Pinhole_Radial.hpp>
#include <openMVG/geometry/Similarity3.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_transform.hpp>

#include "NpyArray.hpp"
#include "localize_camera.hpp"
#include "localize_match.hpp"
#include "localize_render.hpp"
#include "localize_solve.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace loc = pgs::localize;
namespace sfm = openMVG::sfm;
namespace geo = openMVG::geometry;
namespace cam = openMVG::cameras;

namespace
{

enum EXIT_CODE {
  SUCCESS = 0,
  HELP = 1,
  BAD_ARG = 2,
  IO_ERROR = 3,
  NO_CORRESPONDENCES = 5,
  SOLVE_FAILED = 6
};

/// The intrinsic model every hand-held K is expressed in: the lens has visible
/// edge distortion (ADR 0013 Context), and it is what `pgs-retexture
/// --calibration` reads.
constexpr auto kModel = cam::PINHOLE_CAMERA_RADIAL3;

/// Fewest correspondences a resection is attempted on.
constexpr std::size_t kMinCorrespondences = 6;

// --- argument plumbing -------------------------------------------------------

/// Every path an argument names that must already exist, checked up front so a
/// typo fails in milliseconds rather than after a render.
auto requireFiles(const po::variables_map& args,
                  const std::vector<std::string>& names) -> bool
{
  auto ok = true;
  for (const auto& name : names) {
    if (not args.count(name)) {
      continue;
    }
    const auto& value = args[name].value();
    std::vector<std::string> paths;
    if (const auto* one = boost::any_cast<std::string>(&value)) {
      paths.push_back(*one);
    } else if (const auto* many =
                   boost::any_cast<std::vector<std::string>>(&value)) {
      paths = *many;
    }
    for (const auto& p : paths) {
      if (not fs::is_regular_file(p)) {
        std::cerr << "ERROR: --" << name << " " << p << " does not exist\n";
        ok = false;
      }
    }
  }
  return ok;
}

/// Parse a subcommand's own arguments. Returns an exit code, or -1 to go on.
auto parseOrExit(const std::string& name, const po::options_description& opts,
                 const std::vector<std::string>& argv,
                 const std::vector<std::string>& required,
                 po::variables_map& args) -> int
{
  try {
    po::store(po::command_line_parser(argv).options(opts).run(), args);
    if (args.count("help") or argv.empty()) {
      std::cout << "Usage: pgs-localize-joint " << name << " [options]\n\n"
                << opts << "\n";
      return HELP;
    }
    po::notify(args);
  } catch (const po::error& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return BAD_ARG;
  }
  for (const auto& r : required) {
    if (not args.count(r)) {
      std::cerr << "ERROR: --" << r << " is required\n";
      return BAD_ARG;
    }
  }
  return -1;
}

/// Parse "1,0.5,0.25" into a ladder of positive scales, at most 1.
auto parseScales(const std::string& text) -> std::vector<double>
{
  std::vector<double> out;
  std::stringstream ss(text);
  for (std::string item; std::getline(ss, item, ',');) {
    const auto s = std::stod(item);
    if (not(s > 0.0 and s <= 1.0)) {
      throw std::invalid_argument("each scale must be in (0, 1]; got " + item);
    }
    out.push_back(s);
  }
  if (out.empty()) {
    throw std::invalid_argument("no scales given");
  }
  return out;
}

// --- reading the overhead calibration ----------------------------------------

/**
 * @brief The one localized view of a `pgs-localize` calibration, as a camera.
 *
 * @p transform, when given, moves the scene into the mesh's frame **before**
 * the pose is read, exactly as `pgs-localize --sfm-transform` moves a sparse
 * scene (`localize_sparse.cpp`): a 4x4 `[sR | t]` is the similarity
 * `X' = s R (X - C)` with `C = -R^T t / s`.
 */
auto ReadCalibrationCamera(const fs::path& path,
                           const std::optional<Eigen::Matrix4d>& transform)
    -> loc::Camera
{
  sfm::SfM_Data scene;
  if (not sfm::Load(scene, path.string(),
                    sfm::ESfM_Data(sfm::VIEWS | sfm::INTRINSICS |
                                   sfm::EXTRINSICS))) {
    throw std::runtime_error("could not read calibration " + path.string());
  }
  if (transform) {
    const Eigen::Matrix3d m = transform->block<3, 3>(0, 0);
    const Eigen::Vector3d t = transform->block<3, 1>(0, 3);
    const auto scale = m.col(0).norm();
    const Eigen::Matrix3d r = m / scale;
    if (scale <= 0.0 or
        (r.transpose() * r - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
            1e-6) {
      throw std::runtime_error("--sfm-transform is not a similarity");
    }
    const Eigen::Vector3d c = -r.transpose() * t / scale;
    sfm::ApplySimilarity(geo::Similarity3(geo::Pose3(r, c), scale), scene,
                         true);
  }

  for (const auto& [id, view] : scene.GetViews()) {
    if (not scene.IsPoseAndIntrinsicDefined(view.get())) {
      continue;
    }
    const auto pose = scene.GetPoseOrDie(view.get());
    const auto& intrinsic = *scene.GetIntrinsics().at(view->id_intrinsic);
    loc::Camera out;
    out.width = static_cast<int>(intrinsic.w());
    out.height = static_cast<int>(intrinsic.h());
    // CameraFromSolve reads K and radial k1..k3 out of any openMVG intrinsic.
    return loc::CameraFromSolve(out, intrinsic, pose);
  }
  throw std::runtime_error(path.string() + " holds no view with a pose");
}

// --- a minimal JSON writer -----------------------------------------------------

auto quote(const std::string& text) -> std::string
{
  std::string out = "\"";
  for (const auto ch : text) {
    if (ch == '"' or ch == '\\') {
      out += '\\';
    }
    out += ch;
  }
  return out + "\"";
}

auto number(const double value) -> std::string
{
  if (not std::isfinite(value)) {
    return "null";
  }
  std::ostringstream ss;
  ss << std::setprecision(10) << value;
  return ss.str();
}

auto vec(const Eigen::VectorXd& v) -> std::string
{
  std::string out = "[";
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    out += (i ? ", " : "") + number(v[i]);
  }
  return out + "]";
}

// --- position-map ------------------------------------------------------------

auto positionMap(const std::vector<std::string>& argv) -> int
{
  // clang-format off
  po::options_description opts("position-map");
  opts.add_options()
    ("help,h", "print help message")
    ("calibration", po::value<std::string>(), "the overhead view's one-view calibration .json (pgs-localize's output)")
    ("mesh", po::value<std::string>(), "the mesh to render. Its texture is unused: only positions are kept")
    ("output", po::value<std::string>(), "H x W x 3 float32 .npy of world positions, NaN where no surface was hit")
    ("sfm-transform", po::value<std::string>(), "4x4 .npy change of frame applied to the calibration's scene at load, as pgs-localize applies it to a sparse scene")
    ("output-image", po::value<std::string>(), "also write the rendered texture, 8-bit grey, black where nothing textured was hit. With a calibration made for a virtual camera, this and --output are a reference image the mesh supplies on its own -- a virtual overhead")
    ("threads", po::value<int>()->default_value(0), "renderer threads; 0 uses all")
  ;
  // clang-format on
  po::variables_map args;
  if (const auto code = parseOrExit("position-map", opts, argv,
                                    {"calibration", "mesh", "output"}, args);
      code >= 0) {
    return code;
  }
  if (not requireFiles(args, {"calibration", "mesh", "sfm-transform"})) {
    return IO_ERROR;
  }

  try {
    std::optional<Eigen::Matrix4d> transform;
    if (args.count("sfm-transform")) {
      transform = pgs::npy::read(args["sfm-transform"].as<std::string>());
    }
    const auto camera = ReadCalibrationCamera(
        args["calibration"].as<std::string>(), transform);
    std::cout << "Overhead camera:\n";
    loc::DescribeCamera(std::cout, camera);
    if (camera.distorted()) {
      // The renderer casts ideal pinhole rays (ADR 0011 Decision 3), so on a
      // distorted overhead the map sits slightly off the image's own pixels.
      std::cout << "WARNING: the overhead intrinsic carries distortion the "
                   "renderer does not model; its position map is the ideal "
                   "pinhole camera's\n";
    }

    const auto bundle = loc::ReadMeshBundle(args["mesh"].as<std::string>());
    const auto render = loc::RenderMesh(
        bundle, camera, static_cast<std::size_t>(args["threads"].as<int>()));
    std::cout << "  " << render.hits << " of "
              << static_cast<std::size_t>(camera.width) * camera.height
              << " pixels hit the surface\n";
    if (render.hits == 0) {
      std::cerr << "ERROR: the mesh is nowhere in the overhead view -- a frame "
                   "error, or the wrong --sfm-transform\n";
      return SOLVE_FAILED;
    }
    const auto out = fs::path(args["output"].as<std::string>());
    pgs::npy::writePositionMap(out, render.position);
    std::cout << "Wrote position map: " << out << "\n";
    if (args.count("output-image")) {
      const auto image = fs::path(args["output-image"].as<std::string>());
      if (not cv::imwrite(image.string(), render.gray)) {
        throw std::runtime_error("could not write " + image.string());
      }
      std::cout << "Wrote rendered image: " << image << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  return SUCCESS;
}

// --- view --------------------------------------------------------------------

/// One rung of the prior's scale ladder, as reported.
struct ScaleTrial {
  double scale{1.0};
  std::size_t queryFeatures{0};
  loc::MatchReport matches;
  std::size_t solveInliers{0};
};

auto view(const std::vector<std::string>& argv) -> int
{
  // clang-format off
  po::options_description opts("view");
  opts.add_options()
    ("help,h", "print help message")
    ("image", po::value<std::string>(), "the hand-held image to localize")
    ("mesh", po::value<std::string>(), "the IR reference mesh (Decision 2): the rig IR940 capture retexture")
    ("overhead-image", po::value<std::string>(), "the overhead view's 940 nm image, matched against for the prior")
    ("overhead-calibration", po::value<std::string>(), "the overhead view's one-view calibration. Checked for size against the image and the position map")
    ("overhead-position-map", po::value<std::string>(), "the overhead position map from `position-map`")
    ("camera", po::value<std::string>(), "starting K and distortion as a flat camera file (any pose in it is ignored). Required: a K resected from one near-planar view is degenerate (ADR 0011), and `adjust` refines this one jointly")
    ("output-calibration", po::value<std::string>(), "write the view's mesh-only one-view calibration here")
    ("output-correspondences", po::value<std::string>(), "write the inlier 2D<->3D correspondences here (.npy, N x 5 float64: x y X Y Z, x y in the query's own distorted pixels), the input to `adjust`")
    ("report", po::value<std::string>(), "write the JSON report here: the prior's scale ladder, match counts, held-out scores, footprint")
    ("mask", po::value<std::string>(), "query pixels never to match, where zero (the sample square). Same size as --image")
    ("output-camera", po::value<std::string>(), "also write the solved camera as a flat camera file")
    ("qa-render", po::value<std::string>(), "write the render at the solved pose, plus a difference image beside it. A frame error moves no residual; looking is what catches it")
    ("prior-scales", po::value<std::string>()->default_value("1,0.5,0.25,0.125"), "query downscales tried against the overhead image, comma-separated in (0, 1]. The scale whose resection keeps the most inliers is the prior. Phase 1 measures what this ladder should be")
    ("heldout-split", po::value<int>()->default_value(400), "side, in query pixels, of the held-out checkerboard (ADR 0011 Decision 7)")
    ("seed", po::value<unsigned int>(), "seed the RANSACs; unset is nondeterministic")
    ("threads", po::value<int>()->default_value(0), "renderer threads; 0 uses all")
  ;
  // clang-format on
  po::variables_map args;
  if (const auto code = parseOrExit(
          "view", opts, argv,
          {"image", "mesh", "overhead-image", "overhead-calibration",
           "overhead-position-map", "camera", "output-calibration",
           "output-correspondences", "report"},
          args);
      code >= 0) {
    return code;
  }
  if (not requireFiles(args, {"image", "mesh", "overhead-image",
                              "overhead-calibration", "overhead-position-map",
                              "camera", "mask"})) {
    return IO_ERROR;
  }
  // No --sfm-transform here, deliberately: the position map was rendered in the
  // mesh's frame (`position-map` applied it) and the IR reference is that mesh,
  // so every 3D point this view sees is already in the frame it must solve in.
  const auto path =[&args](const char* name) {
    return fs::path(args[name].as<std::string>());
  };
  const auto threads = static_cast<std::size_t>(args["threads"].as<int>());

  std::vector<double> scales;
  try {
    scales = parseScales(args["prior-scales"].as<std::string>());
  } catch (const std::exception& e) {
    std::cerr << "ERROR: --prior-scales: " << e.what() << "\n";
    return BAD_ARG;
  }

  // Matching options. The refinement is ADR 0011's render-and-match, so it
  // takes that tool's measured defaults unchanged. The prior matches against
  // a whole-plate image, so cascade hashing replaces brute force there: the
  // overhead carries an order of magnitude more features than any render.
  loc::MatchOptions refineOpts;
  if (args.count("seed")) {
    refineOpts.seed = args["seed"].as<unsigned int>();
  }
  auto priorOpts = refineOpts;
  priorOpts.matcher = openMVG::matching::CASCADE_HASHING_L2;
  // No Lucas-Kanade on the prior: OpenCV's pyramidal LK needs both images the
  // same size, which a render at the query's resolution always is and the
  // overhead never is. Nor would it buy anything -- the prior only has to land
  // inside render-and-match's basin (861 px, ADR 0011), and the refinement,
  // whose sizes do match, keeps LK for the precision that is emitted.
  priorOpts.refineMatches = false;
  const auto solver = loc::ParseSolver("auto", true);

  // --- inputs ------------------------------------------------------------------
  const auto image = path("image");
  const auto queryRaw = cv::imread(image.string(), cv::IMREAD_UNCHANGED);
  if (queryRaw.empty()) {
    std::cerr << "ERROR: could not read " << image << "\n";
    return IO_ERROR;
  }
  const int queryW = queryRaw.cols;
  const int queryH = queryRaw.rows;
  cv::Mat mask;
  if (args.count("mask")) {
    mask = cv::imread(path("mask").string(), cv::IMREAD_GRAYSCALE);
    if (mask.cols != queryW or mask.rows != queryH) {
      std::cerr << "ERROR: --mask is not the query's size\n";
      return BAD_ARG;
    }
  }

  loc::Camera camera;
  std::shared_ptr<cam::IntrinsicBase> intrinsic;
  try {
    camera = loc::RescaleCamera(loc::ReadCameraFile(path("camera")), queryW,
                                queryH);
    intrinsic = loc::MakeIntrinsic(camera);
    if (const auto complaint = loc::ComplainAboutIntrinsic(*intrinsic)) {
      std::cerr << "ERROR: --camera does not describe a physical camera: "
                << *complaint << "\n";
      return BAD_ARG;
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return BAD_ARG;
  }
  std::cout << "Starting K:\n";
  loc::DescribeCamera(std::cout, camera);

  // The overhead, wrapped as a Render: an image and its position map is
  // exactly what CorrespondFromReference lifts through, so the prior reuses the
  // matcher ADR 0011 measured rather than a second one.
  loc::Render overhead;
  try {
    const auto overheadRaw =
        cv::imread(path("overhead-image").string(), cv::IMREAD_UNCHANGED);
    if (overheadRaw.empty()) {
      throw std::runtime_error("could not read --overhead-image");
    }
    overhead.gray = loc::ToGray8(overheadRaw);
    overhead.position = pgs::npy::readPositionMap(path("overhead-position-map"));
    if (overhead.position.size() != overhead.gray.size()) {
      throw std::runtime_error(
          "the overhead position map is " +
          std::to_string(overhead.position.cols) + "x" +
          std::to_string(overhead.position.rows) + " but the image is " +
          std::to_string(overhead.gray.cols) + "x" +
          std::to_string(overhead.gray.rows) +
          "; it was rendered for a different image");
    }
    const auto overheadCam =
        ReadCalibrationCamera(path("overhead-calibration"), std::nullopt);
    if (overheadCam.width != overhead.gray.cols or
        overheadCam.height != overhead.gray.rows) {
      throw std::runtime_error("--overhead-calibration is not this image's");
    }
    // Only pixels that lift can become correspondences; describing the rest
    // costs time and adds nothing.
    cv::Mat finite;
    cv::compare(overhead.position, overhead.position, finite, cv::CMP_EQ);
    cv::extractChannel(finite, overhead.textured, 0);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }

  std::ostringstream report;
  report << "{\n  \"tool\": \"pgs-localize-joint view\",\n"
         << "  \"image\": " << quote(image.string()) << ",\n"
         << "  \"mesh\": " << quote(path("mesh").string()) << ",\n"
         << "  \"overhead_image\": " << quote(path("overhead-image").string())
         << ",\n"
         << "  \"camera\": " << quote(path("camera").string()) << ",\n";

  // --- prior: the query against the overhead image, over a scale ladder --------
  std::cout << "Describing the overhead image ("
            << cv::countNonZero(overhead.textured) << " pixels on the mesh)"
            << std::endl;
  const auto overheadFeatures =
      loc::DescribeReference(overhead, priorOpts, overhead.textured);
  std::cout << "  " << overheadFeatures.regions->RegionCount()
            << " overhead features\n";

  const auto query01 = loc::ToGray01(queryRaw);
  std::vector<ScaleTrial> trials;
  std::optional<loc::SolveResult> prior;
  double priorScale = 0.0;
  for (const auto s : scales) {
    ScaleTrial trial;
    trial.scale = s;
    cv::Mat q01 = query01;
    cv::Mat qMask = mask;
    if (s < 1.0) {
      cv::resize(query01, q01, {}, s, s, cv::INTER_AREA);
      if (not mask.empty()) {
        cv::resize(mask, qMask, q01.size(), 0, 0, cv::INTER_NEAREST);
      }
    }
    const auto qNorm = loc::LocalNormalize(q01, priorOpts.normalizeSigma,
                                           priorOpts.normalizeClip);
    const auto qRegions = loc::DescribeImage(qNorm, qMask, priorOpts.preset);
    trial.queryFeatures = qRegions->RegionCount();
    auto corrs = loc::CorrespondFromReference(
        overhead, overheadFeatures, qNorm, *qRegions, priorOpts,
        trial.matches);
    // Back into the full-resolution query's pixels, centres to centres.
    for (auto& c : corrs) {
      c.uv = ((c.uv.array() + 0.5) / s - 0.5).matrix();
    }
    if (corrs.size() >= kMinCorrespondences) {
      auto solved = loc::SolvePose(corrs, intrinsic, kModel, solver, queryW,
                                   queryH, 0.0);
      if (solved.ok) {
        trial.solveInliers = solved.inliers.size();
        if (not prior or solved.inliers.size() > prior->inliers.size()) {
          prior = std::move(solved);
          priorScale = s;
        }
      }
    }
    std::cout << "  scale " << s << ": " << trial.queryFeatures
              << " query features, " << trial.matches.lowePassed
              << " Lowe, " << trial.matches.similarityInliers
              << " similarity, " << trial.matches.lifted << " lifted, "
              << trial.solveInliers << " resection inliers\n";
    trials.push_back(trial);
  }

  report << "  \"prior\": {\n    \"scale\": " << number(priorScale)
         << ",\n    \"inliers\": "
         << (prior ? prior->inliers.size() : std::size_t{0})
         << ",\n    \"ladder\": [";
  for (std::size_t i = 0; i < trials.size(); ++i) {
    const auto& t = trials[i];
    report << (i ? ", " : "") << "{\"scale\": " << number(t.scale)
           << ", \"query_features\": " << t.queryFeatures
           << ", \"lowe_passed\": " << t.matches.lowePassed
           << ", \"similarity_inliers\": " << t.matches.similarityInliers
           << ", \"similarity_scale\": " << number(t.matches.similarityScale)
           << ", \"lifted\": " << t.matches.lifted
           << ", \"resection_inliers\": " << t.solveInliers << "}";
  }
  report << "]\n  },\n";

  const auto writeReport = [&](const std::string& tail) {
    std::ofstream f(path("report"));
    f << report.str() << tail << "}\n";
  };

  if (not prior) {
    std::cerr << "ERROR: no scale of the query resected against the overhead "
                 "image. Try a wider --prior-scales ladder; if none works, the "
                 "view needs a manual seed (ADR 0013 Decision 2)\n";
    writeReport("  \"status\": \"no-prior\"\n");
    return NO_CORRESPONDENCES;
  }

  // --- refinement: render-and-match against the IR reference --------------
  const auto priorCam = loc::CameraFromSolve(camera, *intrinsic, prior->pose);
  std::cout << "Prior pose (scale " << priorScale << "):\n";
  loc::DescribeCamera(std::cout, priorCam);

  loc::MeshBundle bundle;
  loc::Render render;
  cv::Mat queryNorm;
  std::vector<loc::Correspondence> corrs;
  loc::MatchReport refineMatches;
  try {
    bundle = loc::ReadMeshBundle(path("mesh"));
    if (not bundle.has_texture()) {
      throw std::runtime_error("the mesh carries no usable texture");
    }
    render = loc::RenderMesh(bundle, priorCam, threads);
    queryNorm = loc::LocalNormalize(query01, refineOpts.normalizeSigma,
                                    refineOpts.normalizeClip);
    const auto qRegions =
        loc::DescribeImage(queryNorm, mask, refineOpts.preset);
    // The texture mask: no keypoint from the edge of an untextured region,
    // which exists in no photograph (ADR 0013 Decision 2).
    const auto renderFeatures =
        loc::DescribeReference(render, refineOpts, render.textured);
    corrs = loc::CorrespondFromReference(render, renderFeatures, queryNorm,
                                         *qRegions, refineOpts, refineMatches);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  std::cout << "Render-and-match: " << render.hits << " hits, "
            << cv::countNonZero(render.textured) << " textured; "
            << refineMatches.lowePassed << " Lowe, "
            << refineMatches.similarityInliers << " similarity, "
            << refineMatches.lifted << " lifted\n";

  report << "  \"refine\": {\"render_hits\": " << render.hits
         << ", \"textured_px\": " << cv::countNonZero(render.textured)
         << ", \"detected_render\": " << refineMatches.detectedRender
         << ", \"detected_query\": " << refineMatches.detectedQuery
         << ", \"lowe_passed\": " << refineMatches.lowePassed
         << ", \"similarity_inliers\": " << refineMatches.similarityInliers
         << ", \"lk_kept\": " << refineMatches.lkKept
         << ", \"lifted\": " << refineMatches.lifted << "},\n";

  if (corrs.size() < kMinCorrespondences) {
    std::cerr << "ERROR: only " << corrs.size()
              << " correspondences against the IR reference\n";
    writeReport("  \"status\": \"no-correspondences\"\n");
    return NO_CORRESPONDENCES;
  }
  const auto solved = loc::SolvePose(corrs, intrinsic, kModel, solver, queryW,
                                     queryH, 0.0);
  if (not solved.ok) {
    std::cerr << "ERROR: resection failed on " << corrs.size()
              << " correspondences\n";
    writeReport("  \"status\": \"solve-failed\"\n");
    return SOLVE_FAILED;
  }
  const auto heldout = loc::HeldoutScore(corrs, intrinsic, kModel, solver,
                                         queryW, queryH, 0.0,
                                         args["heldout-split"].as<int>());
  const auto outCam = loc::CameraFromSolve(camera, *intrinsic, solved.pose);
  std::cout << "Solved: " << solved.inliers.size() << " inliers, inlier RMS "
            << solved.inlierRmsPx << " px";
  if (heldout.ok) {
    std::cout << "; held-out rms " << heldout.stats.rmsPx << " px";
  }
  std::cout << "\n";
  loc::DescribeCamera(std::cout, outCam);

  // Footprint: the extent, in scene units, of the surface the solved view saw.
  // Phase 1 reads this to size the resolution gap (ADR 0013).
  Eigen::Vector3d lo = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  Eigen::Vector3d hi = -lo;
  for (const auto& c : corrs) {
    lo = lo.cwiseMin(c.X);
    hi = hi.cwiseMax(c.X);
  }

  // Row-major, as the camera file's `pose` is written.
  const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Rrow =
      solved.pose.rotation();
  report << "  \"solve\": {\"n_inliers\": " << solved.inliers.size()
         << ", \"inlier_rms_px\": " << number(solved.inlierRmsPx)
         << ", \"spread_px\": [" << number(solved.spreadX) << ", "
         << number(solved.spreadY) << "]},\n"
         << "  \"heldout\": {\"ok\": " << (heldout.ok ? "true" : "false")
         << ", \"fit_count\": " << heldout.fitCount
         << ", \"test_count\": " << heldout.testCount
         << ", \"rms_px\": " << number(heldout.stats.rmsPx)
         << ", \"p99_px\": " << number(heldout.stats.p99Px)
         << ", \"bias_px\": " << number(heldout.stats.biasPx) << "},\n"
         << "  \"pose\": {\"center\": " << vec(solved.pose.center())
         << ", \"rotation\": "
         << vec(Eigen::Map<const Eigen::VectorXd>(Rrow.data(), 9)) << "},\n"
         << "  \"footprint\": {\"lo\": " << vec(lo) << ", \"hi\": " << vec(hi)
         << "},\n";

  // --- emit --------------------------------------------------------------------
  try {
    loc::WriteCalibration(path("output-calibration"), *intrinsic, solved.pose,
                          image);
    std::vector<std::vector<double>> rows;
    rows.reserve(solved.inliers.size());
    for (const auto i : solved.inliers) {
      const auto& c = corrs[i];
      rows.push_back({c.uv.x(), c.uv.y(), c.X.x(), c.X.y(), c.X.z()});
    }
    pgs::npy::writeTable(path("output-correspondences"), rows, 5);
    if (args.count("output-camera")) {
      loc::WriteCameraFile(path("output-camera"), outCam);
    }
    if (args.count("qa-render")) {
      const auto out = path("qa-render");
      const auto qa = loc::RenderMesh(bundle, outCam, threads);
      const auto qaNorm = loc::LocalNormalize(
          loc::ToGray01(qa.gray), refineOpts.normalizeSigma,
          refineOpts.normalizeClip);
      cv::Mat diff;
      cv::absdiff(qaNorm, queryNorm, diff);
      auto diffPath = out;
      diffPath.replace_filename(out.stem().string() + "_diff" +
                                out.extension().string());
      if (not cv::imwrite(out.string(), qaNorm) or
          not cv::imwrite(diffPath.string(), diff)) {
        throw std::runtime_error("could not write " + out.string());
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  writeReport("  \"status\": \"done\"\n");
  std::cout << "Wrote " << path("output-calibration") << "\n";
  return SUCCESS;
}

// --- adjust ------------------------------------------------------------------

/// Huber scale, in pixels, for every residual of the joint adjustment: the
/// mesh anchors were already RANSAC inliers, so this only has to soften the
/// odd survivor and the tie points that triangulated badly.
constexpr double kLossPx = 2.0;

/// How far a tie point may reproject from its observations, under the
/// mesh-only poses and the starting K, before it is not trusted as a tie.
constexpr double kTieMaxPx = 4.0;

/// The focal cost curve is "flat" when either end of the +-20% sweep costs less
/// than this much more, relatively, than its minimum. Provisional, set between
/// the two measurements there are: the synthetic tilted views' ends at 30-38x
/// their minimum, and PHerc0128 Cr04's near-nadir hand-held views at 1.03x and
/// 1.05x -- where a 1% line passed a focal that moved the principal point
/// 161 px and made an overlapping pair worse (ADR 0013 Decision 4).
constexpr double kFlatRise = 0.25;

/// KParams indices, for holding parts of K in a solve.
constexpr int kF = 0, kCx = 1, kCy = 2, kK2 = 4, kK3 = 5;

/// `pinhole_radial_k3`, in the order openMVG's getParams() gives it.
using KParams = std::array<double, 6>;  // f, cx, cy, k1, k2, k3
/// A pose as Ceres adjusts it: angle-axis rotation, then translation, of
/// x_cam = R X + t.
using PoseParams = std::array<double, 6>;

/// openMVG's pinhole_radial_k3 projection, exactly: distortion multiplies the
/// normalized coordinates, then K. What `pgs-retexture` and
/// `utils.visibility.project_points` apply, so the emitted K means the same
/// thing to every consumer.
template <typename T>
void ProjectK3(const T* k, const T* pose, const T* X, T* uv)
{
  T p[3];
  ceres::AngleAxisRotatePoint(pose, X, p);
  p[0] += pose[3];
  p[1] += pose[4];
  p[2] += pose[5];
  const T x = p[0] / p[2];
  const T y = p[1] / p[2];
  const T r2 = x * x + y * y;
  const T d = T(1.0) + k[3] * r2 + k[4] * r2 * r2 + k[5] * r2 * r2 * r2;
  uv[0] = k[0] * x * d + k[1];
  uv[1] = k[0] * y * d + k[2];
}

/// A mesh anchor: its world point is the mesh's and does not move.
struct AnchorCost {
  double u, v, X[3];
  template <typename T>
  auto operator()(const T* k, const T* pose, T* r) const -> bool
  {
    const T Xt[3] = {T(X[0]), T(X[1]), T(X[2])};
    T uv[2];
    ProjectK3(k, pose, Xt, uv);
    r[0] = uv[0] - T(u);
    r[1] = uv[1] - T(v);
    return true;
  }
};

/// A tie point: seen by two or more views, its world point is adjusted.
struct TieCost {
  double u, v;
  template <typename T>
  auto operator()(const T* k, const T* pose, const T* X, T* r) const -> bool
  {
    T uv[2];
    ProjectK3(k, pose, X, uv);
    r[0] = uv[0] - T(u);
    r[1] = uv[1] - T(v);
    return true;
  }
};

auto PoseToParams(const geo::Pose3& pose) -> PoseParams
{
  PoseParams p{};
  const Eigen::Matrix3d R = pose.rotation();
  ceres::RotationMatrixToAngleAxis(R.data(), p.data());  // column-major
  const Eigen::Vector3d t = pose.translation();
  p[3] = t[0];
  p[4] = t[1];
  p[5] = t[2];
  return p;
}

auto ParamsToPose(const PoseParams& p) -> geo::Pose3
{
  Eigen::Matrix3d R;
  ceres::AngleAxisToRotationMatrix(p.data(), R.data());
  const Eigen::Vector3d t(p[3], p[4], p[5]);
  return geo::Pose3(R, -R.transpose() * t);
}

auto MakeK3(const KParams& k, const int width, const int height)
    -> std::shared_ptr<cam::Pinhole_Intrinsic_Radial_K3>
{
  return std::make_shared<cam::Pinhole_Intrinsic_Radial_K3>(
      width, height, k[0], k[1], k[2], k[3], k[4], k[5]);
}

/// One localized view, as `view` left it.
struct ViewInput {
  fs::path calibration;
  fs::path output;
  fs::path image;
  /// The key a tie scene's views are matched to this one by.
  std::string filename;
  int width{0};
  int height{0};
  geo::Pose3 meshPose;
  KParams meshK{};
  std::vector<AnchorCost> anchors;
};

/// A tie point and the views that see it.
struct Track {
  std::array<double, 3> X{};
  std::vector<std::pair<std::size_t, std::array<double, 2>>> obs;
};

struct Solution {
  KParams k{};
  std::vector<PoseParams> poses;
  std::vector<std::array<double, 3>> ties;
  ceres::Solver::Summary summary;
};

auto ReadViewInput(const fs::path& calibration, const fs::path& correspondences)
    -> ViewInput
{
  sfm::SfM_Data scene;
  if (not sfm::Load(scene, calibration.string(),
                    sfm::ESfM_Data(sfm::VIEWS | sfm::INTRINSICS |
                                   sfm::EXTRINSICS))) {
    throw std::runtime_error("could not read " + calibration.string());
  }
  for (const auto& [id, view] : scene.GetViews()) {
    if (not scene.IsPoseAndIntrinsicDefined(view.get())) {
      continue;
    }
    const auto& intrinsic = *scene.GetIntrinsics().at(view->id_intrinsic);
    const auto type = intrinsic.getType();
    if (type != cam::PINHOLE_CAMERA and type != cam::PINHOLE_CAMERA_RADIAL1 and
        type != cam::PINHOLE_CAMERA_RADIAL3) {
      throw std::runtime_error(calibration.string() +
                               ": only pinhole and radial intrinsics adjust "
                               "into pinhole_radial_k3");
    }
    ViewInput in;
    in.calibration = calibration;
    in.image = fs::path(scene.s_root_path) / view->s_Img_path;
    in.filename = fs::path(view->s_Img_path).filename().string();
    in.width = static_cast<int>(intrinsic.w());
    in.height = static_cast<int>(intrinsic.h());
    in.meshPose = scene.GetPoseOrDie(view.get());
    const auto p = intrinsic.getParams();
    in.meshK = {p[0], p[1], p[2], 0.0, 0.0, 0.0};
    for (std::size_t i = 3; i < p.size() and i < 6; ++i) {
      in.meshK[i] = p[i];
    }
    for (const auto& row : pgs::npy::readTable(correspondences, 5)) {
      in.anchors.push_back({row[0], row[1], {row[2], row[3], row[4]}});
    }
    return in;
  }
  throw std::runtime_error(calibration.string() + " holds no view with a pose");
}

/// RMS reprojection of a view's anchors under one K and one pose, in pixels.
auto AnchorRms(const ViewInput& view, const KParams& k, const PoseParams& pose)
    -> double
{
  if (view.anchors.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  for (const auto& a : view.anchors) {
    double r[2];
    a(k.data(), pose.data(), r);
    sum += r[0] * r[0] + r[1] * r[1];
  }
  return std::sqrt(sum / static_cast<double>(view.anchors.size()));
}

/**
 * @brief Read the groups' tie scenes and triangulate their tracks into the
 *        mesh's frame.
 *
 * A group's SfM solve is in its own arbitrary frame, so its structure is not
 * used as it stands: each track is re-triangulated from the views' mesh-only
 * poses and the starting K, which puts it in the frame the anchors are in. A
 * track survives only if it is seen by two or more of the views being adjusted,
 * has parallax, lies in front of every camera, and reprojects within
 * kTieMaxPx everywhere.
 */
auto ReadTies(const std::vector<std::string>& scenes,
              const std::vector<ViewInput>& views, const KParams& startK,
              std::ostream& report) -> std::vector<Track>
{
  std::map<std::string, std::size_t> byName;
  for (std::size_t i = 0; i < views.size(); ++i) {
    byName[views[i].filename] = i;
  }
  const auto intrinsic = MakeK3(startK, views.front().width,
                                views.front().height);
  std::vector<PoseParams> poses;
  for (const auto& v : views) {
    poses.push_back(PoseToParams(v.meshPose));
  }

  std::vector<Track> tracks;
  std::size_t seen = 0;
  std::size_t tooFew = 0;
  std::size_t rejected = 0;
  for (const auto& path : scenes) {
    sfm::SfM_Data scene;
    if (not sfm::Load(scene, path,
                      sfm::ESfM_Data(sfm::VIEWS | sfm::STRUCTURE))) {
      throw std::runtime_error("could not read tie scene " + path);
    }
    std::map<openMVG::IndexT, std::size_t> local;
    for (const auto& [id, view] : scene.GetViews()) {
      const auto name = fs::path(view->s_Img_path).filename().string();
      if (const auto it = byName.find(name); it != byName.end()) {
        local[id] = it->second;
      }
    }
    for (const auto& [id, landmark] : scene.GetLandmarks()) {
      ++seen;
      Track track;
      for (const auto& [viewId, ob] : landmark.obs) {
        if (const auto it = local.find(viewId); it != local.end()) {
          track.obs.push_back({it->second, {ob.x[0], ob.x[1]}});
        }
      }
      if (track.obs.size() < 2) {
        ++tooFew;
        continue;
      }
      // Least-squares intersection of the rays: sum (I - d d^T)(X - C) = 0.
      Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
      Eigen::Vector3d b = Eigen::Vector3d::Zero();
      for (const auto& [vi, x] : track.obs) {
        const auto& pose = views[vi].meshPose;
        const auto n = intrinsic->ima2cam(
            intrinsic->get_ud_pixel(openMVG::Vec2(x[0], x[1])));
        const Eigen::Vector3d d =
            (pose.rotation().transpose() * Eigen::Vector3d(n[0], n[1], 1.0))
                .normalized();
        const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d * d.transpose();
        A += P;
        b += P * pose.center();
      }
      if (A.eigenvalues().real().minCoeff() < 1e-6) {
        ++rejected;  // no parallax
        continue;
      }
      const Eigen::Vector3d X = A.ldlt().solve(b);
      track.X = {X[0], X[1], X[2]};
      bool ok = true;
      for (const auto& [vi, x] : track.obs) {
        const auto& pose = views[vi].meshPose;
        if ((pose.rotation() * (X - pose.center()))[2] <= 0.0) {
          ok = false;
          break;
        }
        double r[2];
        TieCost{x[0], x[1]}(startK.data(), poses[vi].data(), track.X.data(), r);
        if (std::hypot(r[0], r[1]) > kTieMaxPx) {
          ok = false;
          break;
        }
      }
      if (not ok) {
        ++rejected;
        continue;
      }
      tracks.push_back(std::move(track));
    }
  }
  std::size_t observations = 0;
  for (const auto& t : tracks) {
    observations += t.obs.size();
  }
  std::cout << "Tie points: " << tracks.size() << " of " << seen
            << " tracks (" << tooFew << " seen by fewer than two of these "
            << "views, " << rejected << " rejected), " << observations
            << " observations\n";
  report << "  \"ties\": {\"scenes\": " << scenes.size()
         << ", \"tracks_read\": " << seen << ", \"too_few_views\": " << tooFew
         << ", \"rejected\": " << rejected << ", \"used\": " << tracks.size()
         << ", \"observations\": " << observations << "},\n";
  return tracks;
}

/// The KParams indices in @p a or @p b, sorted and once each.
auto Held(std::vector<int> a, const std::vector<int>& b) -> std::vector<int>
{
  a.insert(a.end(), b.begin(), b.end());
  std::sort(a.begin(), a.end());
  a.erase(std::unique(a.begin(), a.end()), a.end());
  return a;
}

/// The names of the KParams a solve holding @p held left free, as JSON.
auto FreeJson(const std::vector<int>& held) -> std::string
{
  static const std::array<const char*, 6> names{"f",  "cx", "cy",
                                                "k1", "k2", "k3"};
  std::string out = "[";
  for (int i = 0; i < 6; ++i) {
    if (std::find(held.begin(), held.end(), i) == held.end()) {
      out += std::string(out.size() > 1 ? ", " : "") + quote(names[i]);
    }
  }
  return out + "]";
}

/**
 * @brief The joint adjustment: one K, every pose, anchors fixed, ties free.
 *
 * @param held the parts of K (KParams indices) held at @p sol's values; all
 *        six is a fixed K
 */
auto Solve(const std::vector<ViewInput>& views,
           const std::vector<Track>& tracks, Solution sol,
           const std::vector<int>& held, const int threads) -> Solution
{
  ceres::Problem problem;
  for (std::size_t i = 0; i < views.size(); ++i) {
    for (const auto& a : views[i].anchors) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<AnchorCost, 2, 6, 6>(
              new AnchorCost(a)),
          new ceres::HuberLoss(kLossPx), sol.k.data(), sol.poses[i].data());
    }
  }
  for (std::size_t t = 0; t < tracks.size(); ++t) {
    for (const auto& [vi, x] : tracks[t].obs) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<TieCost, 2, 6, 6, 3>(
              new TieCost{x[0], x[1]}),
          new ceres::HuberLoss(kLossPx), sol.k.data(), sol.poses[vi].data(),
          sol.ties[t].data());
    }
  }
  if (held.size() >= 6) {
    problem.SetParameterBlockConstant(sol.k.data());
  } else if (not held.empty()) {
    problem.SetManifold(sol.k.data(), new ceres::SubsetManifold(6, held));
  }

  ceres::Solver::Options options;
  // Anchors only: a few dozen parameters over many residuals, which dense QR
  // handles outright. With ties, the points are many and independent, which is
  // the problem a Schur complement exists for.
  options.linear_solver_type =
      tracks.empty() ? ceres::DENSE_QR : ceres::SPARSE_SCHUR;
  options.max_num_iterations = 200;
  options.function_tolerance = 1e-12;
  options.parameter_tolerance = 1e-12;
  options.num_threads = threads > 0 ? threads : 1;
  ceres::Solve(options, &problem, &sol.summary);
  return sol;
}

auto adjust(const std::vector<std::string>& argv) -> int
{
  // clang-format off
  po::options_description opts("adjust");
  opts.add_options()
    ("help,h", "print help message")
    ("calibration", po::value<std::vector<std::string>>()->composing(), "a view's mesh-only calibration, from `view`. Repeated, once per view")
    ("correspondences", po::value<std::vector<std::string>>()->composing(), "that view's correspondences, from `view`. Repeated, paired with --calibration by position")
    ("output-calibration", po::value<std::vector<std::string>>()->composing(), "where that view's final calibration goes. Repeated, paired by position")
    ("tie-scene", po::value<std::vector<std::string>>()->composing(), "a group's own SfM solve; its tracks are re-triangulated into the mesh's frame and become free tie points. Views are matched to it by image filename")
    ("output-scene", po::value<std::string>(), "every view at its adjusted pose, sharing one intrinsic, with the tie points as structure")
    ("output-camera", po::value<std::string>(), "the shared K and distortion as a flat camera file")
    ("report", po::value<std::string>(), "JSON report: K before and after, per-view divergence, the focal cost curve")
    ("camera", po::value<std::string>(), "starting K and distortion. Default: the K the first calibration was solved with")
    ("fix-intrinsic", po::bool_switch(), "hold K at --camera (a board-calibrated camera)")
    ("focal-sweep", po::value<std::string>(), "lo:hi:steps, in pixels, for the focal cost curve. The curve always runs; unset sweeps 0.8x to 1.2x the adjusted focal in 9 steps")
    ("divergence-px", po::value<double>()->default_value(1.0, "1"), "a view whose anchors reproject this much worse under the joint solve than under its own mesh-only one is divergent: its mesh-only calibration is emitted instead (ADR 0013 Decision 5). Provisional; nothing measured sets it yet")
    ("threads", po::value<int>()->default_value(0), "solver threads; 0 uses 1")
  ;
  // clang-format on
  po::variables_map args;
  if (const auto code = parseOrExit(
          "adjust", opts, argv,
          {"calibration", "correspondences", "output-calibration",
           "output-scene", "output-camera", "report"},
          args);
      code >= 0) {
    return code;
  }
  const auto calibrations = args["calibration"].as<std::vector<std::string>>();
  const auto correspondences =
      args["correspondences"].as<std::vector<std::string>>();
  const auto outputs = args["output-calibration"].as<std::vector<std::string>>();
  if (correspondences.size() != calibrations.size() or
      outputs.size() != calibrations.size()) {
    std::cerr << "ERROR: --calibration, --correspondences and "
                 "--output-calibration are one each per view\n";
    return BAD_ARG;
  }
  const auto fixK = args["fix-intrinsic"].as<bool>();
  if (fixK and not args.count("camera")) {
    std::cerr << "ERROR: --fix-intrinsic holds K at --camera; give one\n";
    return BAD_ARG;
  }
  if (not requireFiles(args, {"calibration", "correspondences", "tie-scene",
                              "camera"})) {
    return IO_ERROR;
  }
  const auto threads = args["threads"].as<int>();
  const auto divergencePx = args["divergence-px"].as<double>();

  std::ostringstream report;
  report << "{\n  \"tool\": \"pgs-localize-joint adjust\",\n";

  // --- inputs ------------------------------------------------------------
  std::vector<ViewInput> views;
  try {
    for (std::size_t i = 0; i < calibrations.size(); ++i) {
      auto v = ReadViewInput(calibrations[i], correspondences[i]);
      v.output = outputs[i];
      if (not views.empty() and (v.width != views.front().width or
                                 v.height != views.front().height)) {
        throw std::runtime_error(
            calibrations[i] + " is " + std::to_string(v.width) + "x" +
            std::to_string(v.height) +
            "; one shared K needs every view at one resolution");
      }
      views.push_back(std::move(v));
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  const int width = views.front().width;
  const int height = views.front().height;

  KParams startK = views.front().meshK;
  if (args.count("camera")) {
    try {
      const auto c = loc::RescaleCamera(
          loc::ReadCameraFile(args["camera"].as<std::string>()), width, height);
      if (std::abs(c.fx - c.fy) > 1e-6 * c.fx) {
        throw std::runtime_error("--camera has fx != fy; pinhole_radial_k3 "
                                 "carries one focal");
      }
      startK = {c.fx, c.cx, c.cy, c.disto[0], c.disto[1], c.disto[2]};
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return BAD_ARG;
    }
  }
  std::size_t anchors = 0;
  for (const auto& v : views) {
    anchors += v.anchors.size();
  }
  std::cout << views.size() << " views, " << anchors << " mesh anchors; "
            << "starting f " << startK[0] << "\n";

  std::vector<Track> tracks;
  try {
    const auto scenes =
        args.count("tie-scene")
            ? args["tie-scene"].as<std::vector<std::string>>()
            : std::vector<std::string>{};
    tracks = ReadTies(scenes, views, startK, report);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }

  // --- the joint solve ---------------------------------------------------
  Solution start;
  start.k = startK;
  for (const auto& v : views) {
    start.poses.push_back(PoseToParams(v.meshPose));
  }
  for (const auto& t : tracks) {
    start.ties.push_back(t.X);
  }
  // k2 and k3 are freed only with tie points, which are dense at the frame
  // edges where they act. Anchors alone see distortion only through the
  // mesh's own ~2 px of error, and on a narrow lens the r^4 and r^6 terms are
  // nearly collinear with r^2 over the frame: PHerc0128 Cr04's corner is at
  // r = 0.24, where they came back as k2 = 6.4, k3 = -84 (ADR 0013 Decision 4).
  const std::vector<int> base =
      fixK ? std::vector<int>{0, 1, 2, 3, 4, 5}
           : (tracks.empty() ? std::vector<int>{kK2, kK3} : std::vector<int>{});
  const auto sol = Solve(views, tracks, start, base, threads);
  std::cout << sol.summary.BriefReport() << "\n";
  if (not sol.summary.IsSolutionUsable()) {
    std::cerr << "ERROR: the joint adjustment did not converge\n";
    return SOLVE_FAILED;
  }
  std::cout << "Adjusted f " << sol.k[0] << ", principal point (" << sol.k[1]
            << ", " << sol.k[2] << "), k = (" << sol.k[3] << ", " << sol.k[4]
            << ", " << sol.k[5] << ")\n";

  const auto kJson = [](const KParams& k) {
    return "{\"f\": " + number(k[0]) + ", \"cx\": " + number(k[1]) +
           ", \"cy\": " + number(k[2]) + ", \"k\": [" + number(k[3]) + ", " +
           number(k[4]) + ", " + number(k[5]) + "]}";
  };
  report << "  \"intrinsic\": {\"start\": " << kJson(startK)
         << ", \"adjusted\": " << kJson(sol.k)
         << ", \"free\": " << FreeJson(base)
         << ", \"fixed\": " << (fixK ? "true" : "false") << "},\n"
         << "  \"solve\": {\"initial_cost\": "
         << number(sol.summary.initial_cost)
         << ", \"final_cost\": " << number(sol.summary.final_cost)
         << ", \"iterations\": " << sol.summary.iterations.size()
         << ", \"termination\": "
         << quote(ceres::TerminationTypeToString(
                sol.summary.termination_type))
         << "},\n";

  // --- the focal cost curve (Decision 4) ---------------------------------
  // Measured every run, never assumed: a flat curve says this set of views
  // does not determine the focal, whatever the adjustment reported.
  double lo = 0.8 * sol.k[0];
  double hi = 1.2 * sol.k[0];
  int steps = 9;
  if (args.count("focal-sweep")) {
    const auto text = args["focal-sweep"].as<std::string>();
    std::stringstream ss(text);
    std::string a, b, c;
    if (not std::getline(ss, a, ':') or not std::getline(ss, b, ':') or
        not std::getline(ss, c)) {
      std::cerr << "ERROR: --focal-sweep is lo:hi:steps\n";
      return BAD_ARG;
    }
    try {
      lo = std::stod(a);
      hi = std::stod(b);
      steps = std::stoi(c);
    } catch (const std::exception&) {
      std::cerr << "ERROR: --focal-sweep is lo:hi:steps\n";
      return BAD_ARG;
    }
    if (not(lo > 0.0 and hi > lo and steps >= 3)) {
      std::cerr << "ERROR: --focal-sweep needs 0 < lo < hi and steps >= 3\n";
      return BAD_ARG;
    }
  }
  std::vector<std::pair<double, double>> curve;
  for (int s = 0; s < steps; ++s) {
    auto at = sol;
    at.k[kF] = lo + (hi - lo) * s / (steps - 1);
    const auto held = Solve(views, tracks, at, Held(base, {kF}), threads);
    curve.emplace_back(at.k[kF], held.summary.final_cost);
  }
  const auto minIt = std::min_element(
      curve.begin(), curve.end(),
      [](const auto& a, const auto& b) { return a.second < b.second; });
  const auto minCost = minIt->second;
  const bool interior = minIt != curve.begin() and minIt != curve.end() - 1;
  const bool rises = curve.front().second >= minCost * (1.0 + kFlatRise) and
                     curve.back().second >= minCost * (1.0 + kFlatRise);
  const bool observable = interior and rises;
  std::cout << "Focal cost curve: minimum at f " << minIt->first << "; "
            << (observable ? "the focal is determined by these views"
                           : "FLAT or at the edge of the sweep -- these views "
                             "do not determine the focal")
            << "\n";
  report << "  \"focal_curve\": {\"observable\": "
         << (observable ? "true" : "false")
         << ", \"minimum_f\": " << number(minIt->first) << ", \"points\": [";
  for (std::size_t i = 0; i < curve.size(); ++i) {
    report << (i ? ", " : "") << "[" << number(curve[i].first) << ", "
           << number(curve[i].second) << "]";
  }
  report << "]},\n";

  // A flat curve is not a weak measurement of the focal, it is none: the
  // adjustment then drifts along the focal/standoff degeneracy to whatever K
  // it likes, and its residuals improve while every pose walks away from the
  // truth (measured on untilted synthetic views: f 1600 -> 11652, every camera
  // 377 units off, anchors reprojecting *better*). The principal point rides
  // the same tilt, trading against rotation. So when the curve is flat, both
  // are held at the starting K and the set is re-solved for distortion alone,
  // which does not need tilt: a radial error is not something a pose can
  // absorb, on any view (Decision 4).
  enum class Emit { Adjusted, Distortion, Start };
  auto emit = (fixK or observable) ? Emit::Adjusted : Emit::Distortion;
  auto chosen = sol;
  if (emit == Emit::Distortion) {
    chosen = Solve(views, tracks, start, Held(base, {kF, kCx, kCy}), threads);
    std::cout << "These views do not determine the focal: f and the principal "
                 "point are held at the starting K, and only distortion is "
                 "adjusted. k = ("
              << chosen.k[3] << ", " << chosen.k[4] << ", " << chosen.k[5]
              << ")\n";
    if (not chosen.summary.IsSolutionUsable()) {
      emit = Emit::Start;
    }
  }
  if (emit != Emit::Start and not fixK) {
    if (const auto complaint =
            loc::ComplainAboutIntrinsic(*MakeK3(chosen.k, width, height))) {
      if (emit == Emit::Adjusted) {
        std::cerr << "ERROR: the joint adjustment produced a K that is not a "
                     "physical camera: "
                  << *complaint << "\n";
        return SOLVE_FAILED;
      }
      std::cout << "WARNING: the distortion-only solve is not a physical "
                   "camera (" << *complaint << ")\n";
      emit = Emit::Start;
    }
  }
  if (emit != Emit::Adjusted) {
    std::cout << "WARNING: these views do not determine the focal, so no "
                 "adjusted focal is emitted"
              << (emit == Emit::Start
                      ? ", and every view keeps its mesh-only calibration at "
                        "the starting K"
                      : "")
              << ". To do better, calibrate the camera on a board and pass it "
                 "with --camera --fix-intrinsic, or add views tilted about "
                 "different axes (ADR 0013 Decision 4)\n";
  }
  const bool trustK = emit != Emit::Start;
  const KParams& emitK = trustK ? chosen.k : startK;
  report << "  \"intrinsic_emitted\": "
         << (emit == Emit::Adjusted     ? "\"adjusted\""
             : emit == Emit::Distortion ? "\"distortion\""
                                        : "\"start\"");
  if (emit == Emit::Distortion) {
    report << ",\n  \"distortion_only\": {\"k\": " << kJson(chosen.k)
           << ", \"free\": " << FreeJson(Held(base, {kF, kCx, kCy}))
           << ", \"final_cost\": " << number(chosen.summary.final_cost) << "}";
  }
  report << ",\n";

  // --- per-view divergence (Decision 5), and emit -------------------------
  const auto adjusted = MakeK3(emitK, width, height);
  report << "  \"views\": [\n";
  std::size_t divergent = 0;
  try {
    for (std::size_t i = 0; i < views.size(); ++i) {
      const auto& v = views[i];
      const auto meshParams = PoseToParams(v.meshPose);
      const auto meshRms = AnchorRms(v, v.meshK, meshParams);
      const auto jointRms = AnchorRms(v, chosen.k, chosen.poses[i]);
      const auto jointPose = ParamsToPose(chosen.poses[i]);
      const auto shift = (jointPose.center() - v.meshPose.center()).norm();
      const Eigen::AngleAxisd turn(jointPose.rotation() *
                                   v.meshPose.rotation().transpose());
      const bool isDivergent = trustK and jointRms > meshRms + divergencePx;
      divergent += isDivergent;
      const bool meshOnly = isDivergent or not trustK;
      if (meshOnly) {
        // Agreement with the mesh outranks agreement with the set.
        fs::copy_file(v.calibration, v.output,
                      fs::copy_options::overwrite_existing);
      } else {
        loc::WriteCalibration(v.output, *adjusted, jointPose, v.image);
      }
      report << "    {\"image\": " << quote(v.image.string())
             << ", \"anchors\": " << v.anchors.size()
             << ", \"mesh_only_rms_px\": " << number(meshRms)
             << ", \"joint_rms_px\": " << number(jointRms)
             << ", \"center_shift\": " << number(shift)
             << ", \"rotation_deg\": "
             << number(turn.angle() * 180.0 / M_PI)
             << ", \"divergent\": " << (isDivergent ? "true" : "false")
             << ", \"emitted\": "
             << (meshOnly ? "\"mesh-only\"" : "\"joint\"") << "}"
             << (i + 1 < views.size() ? "," : "") << "\n";
    }
    report << "  ],\n  \"divergent\": " << divergent << "\n}\n";

    // The set as one scene: one intrinsic, every adjusted pose, ties as
    // structure.
    sfm::SfM_Data scene;
    scene.intrinsics[0] = adjusted;
    for (std::size_t i = 0; i < views.size(); ++i) {
      const auto id = static_cast<openMVG::IndexT>(i);
      scene.views[id] = std::make_shared<sfm::View>(
          views[i].image.string(), id, 0, id, width, height);
      scene.poses[id] = trustK ? ParamsToPose(chosen.poses[i])
                               : views[i].meshPose;
    }
    for (std::size_t t = 0; t < tracks.size(); ++t) {
      sfm::Landmark lm;
      const auto& X = trustK ? chosen.ties[t] : start.ties[t];
      lm.X = Eigen::Vector3d(X[0], X[1], X[2]);
      for (const auto& [vi, x] : tracks[t].obs) {
        lm.obs[static_cast<openMVG::IndexT>(vi)] =
            sfm::Observation(openMVG::Vec2(x[0], x[1]),
                             static_cast<openMVG::IndexT>(t));
      }
      scene.structure[static_cast<openMVG::IndexT>(t)] = lm;
    }
    if (not sfm::Save(scene, args["output-scene"].as<std::string>(),
                      sfm::ESfM_Data(sfm::ALL))) {
      throw std::runtime_error("could not write --output-scene");
    }

    loc::Camera shared;
    shared.fx = shared.fy = emitK[0];
    shared.cx = emitK[1];
    shared.cy = emitK[2];
    shared.width = width;
    shared.height = height;
    shared.disto = {emitK[3], emitK[4], emitK[5]};
    shared.haveDisto = true;
    loc::WriteCameraFile(args["output-camera"].as<std::string>(), shared);

    std::ofstream(args["report"].as<std::string>()) << report.str();
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  if (divergent > 0) {
    std::cout << "WARNING: " << divergent << " of " << views.size()
              << " views diverged from their mesh-only solve and were emitted "
                 "at it. More than an occasional one means the no-movement "
                 "assumption is wrong for this plate (ADR 0013 Decision 5)\n";
  }
  return SUCCESS;
}

// --- qa ----------------------------------------------------------------------

/// Tile side, in pixels, over which one shift is measured.
constexpr int kQaTile = 64;
/// A tile is measured only when this much of it is valid in both images.
constexpr double kQaTileValid = 0.95;
/// Pairs overlapping less than this, as a fraction of the target view's
/// surface pixels, are not compared: too few tiles to say anything.
constexpr double kQaMinOverlap = 0.05;
/// A comparison with fewer measured tiles than this is reported, not graded.
constexpr std::size_t kQaMinTiles = 10;
/// Provisional grades for a comparison's coherent shift, in target-view
/// pixels. Nothing measured sets them yet (ADR 0013, Not settled).
constexpr double kQaReviewPx = 1.0;
constexpr double kQaFailPx = 3.0;

/// One view as QA sees it: its solved camera and the image it was taken with.
struct QaView {
  std::string label;
  loc::Camera camera;
  cv::Mat gray01;
};

auto ReadQaView(const fs::path& calibration,
                const std::optional<fs::path>& image) -> QaView
{
  sfm::SfM_Data scene;
  if (not sfm::Load(scene, calibration.string(),
                    sfm::ESfM_Data(sfm::VIEWS | sfm::INTRINSICS |
                                   sfm::EXTRINSICS))) {
    throw std::runtime_error("could not read " + calibration.string());
  }
  for (const auto& [id, view] : scene.GetViews()) {
    if (not scene.IsPoseAndIntrinsicDefined(view.get())) {
      continue;
    }
    const auto& intrinsic = *scene.GetIntrinsics().at(view->id_intrinsic);
    QaView out;
    out.camera.width = static_cast<int>(intrinsic.w());
    out.camera.height = static_cast<int>(intrinsic.h());
    out.camera = loc::CameraFromSolve(out.camera, intrinsic,
                                      scene.GetPoseOrDie(view.get()));
    const auto path =
        image ? *image : fs::path(scene.s_root_path) / view->s_Img_path;
    out.label = path.filename().string();
    const auto raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
      throw std::runtime_error("could not read " + path.string());
    }
    if (raw.cols != out.camera.width or raw.rows != out.camera.height) {
      throw std::runtime_error(path.string() + " is not the size its "
                               "calibration says");
    }
    out.gray01 = loc::ToGray01(raw);
    return out;
  }
  throw std::runtime_error(calibration.string() + " holds no view with a pose");
}

/// Normalized camera coordinates to the camera's image coordinates, radial
/// distortion included -- openMVG's pinhole_radial_k3.
auto Distort(const loc::Camera& c, const double x, const double y)
    -> cv::Point2d
{
  const auto r2 = x * x + y * y;
  const auto d = 1.0 + c.disto[0] * r2 + c.disto[1] * r2 * r2 +
                 c.disto[2] * r2 * r2 * r2;
  return {c.fx * x * d + c.cx, c.fy * y * d + c.cy};
}

/**
 * @brief A view's image resampled onto its own ideal pinhole grid -- the grid
 *        the renderer casts on, so it lines up with that view's render.
 *
 * Pixel (u, v) of the grid is the ray through image coordinate (u + 0.5,
 * v + 0.5) (`localize_render.cpp`); through the lens it lands at a distorted
 * image coordinate, which is array position (that - 0.5) in the photograph.
 */
auto OnIdealGrid(const QaView& v) -> cv::Mat
{
  const auto& c = v.camera;
  cv::Mat mapX(c.height, c.width, CV_32F);
  cv::Mat mapY(c.height, c.width, CV_32F);
  for (int r = 0; r < c.height; ++r) {
    for (int u = 0; u < c.width; ++u) {
      const auto p = Distort(c, (u + 0.5 - c.cx) / c.fx, (r + 0.5 - c.cy) / c.fy);
      mapX.at<float>(r, u) = static_cast<float>(p.x - 0.5);
      mapY.at<float>(r, u) = static_cast<float>(p.y - 0.5);
    }
  }
  cv::Mat out;
  cv::remap(v.gray01, out, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
            cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
  return out;
}

/// What one comparison measured.
struct Shift {
  std::string source;
  std::string target;
  double overlap{0.0};
  std::size_t tiles{0};
  /// The coherent shift: the length of the component-wise median of the
  /// tiles' shift vectors. This is what is graded. Misregistration moves the
  /// tiles together, so it survives the median; per-tile noise points every
  /// way and cancels in it.
  double shiftPx{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Vector2d shift{Eigen::Vector2d::Zero()};
  /// The spread: the median and 90th percentile of the tiles' shift
  /// *magnitudes*. A magnitude cannot be negative, so noise alone gives these
  /// a floor well above zero; they say how noisy the measurement was, not how
  /// far off the view is.
  double medianPx{std::numeric_limits<double>::quiet_NaN()};
  double p90Px{std::numeric_limits<double>::quiet_NaN()};
};

/**
 * @brief Measure how far @p moving sits from @p fixed, tile by tile.
 *
 * Both are on the same grid; @p valid marks where both hold surface. Each is
 * locally normalized first -- the two are different exposures, possibly
 * different bands -- and each tile's shift is the phase correlation peak.
 */
auto MeasureShift(const cv::Mat& fixed01, const cv::Mat& moving01,
                  const cv::Mat& valid, Shift& out) -> void
{
  // NaN would poison the normalization's blurs; the mask says where it was.
  cv::Mat a = fixed01.clone();
  cv::Mat b = moving01.clone();
  a.setTo(0.0, ~valid);
  b.setTo(0.0, ~valid);
  cv::Mat na;
  cv::Mat nb;
  loc::LocalNormalize(a, 15.0, 3.0).convertTo(na, CV_32F);
  loc::LocalNormalize(b, 15.0, 3.0).convertTo(nb, CV_32F);
  cv::Mat window;
  cv::createHanningWindow(window, {kQaTile, kQaTile}, CV_32F);

  std::vector<double> mags;
  std::vector<double> dxs;
  std::vector<double> dys;
  for (int y = 0; y + kQaTile <= valid.rows; y += kQaTile) {
    for (int x = 0; x + kQaTile <= valid.cols; x += kQaTile) {
      const cv::Rect tile(x, y, kQaTile, kQaTile);
      if (cv::countNonZero(valid(tile)) <
          kQaTileValid * kQaTile * kQaTile) {
        continue;
      }
      cv::Scalar mean;
      cv::Scalar sd;
      cv::meanStdDev(na(tile), mean, sd);
      if (sd[0] < 5.0) {
        continue;  // nothing in it to correlate
      }
      double response = 0.0;
      const auto s =
          cv::phaseCorrelate(na(tile), nb(tile), window, &response);
      if (response < 0.05) {
        continue;
      }
      // phaseCorrelate reports where `moving` sits relative to `fixed`.
      mags.push_back(std::hypot(s.x, s.y));
      dxs.push_back(s.x);
      dys.push_back(s.y);
    }
  }
  out.tiles = mags.size();
  if (mags.empty()) {
    return;
  }
  const auto median = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  };
  out.shift = Eigen::Vector2d(median(dxs), median(dys));
  out.shiftPx = out.shift.norm();
  std::sort(mags.begin(), mags.end());
  out.medianPx = mags[mags.size() / 2];
  out.p90Px = mags[std::min(mags.size() - 1, mags.size() * 9 / 10)];
}

/**
 * @brief @p source's image as @p target sees the surface, on @p target's ideal
 *        grid, and where that is valid.
 *
 * Each surface point @p target sees (its render's position map) is projected
 * into @p source with @p source's distortion, kept only where @p source sees
 * that same point -- its own render's depth there agrees -- and sampled.
 */
auto WarpInto(const QaView& source, const loc::Render& sourceRender,
              const loc::Render& targetRender, cv::Mat& warped,
              cv::Mat& valid) -> void
{
  const auto& c = source.camera;
  const auto rows = targetRender.position.rows;
  const auto cols = targetRender.position.cols;
  cv::Mat mapX(rows, cols, CV_32F, cv::Scalar(-1));
  cv::Mat mapY(rows, cols, CV_32F, cv::Scalar(-1));
  valid = cv::Mat::zeros(rows, cols, CV_8U);
  for (int r = 0; r < rows; ++r) {
    for (int u = 0; u < cols; ++u) {
      const auto& p = targetRender.position.at<cv::Vec3f>(r, u);
      if (not std::isfinite(p[0])) {
        continue;
      }
      const Eigen::Vector3d xc = c.R * Eigen::Vector3d(p[0], p[1], p[2]) + c.t;
      if (xc[2] <= 0.0) {
        continue;
      }
      const auto nx = xc[0] / xc[2];
      const auto ny = xc[1] / xc[2];
      // Occlusion, on the source's ideal grid where its depth lives.
      const auto iu = static_cast<int>(std::floor(c.fx * nx + c.cx));
      const auto iv = static_cast<int>(std::floor(c.fy * ny + c.cy));
      if (iu < 0 or iv < 0 or iu >= c.width or iv >= c.height) {
        continue;
      }
      const auto seen = sourceRender.depth.at<float>(iv, iu);
      if (not std::isfinite(seen) or
          std::abs(seen - xc[2]) > 0.005 * xc[2]) {
        continue;
      }
      const auto d = Distort(c, nx, ny);
      if (d.x < 0.5 or d.y < 0.5 or d.x > c.width - 1.5 or
          d.y > c.height - 1.5) {
        continue;
      }
      mapX.at<float>(r, u) = static_cast<float>(d.x - 0.5);
      mapY.at<float>(r, u) = static_cast<float>(d.y - 0.5);
      valid.at<unsigned char>(r, u) = 255;
    }
  }
  cv::remap(source.gray01, warped, mapX, mapY, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(0));
}

auto Grade(const Shift& s) -> std::string
{
  if (s.tiles < kQaMinTiles) {
    return "unmeasured";
  }
  return s.shiftPx > kQaFailPx     ? "fail"
         : s.shiftPx > kQaReviewPx ? "review"
                                   : "pass";
}

auto ShiftJson(const Shift& s, const bool graded) -> std::string
{
  std::ostringstream o;
  o << "{\"source\": " << quote(s.source) << ", \"target\": "
    << quote(s.target) << ", \"overlap\": " << number(s.overlap)
    << ", \"tiles\": " << s.tiles << ", \"shift_px\": " << number(s.shiftPx)
    << ", \"shift\": [" << number(s.shift[0]) << ", " << number(s.shift[1])
    << "], \"spread_median_px\": " << number(s.medianPx)
    << ", \"spread_p90_px\": " << number(s.p90Px);
  if (graded) {
    o << ", \"grade\": " << quote(Grade(s));
  }
  return o.str() + "}";
}

auto qa(const std::vector<std::string>& argv) -> int
{
  // clang-format off
  po::options_description opts("qa");
  opts.add_options()
    ("help,h", "print help message")
    ("calibration", po::value<std::vector<std::string>>()->composing(), "a view's final calibration; its image is the one the calibration names. Repeated, once per view")
    ("mesh", po::value<std::string>(), "the mesh the views were localized against; only its geometry is used, for where each view sees the surface")
    ("overhead-calibration", po::value<std::string>(), "the overhead view's calibration, for the overhead comparison (partly independent, ADR 0013 Decision 6)")
    ("overhead-image", po::value<std::string>(), "the overhead image to compare against; with --overhead-calibration")
    ("reference-mesh", po::value<std::string>(), "the IR reference mesh, for the circular comparison: reported, never graded")
    ("report", po::value<std::string>(), "write the JSON report here")
    ("threads", po::value<int>()->default_value(0), "renderer threads; 0 uses all")
  ;
  // clang-format on
  po::variables_map args;
  if (const auto code = parseOrExit("qa", opts, argv,
                                    {"calibration", "mesh", "report"}, args);
      code >= 0) {
    return code;
  }
  if (args.count("overhead-calibration") != args.count("overhead-image")) {
    std::cerr << "ERROR: --overhead-calibration and --overhead-image go "
                 "together\n";
    return BAD_ARG;
  }
  if (not requireFiles(args, {"calibration", "mesh", "overhead-calibration",
                              "overhead-image", "reference-mesh"})) {
    return IO_ERROR;
  }
  const auto threads = static_cast<std::size_t>(args["threads"].as<int>());

  std::vector<QaView> views;
  std::vector<loc::Render> renders;
  std::vector<cv::Mat> ideal;
  std::vector<Shift> pairs;
  std::vector<Shift> overhead;
  std::vector<Shift> circular;
  try {
    const auto mesh = loc::ReadMeshBundle(args["mesh"].as<std::string>());
    for (const auto& c : args["calibration"].as<std::vector<std::string>>()) {
      views.push_back(ReadQaView(c, std::nullopt));
      renders.push_back(loc::RenderMesh(mesh, views.back().camera, threads));
      ideal.push_back(OnIdealGrid(views.back()));
    }

    // Overlap consistency: every pair, each way round being the same
    // measurement, so i < j. Independent of the overhead entirely.
    for (std::size_t j = 0; j < views.size(); ++j) {
      const auto surface = cv::countNonZero(renders[j].depth == renders[j].depth);
      for (std::size_t i = 0; i < j; ++i) {
        Shift s;
        s.source = views[i].label;
        s.target = views[j].label;
        cv::Mat warped;
        cv::Mat valid;
        WarpInto(views[i], renders[i], renders[j], warped, valid);
        valid &= (ideal[j] == ideal[j]);
        s.overlap = surface ? static_cast<double>(cv::countNonZero(valid)) /
                                  surface
                            : 0.0;
        if (s.overlap < kQaMinOverlap) {
          continue;
        }
        MeasureShift(ideal[j], warped, valid, s);
        pairs.push_back(s);
        std::cout << "  " << s.source << " -> " << s.target << ": overlap "
                  << s.overlap << ", " << s.tiles << " tiles, shift "
                  << s.shiftPx << " px (" << Grade(s) << ")\n";
      }
    }

    if (args.count("overhead-calibration")) {
      const auto top = ReadQaView(args["overhead-calibration"].as<std::string>(),
                                  fs::path(args["overhead-image"]
                                               .as<std::string>()));
      const auto topRender = loc::RenderMesh(mesh, top.camera, threads);
      for (std::size_t j = 0; j < views.size(); ++j) {
        Shift s;
        s.source = top.label;
        s.target = views[j].label;
        cv::Mat warped;
        cv::Mat valid;
        WarpInto(top, topRender, renders[j], warped, valid);
        valid &= (ideal[j] == ideal[j]);
        s.overlap = 1.0;
        MeasureShift(ideal[j], warped, valid, s);
        overhead.push_back(s);
        std::cout << "  overhead -> " << s.target << ": " << s.tiles
                  << " tiles, shift " << s.shiftPx << " px (" << Grade(s)
                  << ")\n";
      }
    }

    if (args.count("reference-mesh")) {
      const auto ref =
          loc::ReadMeshBundle(args["reference-mesh"].as<std::string>());
      for (std::size_t j = 0; j < views.size(); ++j) {
        const auto render = loc::RenderMesh(ref, views[j].camera, threads);
        Shift s;
        s.source = "reference render";
        s.target = views[j].label;
        cv::Mat valid = render.textured & (ideal[j] == ideal[j]);
        s.overlap = 1.0;
        MeasureShift(ideal[j], loc::ToGray01(render.gray), valid, s);
        circular.push_back(s);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }

  // The worst graded result, which is what a caller acts on.
  std::string worst = "pass";
  const auto rank = [](const std::string& g) {
    return g == "fail" ? 2 : g == "review" ? 1 : 0;
  };
  for (const auto* set : {&pairs, &overhead}) {
    for (const auto& s : *set) {
      if (rank(Grade(s)) > rank(worst)) {
        worst = Grade(s);
      }
    }
  }

  std::ofstream f(args["report"].as<std::string>());
  f << "{\n  \"tool\": \"pgs-localize-joint qa\",\n"
    << "  \"thresholds_px\": {\"review\": " << number(kQaReviewPx)
    << ", \"fail\": " << number(kQaFailPx)
    << ", \"min_tiles\": " << kQaMinTiles << ", \"provisional\": true},\n"
    << "  \"notes\": " << quote(
         "Shifts are in the target view's pixels. The graded 'shift_px' is "
         "the length of the tiles' median shift vector -- coherent "
         "misregistration; 'spread_*' is how noisy the tiles were. "
         "'pairs' compares hand-held "
         "views with each other and is independent of the overhead. "
         "'overhead' is partly independent: the overhead was solved "
         "separately, but both rest on the same mesh, so an error in the "
         "mesh is shared and cannot show here. 'circular' compares each view "
         "with the texture it was registered to; it shows convergence, not "
         "accuracy, and gates nothing. No view sits on the mesh better than "
         "the reconstruction allows: ADR 0011 attributes ~1.9 px of its "
         "~2 px floor to the mesh and its texture (ADR 0013 Decision 2).")
    << ",\n  \"pairs\": [";
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    f << (i ? ",\n    " : "\n    ") << ShiftJson(pairs[i], true);
  }
  f << "\n  ],\n  \"overhead\": [";
  for (std::size_t i = 0; i < overhead.size(); ++i) {
    f << (i ? ",\n    " : "\n    ") << ShiftJson(overhead[i], true);
  }
  f << "\n  ],\n  \"circular\": [";
  for (std::size_t i = 0; i < circular.size(); ++i) {
    f << (i ? ",\n    " : "\n    ") << ShiftJson(circular[i], false);
  }
  f << "\n  ],\n  \"worst\": " << quote(worst) << "\n}\n";
  if (not f) {
    std::cerr << "ERROR: could not write the report\n";
    return IO_ERROR;
  }
  std::cout << "QA: " << pairs.size() << " overlapping pairs, worst grade "
            << worst << "\n";
  return SUCCESS;
}

} // namespace

auto main(int argc, char* argv[]) -> int
{
  const std::map<std::string, int (*)(const std::vector<std::string>&)>
      commands{{"position-map", positionMap}, {"view", view},
               {"adjust", adjust}, {"qa", qa}};

  const std::string first = argc > 1 ? argv[1] : "";
  const auto it = commands.find(first);
  if (it == commands.end()) {
    std::cout << "Usage: pgs-localize-joint <position-map|view|adjust|qa> "
                 "[options]\n"
                 "       pgs-localize-joint <command> --help\n\n"
                 "Localize a hand-held view set against a solved scene "
                 "(ADR 0013).\n";
    return (first.empty() or first == "--help" or first == "-h") ? HELP
                                                                 : BAD_ARG;
  }
  return it->second({argv + 2, argv + argc});
}
