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
 *
 * Everything here is built from `localize_core`, the code `pgs-localize` runs
 * and ADR 0011 measured; this file is orchestration over it. The command line
 * is mirrored by `pgs_recon/viewset/tools.py`; the two are edited together.
 */

#include <algorithm>
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

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
  SOLVE_FAILED = 6,
  NOT_IMPLEMENTED = 7
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

auto notImplemented(const std::string& what) -> int
{
  std::cerr << "pgs-localize-joint: " << what << " is not implemented yet "
               "(ADR 0013).\n";
  return NOT_IMPLEMENTED;
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

auto adjust(const std::vector<std::string>& argv) -> int
{
  // clang-format off
  po::options_description opts("adjust");
  opts.add_options()
    ("help,h", "print help message")
    ("calibration", po::value<std::vector<std::string>>()->composing(), "a view's mesh-only calibration. Repeated, once per view")
    ("correspondences", po::value<std::vector<std::string>>()->composing(), "that view's correspondences. Repeated, paired with --calibration by position")
    ("output-calibration", po::value<std::vector<std::string>>()->composing(), "where that view's final calibration goes. Repeated, paired by position")
    ("tie-scene", po::value<std::vector<std::string>>()->composing(), "a group's own SfM solve; its tracks become free tie points. Views are matched to it by image filename")
    ("output-scene", po::value<std::string>(), "every view at its adjusted pose, sharing one intrinsic")
    ("output-camera", po::value<std::string>(), "the shared K and distortion as a flat camera file")
    ("report", po::value<std::string>(), "JSON report: focal cost curve, per-view divergence, gates")
    ("camera", po::value<std::string>(), "starting K and distortion")
    ("fix-intrinsic", po::bool_switch(), "hold K at --camera (a board-calibrated camera)")
    ("focal-sweep", po::value<std::string>(), "lo:hi:steps, in pixels, for the focal cost curve. The curve always runs; unset picks a range around the starting focal")
    ("threads", po::value<int>()->default_value(0), "worker threads; 0 uses all")
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
  const auto n = args["calibration"].as<std::vector<std::string>>().size();
  if (args["correspondences"].as<std::vector<std::string>>().size() != n or
      args["output-calibration"].as<std::vector<std::string>>().size() != n) {
    std::cerr << "ERROR: --calibration, --correspondences and "
                 "--output-calibration are one each per view\n";
    return BAD_ARG;
  }
  if (args["fix-intrinsic"].as<bool>() and not args.count("camera")) {
    std::cerr << "ERROR: --fix-intrinsic holds K at --camera; give one\n";
    return BAD_ARG;
  }
  if (not requireFiles(args, {"calibration", "correspondences", "tie-scene",
                              "camera"})) {
    return IO_ERROR;
  }

  // Task 4. One openMVG SfM_Data: every view, one shared pinhole_radial_k3
  // intrinsic, each view's correspondences as landmarks whose structure is held
  // constant, each tie scene's tracks as free landmarks. Bundle_Adjustment_Ceres
  // with a robust loss, the intrinsic refined unless --fix-intrinsic.
  //
  // Focal cost curve: re-solve with f held at each --focal-sweep step; report
  // the curve and whether it has a minimum. A flat curve is reported, not
  // hidden (Decision 4).
  //
  // Divergence: each view's adjusted pose against its mesh-only one. Past the
  // gate, emit the mesh-only pose and flag the view (Decision 5).
  return notImplemented("adjust");
}

} // namespace

auto main(int argc, char* argv[]) -> int
{
  const std::map<std::string, int (*)(const std::vector<std::string>&)>
      commands{{"position-map", positionMap}, {"view", view},
               {"adjust", adjust}};

  const std::string first = argc > 1 ? argv[1] : "";
  const auto it = commands.find(first);
  if (it == commands.end()) {
    std::cout << "Usage: pgs-localize-joint <position-map|view|adjust> "
                 "[options]\n"
                 "       pgs-localize-joint <command> --help\n\n"
                 "Localize a hand-held view set against a solved scene "
                 "(ADR 0013).\n";
    return (first.empty() or first == "--help" or first == "-h") ? HELP
                                                                 : BAD_ARG;
  }
  return it->second({argv + 2, argv + argc});
}
