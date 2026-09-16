/**
 * @file
 * @brief `pgs-localize`: put a camera into a solved scene and emit a reusable
 *        calibration.
 *
 * Two ways to find the 3D<->2D correspondences a resection needs:
 *
 *   - **sparse**: match the query's descriptors against the SfM structure's,
 *     the way `openMVG_main_SfM_Localization` does (`--input-scene` plus
 *     `--matches-dir`);
 *   - **render-and-match**: render a same-modality textured mesh from a prior
 *     pose, match render against query, and lift the render-side keypoints
 *     through the render's own position map (`--mesh` plus `--camera`).
 *
 * Give all four and they chain: sparse resects, its pose becomes the prior, and
 * render-and-match refines from there. That is the recommended mode -- sparse
 * is a poor answer but a fine prior, and render-and-match converges from one
 * 861 px off.
 *
 * See ADR 0011. The tool knows nothing about modality: it assumes the caller
 * paired a mesh and a query that look like each other.
 */

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <educelab/core/utils/String.hpp>

#include <opencv2/imgcodecs.hpp>

#include "localize_camera.hpp"
#include "localize_match.hpp"
#include "localize_render.hpp"
#include "localize_solve.hpp"
#include "localize_sparse.hpp"
#include "Npy.hpp"

namespace el = educelab;
namespace loc = pgs::localize;
namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace
{

enum EXIT_CODE {
  SUCCESS = 0,
  HELP = 1,
  BAD_ARG = 2,
  IO_ERROR = 3,
  SELF_TEST_FAILED = 4,
  NO_CORRESPONDENCES = 5,
  SOLVE_FAILED = 6
};

/// A JSON string body: the escapes a filesystem path can actually contain.
/// `decimate.cpp` carries its own copy of these four; two hand-rolled reports
/// is not yet a library.
auto escape(const std::string& text) -> std::string
{
  std::string out;
  for (const auto c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

auto quote(const std::string& text) -> std::string
{
  return "\"" + escape(text) + "\"";
}

auto boolean(bool value) -> std::string { return value ? "true" : "false"; }

/// A JSON number. libcore's conversion rather than a stream: locale-independent
/// -- an `LC_NUMERIC` spelling the separator `,` would make the report invalid
/// JSON -- and shortest round-trip. JSON has no infinity or NaN, and an
/// unparseable report is worth less than one admitting a value was not a number.
auto number(double value) -> std::string
{
  return std::isfinite(value) ? el::to_string(value) : "null";
}

/// A path, or `null` for one never given.
auto pathOrNull(const fs::path& p) -> std::string
{
  return p.empty() ? std::string{"null"} : quote(p.string());
}

/// How a gate row came out. `unset` is not a pass: it says the gate had
/// nothing to judge, which is the standoff gate's state whenever no expected
/// distance was supplied.
enum class Gate { Unset, Pass, Review, Fail };

auto to_string(const Gate g) -> std::string
{
  switch (g) {
    case Gate::Pass:
      return "pass";
    case Gate::Review:
      return "review";
    case Gate::Fail:
      return "fail";
    default:
      return "unset";
  }
}

/// Grade a value whose *larger* readings are worse.
auto gradeAbove(double value, double review, double fail) -> Gate
{
  if (not std::isfinite(value)) {
    return Gate::Unset;
  }
  if (value > fail) {
    return Gate::Fail;
  }
  return value > review ? Gate::Review : Gate::Pass;
}

/// Grade a count whose *smaller* readings are worse.
auto gradeBelow(std::size_t value, std::size_t review, std::size_t fail) -> Gate
{
  if (value < fail) {
    return Gate::Fail;
  }
  return value < review ? Gate::Review : Gate::Pass;
}

/**
 * @brief The published gate thresholds, measured over twelve datasets
 *        (`spectral-selfcal-workplan.md` §12).
 *
 * Baked in rather than flagged, unlike `--expected-standoff`, and the
 * difference is the point: these four are properties of the *method* -- what
 * this matcher, this solve and this held-out split produce on a mesh that
 * registers -- while a standoff of 154.2 cm is a property of one rig and
 * belongs to the caller.
 *
 *   held-out rms_px  observed 1.81-2.53 (mean 2.10, sd 0.25)
 *   p99_px           observed 3.99-5.65 (mean 4.77)
 *   |bias|_px        observed 0.06-0.40 (mean 0.17)
 *   n_inliers        observed 162-1946 (median 601)
 */
constexpr double kRmsReview = 3.5;
constexpr double kRmsFail = 5.0;
constexpr double kP99Review = 8.0;
constexpr double kP99Fail = 12.0;
constexpr double kBiasReview = 1.2;
constexpr double kBiasFail = 3.0;
constexpr std::size_t kInliersReview = 250;
constexpr std::size_t kInliersFail = 64;

/// Everything the report says, gathered as the run produces it.
struct Report {
  fs::path image;
  fs::path mesh;
  fs::path inputScene;
  fs::path matchesDir;
  fs::path camera;
  std::string kSource{"dlt"};
  std::string priorSource{"camera"};
  std::string backend{"render-and-match"};

  loc::MatchReport matches;
  std::size_t meshCorrespondences{0};
  std::size_t sparseCorrespondences{0};

  std::size_t inliers{0};
  /// Why the fitted intrinsic was refused, or empty when it was not.
  std::string intrinsicComplaint;
  double spreadX{0.0};
  double spreadY{0.0};
  int passes{1};

  loc::HeldoutResult heldout;

  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double standoff{0.0};
  double expectedStandoff{0.0};
  double standoffTolerance{0.0};
  bool standoffGated{false};

  Gate gateRms{Gate::Unset};
  Gate gateP99{Gate::Unset};
  Gate gateBias{Gate::Unset};
  Gate gateInliers{Gate::Unset};
  Gate gateStandoff{Gate::Unset};
};

/// The worst grade in the report, which is what a caller acts on.
auto worst(const Report& r) -> Gate
{
  Gate out = Gate::Unset;
  for (const auto g : {r.gateRms, r.gateP99, r.gateBias, r.gateInliers,
                       r.gateStandoff}) {
    if (g == Gate::Fail) {
      return Gate::Fail;
    }
    if (g == Gate::Review or (g == Gate::Pass and out == Gate::Unset)) {
      out = g;
    }
  }
  return out;
}

void grade(Report& r)
{
  r.gateInliers = gradeBelow(r.inliers, kInliersReview, kInliersFail);
  if (r.heldout.ok) {
    r.gateRms = gradeAbove(r.heldout.stats.rmsPx, kRmsReview, kRmsFail);
    r.gateP99 = gradeAbove(r.heldout.stats.p99Px, kP99Review, kP99Fail);
    r.gateBias = gradeAbove(r.heldout.stats.biasPx, kBiasReview, kBiasFail);
  }
  if (r.standoffGated) {
    // The flag is the review threshold; twice it fails. That is the shape the
    // measured table has -- review past 1 cm, fail past 2 -- and it keeps the
    // caller stating one number rather than two.
    r.gateStandoff = gradeAbove(std::abs(r.standoff - r.expectedStandoff),
                                r.standoffTolerance,
                                2.0 * r.standoffTolerance);
  }
}

/**
 * @brief Write the QA report.
 *
 * The keys are the interface: a caller goes to this file for the numbers that
 * decide whether a registration shipped, so renaming one is a breaking change.
 * `heldout` is named that and not `independent` on purpose -- the matches were
 * found using a render from the pose being scored, so it bounds the fit and not
 * the registration.
 */
void writeReport(const fs::path& path, const Report& r)
{
  std::ofstream f{path};
  if (not f.is_open()) {
    throw std::runtime_error("cannot open report for writing: " +
                             path.string());
  }
  const auto& s = r.heldout.stats;
  f << "{\n"
    << "  \"tool\": \"pgs-localize\",\n"
    << "  \"input\": {\n"
    << "    \"image\": " << pathOrNull(r.image) << ",\n"
    << "    \"mesh\": " << pathOrNull(r.mesh) << ",\n"
    << "    \"input_scene\": " << pathOrNull(r.inputScene) << ",\n"
    << "    \"matches_dir\": " << pathOrNull(r.matchesDir) << ",\n"
    << "    \"camera\": " << pathOrNull(r.camera) << "\n"
    << "  },\n"
    << "  \"k_source\": " << quote(r.kSource) << ",\n"
    << "  \"prior_source\": " << quote(r.priorSource) << ",\n"
    << "  \"backend\": " << quote(r.backend) << ",\n"
    << "  \"matches\": {\n"
    << "    \"detected_query\": " << r.matches.detectedQuery << ",\n"
    << "    \"detected_render\": " << r.matches.detectedRender << ",\n"
    << "    \"lowe_passed\": " << r.matches.lowePassed << ",\n"
    << "    \"similarity_inliers\": " << r.matches.similarityInliers << ",\n"
    << "    \"similarity_scale\": " << number(r.matches.similarityScale)
    << ",\n"
    << "    \"similarity_rotation_deg\": "
    << number(r.matches.similarityRotationDeg) << ",\n"
    << "    \"lk_kept\": " << r.matches.lkKept << ",\n"
    << "    \"lifted\": " << r.matches.lifted << "\n"
    << "  },\n"
    << "  \"solve\": {\n"
    << "    \"mesh_correspondences\": " << r.meshCorrespondences << ",\n"
    << "    \"sparse_correspondences\": " << r.sparseCorrespondences << ",\n"
    << "    \"n_inliers\": " << r.inliers << ",\n"
    << "    \"inlier_spread_px_x\": " << number(r.spreadX) << ",\n"
    << "    \"inlier_spread_px_y\": " << number(r.spreadY) << ",\n"
    << "    \"passes\": " << r.passes << ",\n"
    << "    \"intrinsic_refused\": "
    << (r.intrinsicComplaint.empty() ? std::string{"null"}
                                     : quote(r.intrinsicComplaint))
    << "\n"
    << "  },\n"
    // Null throughout when the split could not be made: a report that says
    // "not measured" is worth more than one quoting a statistic over nine
    // points as though it were the published one.
    << "  \"heldout\": {\n"
    << "    \"split_px\": " << r.heldout.splitPx << ",\n"
    << "    \"fit_count\": " << r.heldout.fitCount << ",\n"
    << "    \"test_count\": " << r.heldout.testCount << ",\n"
    << "    \"rms_px\": " << (r.heldout.ok ? number(s.rmsPx) : "null") << ",\n"
    << "    \"p99_px\": " << (r.heldout.ok ? number(s.p99Px) : "null") << ",\n"
    << "    \"bias_px\": " << (r.heldout.ok ? number(s.biasPx) : "null")
    << ",\n"
    << "    \"bias_x_px\": " << (r.heldout.ok ? number(s.biasXPx) : "null")
    << ",\n"
    << "    \"bias_y_px\": " << (r.heldout.ok ? number(s.biasYPx) : "null")
    << "\n"
    << "  },\n"
    // The unit is the scene's, whatever that is; the key says cm because this
    // rig's scenes are in centimetres and the name is the interface.
    << "  \"pose\": {\n"
    << "    \"center_cm\": [" << number(r.center[0]) << ", "
    << number(r.center[1]) << ", " << number(r.center[2]) << "],\n"
    << "    \"standoff_cm\": " << number(r.standoff) << ",\n"
    << "    \"expected_standoff_cm\": "
    << (r.standoffGated ? number(r.expectedStandoff) : "null") << ",\n"
    << "    \"standoff_tol_cm\": "
    << (r.standoffGated ? number(r.standoffTolerance) : "null") << ",\n"
    << "    \"standoff_ok\": "
    << (r.standoffGated ? boolean(r.gateStandoff == Gate::Pass) : "null")
    << "\n"
    << "  },\n"
    << "  \"gates\": {\n"
    << "    \"rms\": " << quote(to_string(r.gateRms)) << ",\n"
    << "    \"p99\": " << quote(to_string(r.gateP99)) << ",\n"
    << "    \"bias\": " << quote(to_string(r.gateBias)) << ",\n"
    << "    \"inliers\": " << quote(to_string(r.gateInliers)) << ",\n"
    << "    \"standoff\": " << quote(to_string(r.gateStandoff)) << ",\n"
    << "    \"worst\": " << quote(to_string(worst(r))) << "\n"
    << "  }\n"
    << "}\n";
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  // clang-format off
  po::options_description required("required arguments");
  required.add_options()
    ("help,h", "print help message")
    ("image,i", po::value<std::string>(), "the query image to localize")
  ;

  po::options_description outputs("outputs (at least one of --output-calibration / --output-camera required)");
  outputs.add_options()
    ("output-calibration", po::value<std::string>(), "write the one-view openMVG calibration JSON here. This is what pgs-retexture --calibration consumes")
    ("output-camera", po::value<std::string>(), "write the solved camera in the flat camera-file format here. Valid input to --camera, and what registration-toolkit's --camera-file reads")
    ("report", po::value<std::string>(), "write the JSON QA report here")
    ("qa-render", po::value<std::string>(), "write the render at the SOLVED pose here, plus a difference image against the query beside it. A frame error is a proper rigid motion: it moves no residual, and looking is the only thing that catches it")
  ;

  po::options_description sources("correspondence sources (at least one; both is the recommended mode)");
  sources.add_options()
    ("mesh,m", po::value<std::string>(), "textured mesh for the render-and-match backend, in the same modality as --image. Needs a pose prior, from --camera or from a sparse resection")
    ("camera,c", po::value<std::string>(), "camera file supplying K (always) and the prior pose (mesh backend). Flat key/value: fx fy cx cy width height [k1 k2 k3] and pose, 16 row-major floats of the 4x4 WORLD-TO-CAMERA matrix")
    ("input-scene,s", po::value<std::string>(), "solved openMVG scene (sfm_data.bin or .json) for the sparse backend")
    ("matches-dir", po::value<std::string>(), "the scene's matches directory, holding image_describer.json and the per-view .feat/.desc regions. Required with --input-scene")
    ("sfm-transform", po::value<std::string>(), "4x4 float64 .npy change of frame, applied to the scene AT LOAD so the resection happens in the mesh's frame. Pass it when localizing against a mesh that pgs-center/pgs-sfm-orient moved; omit it when the scene and the mesh share a frame. Unchecked either way -- whether a centered mesh exists is not something this tool can see")
    ("mask", po::value<std::string>(), "detection mask for the QUERY image only: features are kept where the mask is non-zero. Off unless given. Generate one with pgs-generate-mask")
  ;

  po::options_description intrinsics("intrinsics and resection");
  intrinsics.add_options()
    ("camera-model", po::value<std::string>()->default_value("pinhole"), "intrinsic model DLT fits when no --camera is given: pinhole, radial1, radial3, radial3tangential, fisheye. Pinhole by default -- a long-focal overhead view has too little depth baseline to constrain distortion, and radial3 overfits it (ADR 0002). Ignored when --camera supplies K")
    ("resection-method", po::value<std::string>()->default_value("auto"), "absolute pose solver: auto, dlt, p3p-ke, p3p-kneip, p3p-nordberg, p3p-ding, up2p. auto is P3P when K is known and DLT when it is not")
    ("residual-error", po::value<double>()->default_value(0.0), "upper bound, in pixels, on the resection residual. 0 leaves openMVG's own")
  ;

  po::options_description matching("render-and-match options");
  matching.add_options()
    ("describer-preset", po::value<std::string>()->default_value("HIGH"), "openMVG describer preset for the render/query pair: NORMAL, HIGH, ULTRA. Only the mesh backend honors it -- the sparse backend must describe the query exactly as matches-dir was described. HIGH by default, measured over twelve datasets: ULTRA (which upscales the query 2x before building the scale space) costs 5.1x the wall clock and buys better fit statistics -- held-out rms 1.873 against 1.920, p99 5.05 against 5.63 -- but not a better pose. The two agree on the solved standoff to 36 um on average and 170 um at worst, which is inside the noise floor of the matcher itself, and both agree with independently solved reference poses to about 0.13 mm. Reach for ULTRA when match counts are genuinely thin, not for accuracy")
    ("matcher", po::value<std::string>()->default_value("bruteforce-l2"), "descriptor matcher: bruteforce-l2, cascade-hashing. One image pair per run, so the approximation cascade hashing buys speed with is not worth having to rule out when match counts look thin")
    ("lowe-ratio", po::value<double>()->default_value(0.85, "0.85"), "Lowe ratio test threshold for the render/query match")
    ("normalize-sigma", po::value<double>()->default_value(15.0, "15"), "Gaussian sigma, in pixels, of the local mean and SD both images are normalized by. Non-optional: it is what makes \"same modality\" a weak enough requirement to be useful, and without it the match rate collapses")
    ("normalize-clip", po::value<double>()->default_value(3.0, "3"), "clip the normalized images to this many local SDs")
    ("similarity-threshold", po::value<double>()->default_value(6.0, "6"), "inlier threshold, in pixels, for the 2D-similarity RANSAC that filters the matches. Only an outlier filter -- the true relation is perspective-plus-relief -- but it is what rejects the sample square, which is not rigid with respect to the sample. The orthographic prototype used 8 px; 6 is written against the camera-projection render this tool builds")
    ("similarity-iterations", po::value<int>()->default_value(50000), "hard iteration cap for the similarity RANSAC; it exits early once the consensus justifies it")
    ("refine-matches", po::value<bool>()->default_value(true), "refine each match's QUERY-side location with Lucas-Kanade before the solve. Worth a measured 6%")
    ("lk-window", po::value<int>()->default_value(21), "Lucas-Kanade search window, in pixels, per side")
    ("lk-levels", po::value<int>()->default_value(3), "Lucas-Kanade pyramid levels")
  ;

  po::options_description solving("solve options");
  solving.add_options()
    ("fuse-sparse", po::value<bool>()->default_value(true), "in chained mode, put the sparse correspondences into the same solve as the render-and-match ones rather than using the sparse resection only as the prior. Possible at all only because --sfm-transform is applied to the scene at load, which puts both sets in one frame. Measured on PHerc0013Cr04: fusing moved the emitted standoff from 0.18 mm off the reference to 0.02 mm, and moved the held-out rms from 1.61 to 1.97 px -- but that second number is scored over a different population, since the test half now contains sparse correspondences too. Turn it off to get a held-out statistic directly comparable with the published thresholds, which were measured on render-and-match alone")
    ("second-pass-min-inliers", po::value<int>()->default_value(0), "re-render from the refined pose and solve again if the first pass found fewer inliers than this. 0 is off, and off is right: a second pass was measured at 1.820 -> 1.821 -> 1.838 px over twelve datasets. It is a guard on a thin first pass, not an improvement")
    ("seed", po::value<unsigned int>(), "seed the RANSACs. Unset is nondeterministic, which is the default so that run-to-run variance stays visible")
    ("threads", po::value<int>()->default_value(0), "worker threads for the renderer, which is the parallel part; 0 uses all. openMVG's own OpenMP paths are compiled out here -- it sets OPENMVG_USE_OPENMP in its own build and does not export it -- so feature description and regions loading run single-threaded whatever this says")
  ;

  po::options_description qa("QA and gates");
  qa.add_options()
    ("heldout-split", po::value<int>()->default_value(400), "side, in pixels, of the spatial checkerboard that splits the correspondences: one colour solves, the other scores, then the emitted pose is re-solved on all inliers. The published gate thresholds are measured this way and do not transfer to a plain inlier RMS. Named \"heldout\" and not \"independent\" -- the matches were still found with a render from the pose being scored")
    ("expected-standoff", po::value<double>()->default_value(0.0), "expected camera-to-scene distance along the optical axis, in the scene's units. Overrides an expected_standoff key in --camera, which is where it belongs by default: it is a property of the rig with the same lifetime as K, and the camera file's own pose already carries that distance. Pass 0 explicitly to disarm a gate the file arms. Gates only when set; the solved standoff is always reported. It is the only gate independent of the correspondence set, which is what makes it the one that catches a frame error or a focal absorbed into depth")
    ("standoff-tolerance", po::value<double>()->default_value(0.0), "how far the solved standoff may sit from the expected one before the gate reads review; twice that fails. Overrides a standoff_tolerance key in --camera")
    ("self-test", po::bool_switch(), "render a generated mesh from a known pose, check the position map round-trips, and exit")
  ;

  po::options_description parser("options");
  parser.add(required).add(outputs).add(sources).add(intrinsics).add(matching)
        .add(solving).add(qa);
  // clang-format on

  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, parser), args);
    if (argc == 1 or args.count("help") > 0) {
      std::cout << parser << "\n";
      return HELP;
    }
    po::notify(args);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return BAD_ARG;
  }

  if (args["self-test"].as<bool>()) {
    return loc::SelfTestRender(std::cout) ? SUCCESS : SELF_TEST_FAILED;
  }

  // --- Validate the shape of the run --------------------------------------
  if (args.count("image") == 0) {
    std::cerr << "ERROR: --image is required\n";
    return BAD_ARG;
  }
  const auto have = [&args](const char* name) {
    return args.count(name) > 0;
  };
  const auto pathArg = [&args](const char* name) {
    return fs::absolute(args[name].as<std::string>());
  };

  if (not have("output-calibration") and not have("output-camera")) {
    std::cerr << "ERROR: give at least one of --output-calibration or "
                 "--output-camera; a run that emits neither has solved "
                 "nothing anybody can use\n";
    return BAD_ARG;
  }
  if (not have("mesh") and not have("input-scene")) {
    std::cerr << "ERROR: give at least one correspondence source: --mesh (with "
                 "a prior pose) or --input-scene (with --matches-dir)\n";
    return BAD_ARG;
  }
  if (have("input-scene") and not have("matches-dir")) {
    std::cerr << "ERROR: --input-scene needs --matches-dir; the query has to be "
                 "described the way the scene's regions were\n";
    return BAD_ARG;
  }
  if (have("mesh") and not have("camera") and not have("input-scene")) {
    std::cerr << "ERROR: --mesh renders from a prior pose and nothing here "
                 "supplies one. Pass --camera (whose pose becomes the prior) "
                 "or --input-scene (whose resection becomes it)\n";
    return BAD_ARG;
  }

  const auto image = pathArg("image");

  Report report;
  report.image = image;
  if (have("mesh")) {
    report.mesh = pathArg("mesh");
  }
  if (have("camera")) {
    report.camera = pathArg("camera");
  }
  if (have("input-scene")) {
    report.inputScene = pathArg("input-scene");
    report.matchesDir = pathArg("matches-dir");
  }
  report.backend = have("mesh") ? (have("input-scene") ? "chained"
                                                       : "render-and-match")
                                : "sparse";

  loc::MatchOptions matchOpts;
  loc::Camera camera;
  openMVG::cameras::EINTRINSIC cameraModel{};
  std::string solverName;
  try {
    matchOpts.normalizeSigma = args["normalize-sigma"].as<double>();
    matchOpts.normalizeClip = args["normalize-clip"].as<double>();
    matchOpts.loweRatio = args["lowe-ratio"].as<double>();
    matchOpts.similarityThreshold = args["similarity-threshold"].as<double>();
    matchOpts.similarityIterations = args["similarity-iterations"].as<int>();
    matchOpts.refineMatches = args["refine-matches"].as<bool>();
    matchOpts.lkWindow = args["lk-window"].as<int>();
    matchOpts.lkLevels = args["lk-levels"].as<int>();
    if (have("seed")) {
      matchOpts.seed = args["seed"].as<unsigned int>();
    }
    const auto preset =
        el::to_upper_copy(args["describer-preset"].as<std::string>());
    if (preset == "NORMAL") {
      matchOpts.preset = openMVG::features::NORMAL_PRESET;
    } else if (preset == "HIGH") {
      matchOpts.preset = openMVG::features::HIGH_PRESET;
    } else if (preset == "ULTRA") {
      matchOpts.preset = openMVG::features::ULTRA_PRESET;
    } else {
      std::cerr << "ERROR: unknown --describer-preset: " << preset << "\n";
      return BAD_ARG;
    }
    const auto matcher = el::to_lower_copy(args["matcher"].as<std::string>());
    if (matcher == "bruteforce-l2") {
      matchOpts.matcher = openMVG::matching::BRUTE_FORCE_L2;
    } else if (matcher == "cascade-hashing") {
      matchOpts.matcher = openMVG::matching::CASCADE_HASHING_L2;
    } else {
      std::cerr << "ERROR: unknown --matcher: " << matcher << "\n";
      return BAD_ARG;
    }
    cameraModel = loc::ParseCameraModel(
        el::to_lower_copy(args["camera-model"].as<std::string>()));
    solverName = el::to_lower_copy(args["resection-method"].as<std::string>());
    // Parsed here so a bad name is refused before any work happens; resolved
    // per solve below, because "auto" depends on whether K is known *then* --
    // a DLT sparse resection hands one back for the mesh solve to use.
    (void)loc::ParseSolver(solverName, have("camera"));
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return BAD_ARG;
  }
  if (matchOpts.normalizeSigma <= 0.0 or matchOpts.normalizeClip <= 0.0) {
    std::cerr << "ERROR: --normalize-sigma and --normalize-clip must be "
                 "positive; the normalization is not optional\n";
    return BAD_ARG;
  }
  if (matchOpts.loweRatio <= 0.0 or matchOpts.loweRatio > 1.0) {
    std::cerr << "ERROR: --lowe-ratio must be in (0, 1]\n";
    return BAD_ARG;
  }

  const auto threads = args["threads"].as<int>();
  if (threads < 0) {
    std::cerr << "ERROR: --threads cannot be negative; 0 uses all\n";
    return BAD_ARG;
  }
  const auto residualError = args["residual-error"].as<double>();

  // P3P against no intrinsic is the combination `pgs-calibrate` had to refuse,
  // because the stock binary falls back to DLT without saying so. In process
  // there is nothing to fall back from, so say so and pick DLT.
  const auto solverFor = [&solverName](const bool haveK) {
    auto chosen = loc::ParseSolver(solverName, haveK);
    if (not haveK and chosen != openMVG::resection::SolverType::DLT_6POINTS) {
      std::cout << "WARNING: --resection-method " << solverName
                << " needs a known intrinsic and none is available yet; "
                   "resecting with DLT, which recovers one\n";
      chosen = openMVG::resection::SolverType::DLT_6POINTS;
    }
    return chosen;
  };

  // --- The query ----------------------------------------------------------
  std::cout << "Reading query: " << image << std::endl;
  const auto queryRaw = cv::imread(image.string(), cv::IMREAD_UNCHANGED);
  if (queryRaw.empty()) {
    std::cerr << "ERROR: could not read query image: " << image << "\n";
    return IO_ERROR;
  }
  const int queryW = queryRaw.cols;
  const int queryH = queryRaw.rows;
  std::cout << "  " << queryW << "x" << queryH << ", " << queryRaw.channels()
            << " channel(s)\n";

  cv::Mat mask;
  if (have("mask")) {
    const auto maskPath = pathArg("mask");
    mask = cv::imread(maskPath.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
      std::cerr << "ERROR: could not read mask: " << maskPath << "\n";
      return IO_ERROR;
    }
    if (mask.cols != queryW or mask.rows != queryH) {
      std::cerr << "ERROR: mask is " << mask.cols << "x" << mask.rows
                << " but the query is " << queryW << "x" << queryH << "\n";
      return BAD_ARG;
    }
    std::cout << "  masked by " << maskPath << "\n";
  }

  // --- K and the prior pose -----------------------------------------------
  if (have("camera")) {
    try {
      camera = loc::RescaleCamera(loc::ReadCameraFile(report.camera), queryW,
                                  queryH);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return BAD_ARG;
    }
    // openMVG's pinhole models carry one focal length. Coercing two into one
    // would leave the render and the solve describing different cameras.
    if (std::abs(camera.fx - camera.fy) > 1e-6 * camera.fx) {
      std::cerr << "ERROR: " << report.camera << " has fx = " << camera.fx
                << " and fy = " << camera.fy
                << "; openMVG's pinhole model carries a single focal length, "
                   "so this camera cannot be expressed\n";
      return BAD_ARG;
    }
    report.kSource = "camera-file";
    std::cout << "Camera (" << report.camera.filename().string() << "):\n";
    loc::DescribeCamera(std::cout, camera);
  }

  auto intrinsic =
      have("camera") ? loc::MakeIntrinsic(camera)
                     : std::shared_ptr<openMVG::cameras::IntrinsicBase>();
  // Checked here rather than only at the end: a transposed cx/cy, or a camera
  // file written by an earlier degenerate run, would otherwise drive a full
  // render-and-match solve for minutes before anything complained -- and the
  // complaint would name a DLT fit that never happened.
  if (intrinsic) {
    if (const auto complaint = loc::ComplainAboutIntrinsic(*intrinsic)) {
      std::cerr << "ERROR: " << report.camera
                << " does not describe a physical camera: " << *complaint
                << ". Fix the camera file, or drop --camera to let the "
                   "resection recover an intrinsic.\n";
      return BAD_ARG;
    }
  }

  std::optional<Eigen::Matrix4d> sfmTransform;
  if (have("sfm-transform")) {
    try {
      sfmTransform = pgs::npy::read(pathArg("sfm-transform"));
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return BAD_ARG;
    }
    // It moves the *scene*, and without one there is no scene to move. Said
    // out loud because a change of frame that quietly did not happen is a
    // rigid motion, and no number downstream would show it.
    if (not have("input-scene")) {
      std::cout << "WARNING: --sfm-transform changes the frame the scene is "
                   "loaded in, and there is no --input-scene here. It does "
                   "nothing on the mesh backend, which works in the mesh's own "
                   "frame already.\n";
    }
  }

  // --- Sparse backend ------------------------------------------------------
  std::vector<loc::Correspondence> sparseCorrs;
  loc::SparseReport sparseReport;
  std::optional<openMVG::geometry::Pose3> prior;
  if (camera.havePose) {
    prior = openMVG::geometry::Pose3(camera.R, camera.center());
    report.priorSource = "camera";
  }

  if (have("input-scene")) {
    try {
      std::cout << "Loading scene: " << report.inputScene << std::endl;
      const auto scene = loc::LoadSparseScene(report.inputScene,
                                              report.matchesDir, sfmTransform);
      std::cout << "Describing the query with the scene's describer"
                << std::endl;
      const auto queryGray8 = loc::ToGray8(queryRaw);
      const auto sparseRegions = loc::DescribeWithScene(*scene, queryGray8,
                                                        mask);
      std::cout << "  " << sparseRegions->RegionCount() << " query features\n";
      std::cout << "Matching the query against the structure" << std::endl;
      const auto found = loc::LocalizeAgainstScene(
          *scene, *sparseRegions, intrinsic, solverFor(intrinsic != nullptr),
          queryW, queryH, residualError, sparseReport);
      std::cout << "  " << sparseReport.landmarks << " landmarks over "
                << sparseReport.views << " views; " << sparseReport.putative
                << " putative, " << sparseReport.inliers << " inliers\n";
      if (found.ok) {
        // Resect once here so an unknown K comes back fitted: the render needs
        // one, and the refined pose is a better prior than the raw resection.
        const auto solvedSparse =
            loc::SolvePose(found.correspondences, intrinsic, cameraModel,
                           solverFor(intrinsic != nullptr), queryW, queryH,
                           residualError);
        if (solvedSparse.ok) {
          sparseCorrs = found.correspondences;
          prior = solvedSparse.pose;
          report.priorSource = "sparse";
          if (not intrinsic) {
            intrinsic = solvedSparse.fittedIntrinsic;
            if (const auto complaint =
                    loc::ComplainAboutIntrinsic(*intrinsic)) {
              std::cerr << "ERROR: the sparse resection recovered an intrinsic "
                           "that is not a physical camera: "
                        << *complaint
                        << ".\nDLT estimates the focal from the "
                           "correspondences and goes degenerate when the scene "
                           "is near-fronto-parallel or the matches are few or "
                           "ill-distributed (ADR 0002). Pass --camera so K is "
                           "known and the resection can use P3P.\n";
              // Nothing downstream has run, so the report is thin -- but a
              // caller that asked for one gets a file naming what happened
              // rather than a missing file to interpret.
              report.intrinsicComplaint = *complaint;
              report.kSource = "dlt";
              if (have("report")) {
                try {
                  writeReport(pathArg("report"), report);
                } catch (const std::exception& e) {
                  std::cerr << "ERROR: " << e.what() << "\n";
                }
              }
              return SOLVE_FAILED;
            }
            std::cout << "  DLT recovered an intrinsic: f = "
                      << intrinsic->getParams()[0] << " px\n";
          }
          const loc::Camera sparseCam =
              loc::CameraFromSolve(camera, *intrinsic, solvedSparse.pose);
          std::cout << "Sparse resection:\n";
          loc::DescribeCamera(std::cout, sparseCam);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return IO_ERROR;
    }

    if (report.priorSource != "sparse") {
      // Falling back rather than aborting: render-and-match converges from a
      // generic prior 861 px off, so a failed sparse resection would otherwise
      // make chained mode less robust than --mesh --camera alone.
      if (not have("mesh")) {
        std::cerr << "ERROR: sparse resection failed and there is no --mesh to "
                     "fall back to\n";
        return SOLVE_FAILED;
      }
      if (not prior) {
        std::cerr << "ERROR: sparse resection failed and --camera carries no "
                     "pose to fall back on\n";
        return SOLVE_FAILED;
      }
      report.priorSource = "sparse-failed-fell-back";
      std::cout << "WARNING: sparse resection failed; falling back to the "
                   "--camera prior\n";
    }
  }

  if (have("mesh") and not prior) {
    std::cerr << "ERROR: --mesh has no prior pose to render from\n";
    return BAD_ARG;
  }
  if (not intrinsic) {
    std::cerr << "ERROR: no intrinsic is known and none was recovered\n";
    return BAD_ARG;
  }

  // --- Render and match ----------------------------------------------------
  std::vector<loc::Correspondence> meshCorrs;
  loc::MeshBundle bundle;
  cv::Mat queryNorm;
  if (have("mesh")) {
    const auto priorCam = loc::CameraFromSolve(camera, *intrinsic, *prior);
    std::cout << "Prior pose (" << report.priorSource << "):\n";
    loc::DescribeCamera(std::cout, priorCam);
    try {
      std::cout << "Loading mesh: " << report.mesh << std::endl;
      bundle = loc::ReadMeshBundle(report.mesh);
      std::cout << "  " << bundle.mesh.num_vertices() << " vertices, "
                << bundle.mesh.num_faces() << " faces, "
                << bundle.textures.size() << " chart(s)\n";
      if (not bundle.has_texture()) {
        std::cerr << "ERROR: " << report.mesh
                  << " carries no usable texture; there is nothing to match "
                     "against\n";
        return IO_ERROR;
      }

      std::cout << "Normalizing and describing the query" << std::endl;
      queryNorm = loc::LocalNormalize(loc::ToGray01(queryRaw),
                                      matchOpts.normalizeSigma,
                                      matchOpts.normalizeClip);
      const auto queryRegions = loc::DescribeImage(queryNorm, mask,
                                                   matchOpts.preset);
      std::cout << "  " << queryRegions->RegionCount() << " query features\n";

      std::cout << "Rendering from the prior pose" << std::endl;
      const auto render = loc::RenderMesh(bundle, priorCam,
                                          static_cast<std::size_t>(threads));
      std::cout << "  " << render.hits << " of "
                << static_cast<std::size_t>(queryW) * queryH
                << " pixels hit the surface\n";
      if (not render.missingCharts.empty()) {
        std::cout << "WARNING: no usable texture image for UV chart(s)";
        for (const auto c : render.missingCharts) {
          std::cout << " " << c;
        }
        std::cout << "; those regions rendered black\n";
      }

      std::cout << "Matching render against query" << std::endl;
      meshCorrs = loc::CorrespondFromRender(render, queryNorm, *queryRegions,
                                            matchOpts, report.matches);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return IO_ERROR;
    }

    std::cout << "  features: " << report.matches.detectedRender << " render, "
              << report.matches.detectedQuery << " query\n";
    std::cout << "  Lowe ratio kept " << report.matches.lowePassed
              << "; similarity RANSAC kept " << report.matches.similarityInliers
              << " (scale " << report.matches.similarityScale << ", rotation "
              << report.matches.similarityRotationDeg << " deg)\n";
    std::cout << "  Lucas-Kanade kept " << report.matches.lkKept << "; lifted "
              << report.matches.lifted << " to 3D\n";
  }

  // --- Solve ---------------------------------------------------------------
  // Both correspondence sets go into one solve rather than a resect-then-refine
  // sequence. That is only possible because --sfm-transform was applied to the
  // scene at load: the sparse structure and the render-lifted points are then
  // already in one coordinate system.
  auto corrs = meshCorrs;
  const auto fuseSparse = args["fuse-sparse"].as<bool>();
  if (fuseSparse or meshCorrs.empty()) {
    corrs.insert(corrs.end(), sparseCorrs.begin(), sparseCorrs.end());
  } else {
    // Not discarded silently: the sparse resection still supplied the prior,
    // which is most of what it is good for.
    std::cout << "  --fuse-sparse 0: the " << sparseCorrs.size()
              << " sparse correspondences seeded the prior and stop there\n";
    sparseCorrs.clear();
  }
  report.sparseCorrespondences = sparseCorrs.size();
  report.meshCorrespondences = meshCorrs.size();
  if (corrs.size() < 6) {
    std::cerr << "ERROR: only " << corrs.size()
              << " correspondences survived; nothing to resect from\n";
    return NO_CORRESPONDENCES;
  }

  std::cout << "Resecting on " << corrs.size() << " correspondences ("
            << meshCorrs.size() << " mesh, " << sparseCorrs.size()
            << " sparse)" << std::endl;
  auto solved = loc::SolvePose(corrs, intrinsic, cameraModel,
                               solverFor(intrinsic != nullptr), queryW, queryH,
                               residualError);
  if (not solved.ok) {
    std::cerr << "ERROR: resection failed on " << corrs.size()
              << " correspondences\n";
    return SOLVE_FAILED;
  }
  std::cout << "  " << solved.inliers.size() << " inliers, inlier RMS "
            << solved.inlierRmsPx << " px, spread " << solved.spreadX << " x "
            << solved.spreadY << " px\n";

  // --- Second pass, if the first came out thin -----------------------------
  // Not an improvement and not meant as one: re-rendering from the refined pose
  // and solving again was measured at 1.820 -> 1.821 -> 1.838 px over twelve
  // datasets. It is a guard on a first pass that found too little to trust,
  // which is why --second-pass-min-inliers defaults to 0 and this does not run.
  const auto secondPassMin = args["second-pass-min-inliers"].as<int>();
  if (secondPassMin < 0) {
    std::cerr << "ERROR: --second-pass-min-inliers cannot be negative; 0 is "
                 "off\n";
    return BAD_ARG;
  }
  if (secondPassMin > 0 and not have("mesh")) {
    std::cout << "WARNING: --second-pass-min-inliers has nothing to re-render "
                 "without --mesh; ignoring it\n";
  } else if (secondPassMin > 0 and
             solved.inliers.size() < static_cast<std::size_t>(secondPassMin)) {
    std::cout << "Only " << solved.inliers.size() << " inliers, under the "
              << secondPassMin << " --second-pass-min-inliers asks for; "
                 "re-rendering from the refined pose" << std::endl;
    const auto& firstIntrinsic =
        solved.fittedIntrinsic ? *solved.fittedIntrinsic : *intrinsic;
    const auto refinedCam =
        loc::CameraFromSolve(camera, firstIntrinsic, solved.pose);
    try {
      const auto render = loc::RenderMesh(bundle, refinedCam,
                                          static_cast<std::size_t>(threads));
      const auto queryRegions = loc::DescribeImage(queryNorm, mask,
                                                   matchOpts.preset);
      loc::MatchReport second;
      auto again = loc::CorrespondFromRender(render, queryNorm, *queryRegions,
                                             matchOpts, second);
      if (fuseSparse) {
        again.insert(again.end(), sparseCorrs.begin(), sparseCorrs.end());
      }
      std::cout << "  second pass lifted " << second.lifted
                << " (first pass: " << report.matches.lifted << ")\n";
      if (again.size() >= 6) {
        const auto resolved =
            loc::SolvePose(again, intrinsic, cameraModel,
                           solverFor(intrinsic != nullptr), queryW, queryH,
                           residualError);
        // Kept only when it is actually better constrained. A second pass that
        // finds less is a second pass that should not have run.
        if (resolved.ok and resolved.inliers.size() > solved.inliers.size()) {
          solved = resolved;
          corrs = std::move(again);
          report.matches = second;
          report.passes = 2;
          std::cout << "  kept: " << solved.inliers.size()
                    << " inliers, inlier RMS " << solved.inlierRmsPx << " px\n";
        } else {
          std::cout << "  discarded: the second pass is no better constrained "
                       "than the first\n";
        }
      }
    } catch (const std::exception& e) {
      std::cout << "WARNING: the second pass failed (" << e.what()
                << "); keeping the first\n";
    }
  }

  const auto& solvedIntrinsic =
      solved.fittedIntrinsic ? *solved.fittedIntrinsic : *intrinsic;
  // The last guard before anything is written. Unconditional on purpose: a
  // supplied K was checked at load and a DLT-fitted one where it was fitted, so
  // reaching here with a bad intrinsic means something unaccounted for moved
  // it -- which is exactly when a guard earns its place. Held, not acted on
  // yet: the calibration is what must not be emitted, while the report and the
  // QA render are the only things that say *why*, so they are still written
  // below before this exits.
  const auto intrinsicComplaint = loc::ComplainAboutIntrinsic(solvedIntrinsic);

  const auto outCamera =
      loc::CameraFromSolve(camera, solvedIntrinsic, solved.pose);
  std::cout << "Solved camera:\n";
  loc::DescribeCamera(std::cout, outCamera);

  report.intrinsicComplaint = intrinsicComplaint.value_or("");
  report.inliers = solved.inliers.size();
  report.spreadX = solved.spreadX;
  report.spreadY = solved.spreadY;

  // --- Held-out metrics ----------------------------------------------------
  // A spatial checkerboard: solve on one colour, score on the other. The pose
  // emitted above is the one solved on everything; this is one extra resection.
  const auto splitPx = args["heldout-split"].as<int>();
  std::cout << "Scoring on a " << splitPx << " px held-out checkerboard"
            << std::endl;
  report.heldout = loc::HeldoutScore(corrs, intrinsic, cameraModel,
                                     solverFor(intrinsic != nullptr), queryW,
                                     queryH, residualError, splitPx);
  if (report.heldout.ok) {
    const auto& s = report.heldout.stats;
    std::cout << "  fit " << report.heldout.fitCount << " / test "
              << report.heldout.testCount << ": rms " << s.rmsPx << " px, p99 "
              << s.p99Px << " px, |bias| " << s.biasPx << " px (" << s.biasXPx
              << ", " << s.biasYPx << ")\n";
  } else {
    std::cout << "  not measured: the split left fit "
              << report.heldout.fitCount << " / test "
              << report.heldout.testCount
              << ", too few to score with. Lower --heldout-split.\n";
  }

  report.center = outCamera.center();
  report.standoff = outCamera.t[2];
  // The camera file supplies the rig's own standoff, and a flag that is given
  // explicitly overrides it -- including `--expected-standoff 0`, which is how
  // a caller disarms a gate the file arms. `defaulted()` is what separates "not
  // passed" from "passed zero"; `count()` cannot, since a defaulted value
  // counts as present.
  report.expectedStandoff = args["expected-standoff"].defaulted()
                                ? camera.expectedStandoff
                                : args["expected-standoff"].as<double>();
  report.standoffTolerance = args["standoff-tolerance"].defaulted()
                                 ? camera.standoffTolerance
                                 : args["standoff-tolerance"].as<double>();
  if (report.expectedStandoff > 0.0) {
    if (report.standoffTolerance <= 0.0) {
      std::cerr << "ERROR: an expected standoff needs a positive tolerance; a "
                   "gate with no width is not a gate. Set it with "
                   "--standoff-tolerance or a standoff_tolerance key in the "
                   "camera file\n";
      return BAD_ARG;
    }
    report.standoffGated = true;
  } else {
    // Said out loud because `unset` is not a pass: it means the one gate
    // independent of the correspondence set judged nothing, and a wrong focal
    // absorbed into standoff is invisible to every other row.
    std::cout << "WARNING: no expected standoff, so the only gate that does not "
                 "depend on the correspondence set is unarmed. Put "
                 "expected_standoff and standoff_tolerance in the camera file, "
                 "or pass --expected-standoff.\n";
  }
  grade(report);
  std::cout << "  standoff (the scene origin's depth in camera space): "
            << report.standoff << "\n";

  std::cout << "Gates: rms " << to_string(report.gateRms) << ", p99 "
            << to_string(report.gateP99) << ", bias "
            << to_string(report.gateBias) << ", inliers "
            << to_string(report.gateInliers) << ", standoff "
            << to_string(report.gateStandoff) << "\n";
  if (const auto w = worst(report); w == Gate::Fail or w == Gate::Review) {
    std::cout << "WARNING: this solve grades " << to_string(w)
              << ". Look at --qa-render before using it: the failure modes here "
                 "are silent, and a frame error posts a clean residual.\n";
  }

  // --- Emit ----------------------------------------------------------------
  try {
    if (have("output-calibration") and not intrinsicComplaint) {
      const auto out = pathArg("output-calibration");
      loc::WriteCalibration(out, solvedIntrinsic, solved.pose, image);
      std::cout << "Wrote calibration: " << out << "\n";
    }
    if (have("output-camera") and not intrinsicComplaint) {
      const auto out = pathArg("output-camera");
      // The flat format carries radial k1..k3 and nothing else. A Brown model's
      // tangential terms and a fisheye's four-coefficient series have no
      // spelling in it, and `ReadCameraFile` refuses p1/p2 rather than pretend
      // otherwise -- so say which record is the faithful one instead of
      // emitting a file that quietly describes a different camera.
      const auto type = solvedIntrinsic.getType();
      if (type == openMVG::cameras::PINHOLE_CAMERA_BROWN or
          type == openMVG::cameras::PINHOLE_CAMERA_FISHEYE) {
        std::cout << "WARNING: the camera file format carries radial k1..k3 "
                     "only, and this camera was fitted with a model whose "
                     "distortion it cannot express. The camera file's pose and "
                     "K are right; for the distortion, use the calibration "
                     "JSON.\n";
      }
      loc::WriteCameraFile(out, outCamera);
      std::cout << "Wrote camera file: " << out << "\n";
    }
    if (have("report")) {
      const auto out = pathArg("report");
      writeReport(out, report);
      std::cout << "Wrote report: " << out << "\n";
    }
    if (have("qa-render")) {
      // Re-rendered at the SOLVED pose and differenced against the query. This
      // exists because the 180-degree frame error in the source work posted a
      // bit-identical held-out rms: no residual, inlier count or cost can
      // reveal that class of error, and looking is the only thing that can.
      if (not have("mesh")) {
        std::cout << "WARNING: --qa-render needs --mesh to render from; "
                     "skipping\n";
      } else {
        const auto out = pathArg("qa-render");
        std::cout << "Rendering at the solved pose for QA" << std::endl;
        const auto qa = loc::RenderMesh(bundle, outCamera,
                                        static_cast<std::size_t>(threads));
        const auto qaNorm =
            loc::LocalNormalize(loc::ToGray01(qa.gray), matchOpts.normalizeSigma,
                                matchOpts.normalizeClip);
        if (not cv::imwrite(out.string(), qaNorm)) {
          throw std::runtime_error("could not write " + out.string());
        }
        std::cout << "Wrote QA render: " << out << "\n";
        // Differenced after normalization, not before: the two images are the
        // same band but not the same processing, so a raw difference is a
        // picture of the tone curves rather than of the registration.
        cv::Mat diff;
        cv::absdiff(qaNorm, queryNorm, diff);
        auto diffPath = out;
        diffPath.replace_filename(out.stem().string() + "_diff" +
                                  out.extension().string());
        if (not cv::imwrite(diffPath.string(), diff)) {
          throw std::runtime_error("could not write " + diffPath.string());
        }
        std::cout << "Wrote QA difference: " << diffPath << "\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }

  if (intrinsicComplaint) {
    std::cerr << "ERROR: the solve produced an intrinsic that is not a physical "
                 "camera: "
              << *intrinsicComplaint
              << ". A calibration like this loads fine everywhere downstream "
                 "and misaligns the texture, so no calibration or camera file "
                 "was written. The report and any QA render were, because they "
                 "are what say why.\n";
    return SOLVE_FAILED;
  }

  return SUCCESS;
}
