#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <educelab/core/utils/Filesystem.hpp>
#include <educelab/core/utils/String.hpp>

#include <indicators/progress_bar.hpp>

#include <openMVG/geometry/Similarity3.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_transform.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include <Eigen/Core>

#include "global_scaler_io.hpp"
#include "marker_detection.hpp"
#include "marker_types.hpp"

namespace ar = cv::aruco;
namespace el = educelab;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace openMVG;
using namespace indicators;
using namespace pgs;

namespace {

void ScaleLandmarks(pgs::Landmarks &ldms, const double scale) {
  for (auto &[_, ldm] : ldms) {
    if (ldm.X) {
      ldm.X.value() *= scale;
    }
  }
}

enum EXIT_CODE {
  SUCCESS = 0,
  HELP = 1,
  BAD_ARG = 2,
  NO_VIEWS = 3,
  NO_LDMS = 4,
  NO_SCALES = 5
};
} // namespace

auto LoadFilterFile(const fs::path &path) {
  // Open the file
  std::ifstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + path.string());
  }

  // Get all lines, excluding duplicates
  std::string line;
  std::unordered_set<std::string> lines;
  while (std::getline(file, line)) {
    lines.insert(line);
  }

  return lines;
}

auto main(int argc, char* argv[]) -> int
{
  // clang-format off
  po::options_description parser("options");
  parser.add_options()
    ("help,h", "print help message")
    ("input-scene,i", po::value<std::string>()->required(), "input sfm scene file")
    ("output-scene,o", po::value<std::string>(), "output sfm scene file")
    ("scale-method", po::value<std::string>()->default_value("umeyama"), R"(scale method: "umeyama" (weighted-median) or "edge" (median of edge lengths))")
    ("input-mesh",  po::value<std::string>(), "input mesh file (ascii .ply only)")
    ("output-mesh", po::value<std::string>(), "output (scaled) mesh file (.ply)")
    ("histogram-out",         po::value<std::string>(), "where to write the scale‐histogram SVG")
    ("marker-size,s", po::value<double>(), "ArUco marker size in desired world units (required unless --detection-method sample-square, which has a known fixed size)")
    ("detection-method,m", po::value<std::string>()->default_value("markers"), "detection method: markers, sample-square")
    ("sfm-root", po::value<std::string>(), "use the given directory as the sfm root when loading image files")
    ("include-from", po::value<std::string>(), "only consider image files listed by name in the provided txt file")
    ("exclude-from", po::value<std::string>(), "do not consider any of the image files listed by name in the provided txt file")
    ("undistort-images", po::bool_switch(), "undistort images before running marker detection")
    ("min-marker-pix", po::value<int>()->default_value(32), "minimum marker size in pixels")
    ("detect-inverted", po::bool_switch(), "attempt to detect inverted markers")
    ("no-ransac", po::bool_switch(), "use RANSAC to make marker triangulation more resilient to false positive matches")
    ("save-debug-images", po::value<std::string>(), "save debug images to the given directory")
    ("save-landmarks", po::value<std::string>(), "save unscaled, triangulated landmarks to the given mesh file (obj, ply)")
    ("save-scaled-landmarks", po::value<std::string>(), "save scaled, triangulated landmarks to the given mesh file (obj, ply)")
    ("progress,p", po::bool_switch(), "Show progress bar")
  ;
  // clang-format on

  po::variables_map args;
  po::store(po::parse_command_line(argc, argv, parser), args);
  if (argc == 1 or args.count("help") > 0) {
    std::cout << parser << "\n";
    return HELP;
  }
  po::notify(args);

  auto scaleMethod = el::to_lower_copy(args["scale-method"].as<std::string>());
  if (scaleMethod != "umeyama" && scaleMethod != "edge") {
    std::cerr << "ERROR: --scale-method must be \"umeyama\" or \"edge\"\n";
    return BAD_ARG;
  }
  // Get the input and output files
  fs::path sfmPath = args["input-scene"].as<std::string>();

  // Detection method
  auto method = el::to_lower_copy(args["detection-method"].as<std::string>());
  std::function detect = DetectMarkers;
  if (method == "markers") {
    detect = DetectMarkers;
  } else if (method == "sample-square") {
    detect = DetectSampleSquare;
  } else {
    std::cout << "ERROR: Unrecognized detection method: \'" << method << "\'\n";
    return BAD_ARG;
  }

  // Marker size. The PGS sample square is a fixed-size physical target, so its
  // edge length (0.47 cm) is known a priori and need not be supplied.
  static constexpr double kSampleSquareSize = 0.47;
  const bool sampleSquare = method == "sample-square";
  if (!sampleSquare && args.count("marker-size") == 0) {
    std::cerr << "ERROR: --marker-size is required unless --detection-method "
                 "sample-square\n";
    return BAD_ARG;
  }
  auto markerSize = args.count("marker-size")
                        ? args["marker-size"].as<double>()
                        : kSampleSquareSize;
  if (markerSize <= 0.0) {
    std::cerr << "ERROR: --marker-size must be a positive value\n";
    return BAD_ARG;
  }

  // Write a histogram if requested
  bool doHistogram = args.count("histogram-out") > 0;
  ar::DetectorParameters params;
  params.useAruco3Detection = true;
  params.detectInvertedMarker = args["detect-inverted"].as<bool>();
  params.cornerRefinementMethod = ar::CORNER_REFINE_SUBPIX;
  auto minMarkerSize = static_cast<float>(args["min-marker-pix"].as<int>());

  // Boolean options
  auto undistortImages = args["undistort-images"].as<bool>();
  auto saveDebugImages = args.count("save-debug-images") > 0;

  // Load SfM file
  sfm::SfM_Data sfmData;
  sfm::Load(sfmData, sfmPath.string(), sfm::ALL);
  std::cout << "Loaded SfM scene: ";
  std::cout << sfmData.GetViews().size() << " views, ";
  std::cout << sfmData.GetPoses().size() << " poses, ";
  std::cout << sfmData.GetIntrinsics().size() << " intrinsics\n";

  fs::path sfmRoot = sfmData.s_root_path;
  if (args.count("sfm-root") > 0) {
    sfmRoot = args["sfm-root"].as<std::string>();
    std::cout << "Using custom SfM root: " << sfmRoot.string() << "\n";
  }

  // Set up debug directory
  fs::path debugDir;
  if (saveDebugImages) {
    debugDir = args["save-debug-images"].as<std::string>();
    fs::create_directories(debugDir);
  }

  // View filters
  using FilterFunction =
      std::function<bool(const std::shared_ptr<sfm::View> &)>;
  std::vector<FilterFunction> filters{
      [&sfmData](const std::shared_ptr<sfm::View> &view) {
        return sfmData.IsPoseAndIntrinsicDefined(view.get());
      }};
  if (args.count("include-from") > 0) {
    auto includes = LoadFilterFile(args["include-from"].as<std::string>());
    filters.emplace_back([includes](const std::shared_ptr<sfm::View> &view) {
      return includes.count(view->s_Img_path) > 0;
    });
  }
  if (args.count("exclude-from") > 0) {
    auto excludes = LoadFilterFile(args["exclude-from"].as<std::string>());
    filters.emplace_back([excludes](const std::shared_ptr<sfm::View> &view) {
      return excludes.count(view->s_Img_path) == 0;
    });
  }
  std::function filter =
      [&filters](const std::shared_ptr<sfm::View> &view) -> bool {
    return std::all_of(filters.begin(), filters.end(),
                       [&view](const auto &f) { return f(view); });
  };
  sfm::Views views;
  std::copy_if(sfmData.GetViews().begin(), sfmData.GetViews().end(),
               std::inserter(views, views.end()),
               [&filter](const auto &pair) { return filter(pair.second); });
  if (views.empty()) {
    std::cout << "ERROR: No views selected!\n";
    return NO_VIEWS;
  }

  // All observed markers
  std::set<int> markerIDs;
  // All observed marker corners
  pgs::Landmarks landmarks;

  // Detect landmarks
  std::string imgType = undistortImages ? "corrected" : "original";
  std::cout << "Detecting landmarks in " + imgType + " images (using "
            << views.size() << " views)\n";
  std::unique_ptr<ProgressBar> bar;
  std::size_t iter{0};
  auto numIters = views.size();
  auto pad = std::to_string(numIters).size();
  std::size_t viewsWithLandmarks{0};
  if (args["progress"].as<bool>()) {
    bar = std::make_unique<ProgressBar>(
        option::BarWidth{50}, option::Start{" ["},
        option::ForegroundColor{Color::unspecified},
        option::MaxProgress{numIters});
  }
  for (const auto &[viewID, view] : views) {
    // Load the image (views were already filtered above)
    auto path = view->s_Img_path;
    fs::path fullPath = sfmRoot / path;
    auto image = cv::imread(fullPath);
    if (image.empty()) {
      std::cout << "WARNING: Could not load image, skipping: "
                << fullPath.string() << "\n";
      if (bar) {
        bar->tick();
      }
      continue;
    }

    // Undistort the images
    if (undistortImages) {
      auto cam = sfmData.intrinsics.at(view->id_intrinsic);
      image = UndistortImage(image, cam.get());
    }

    // Detect markers
    params.minMarkerLengthRatioOriginalImg =
        minMarkerSize / static_cast<float>(std::max(image.rows, image.cols));
    auto res = detect(image, params);

    if (not res.markerIds.empty()) {
      viewsWithLandmarks += 1;
    }

    // Draw markers
    if (saveDebugImages and not res.markerIds.empty()) {
      ar::drawDetectedMarkers(image, res.markerCorners, res.markerIds);
      if (not res.charucoIDs.empty()) {
        ar::drawDetectedCornersCharuco(image, res.charucoCorners, res.charucoIDs);
      }
      // Write marker image
      auto outFile = debugDir / fs::path(path).replace_extension("jpg");
      cv::imwrite(outFile, image);
    }

    // For each aruco ID found
    for (std::size_t idx = 0; idx < res.markerIds.size(); ++idx) {
      // Keep a list of all discovered markers
      const auto markerID = res.markerIds[idx];
      markerIDs.insert(markerID);

      // Track the marker corner observations
      const auto &corners = res.markerCorners[idx];
      for (std::size_t i = 0; i < corners.size(); i++) {
        // Get the stored landmark
        pgs::Landmark *ldm;
        const auto cornerID = GetLandmarkID(markerID, static_cast<int>(i));
        if (landmarks.count(cornerID) == 0) {
          landmarks.insert({cornerID, {}});
        }
        ldm = &landmarks.at(cornerID);

        auto corner = corners[i];
        ldm->obs.emplace_back(viewID, Vec2{corner.x, corner.y});
      }
    }
    ++iter;
    if (bar) {
      bar->set_option(option::PostfixText{el::to_padded_string(iter, pad, ' ') +
                                          "/" + std::to_string(numIters)});
      bar->tick();
    }
  }
  std::cout << "Detected landmarks in " << viewsWithLandmarks << " of "
            << views.size() << " views\n";

  // Triangulate the control points
  std::size_t numTriangulated{0};
  bool useRansac = not args["no-ransac"].as<bool>();
  auto postTxt = useRansac ? " w/RANSAC\n" : "\n";
  std::cout << "Triangulating landmarks" << postTxt;
  for (auto &[ldmID, ldm] : landmarks) {
    const auto &obs = ldm.obs;
    if (obs.size() < 3) {
      std::cout << "WARNING: Not enough observations to triangulate landmark ";
      std::cout << ldmID << "\n";
      continue;
    }

    // Collect observations and poses
    std::vector<RansacObservation> x;
    x.reserve(obs.size());
    for (const auto &[viewID, o] : obs) {
      auto view = sfmData.views.at(viewID);
      auto cam = sfmData.intrinsics.at(view->id_intrinsic);
      Mat3X pt;
      if (undistortImages) {
        pt = (*cam)(o);
      } else {
        pt = (*cam)(cam->get_ud_pixel(o));
      }

      auto pose = sfmData.GetPoseOrDie(view.get());
      x.emplace_back(o, pt, cam, pose);
    }

    // Triangulate
    bool success{false};
    Vec3 X;
    if (useRansac) {
      std::tie(success, X) = TriangulateRansac(x);
    } else {
      std::tie(success, X) = Triangulate(x);
    }

    if (not success) {
      std::cout << "WARNING: Could not triangulate landmark: " << ldmID << "\n";
      continue;
    }
    ldm.X = X;
    numTriangulated += 1;
  }

  // Need at least 2 triangulated points to even try measuring landmarks
  std::cout << "Triangulated " << numTriangulated << " of " << landmarks.size()
            << " landmarks\n";
  if (numTriangulated < 2) {
    std::cout << "ERROR: Not enough landmarks to estimate scale!\n";
    return NO_LDMS;
  }

  // Decide which stats to compute
  ScaleStats stats;
  if (scaleMethod == "edge") {
    stats = ComputeEdgeScaleStats(landmarks, markerIDs, markerSize);
    std::cout << "Edge-length median scale: " << stats.summary << "\n";
  } else {
    stats = ComputeUmeyamaScaleStats(landmarks, markerIDs, markerSize);
    std::cout << "Umeyama median scale:     " << stats.summary << "\n";
  }

  // Bail out if no marker yielded a usable scale estimate, rather than
  // silently applying the default scale of 1.0
  if (stats.scales.empty()) {
    std::cout << "ERROR: Could not estimate scale from the detected markers!\n";
    return NO_SCALES;
  }

  // Save a histogram file
  if (doHistogram) {
    fs::path histPath = args["histogram-out"].as<std::string>();
    std::cout << "Saving histogram file: " << histPath << "\n";
    pgs::WriteScaleHistogram(histPath, stats.scales, stats.summary,
                             stats.summaryLabel);
  }

  // Scale and save the scene
  if (args.count("output-scene") > 0) {
    fs::path outPath = args["output-scene"].as<std::string>();
    std::cout << "Saving scaled SfM data\n";
    sfm::ApplySimilarity({{}, stats.summary}, sfmData);
    sfm::Save(sfmData, outPath.string(), sfm::ALL);
  }

  // Scale and save a mesh
  if (args.count("input-mesh") && args.count("output-mesh")) {
    fs::path inPath = args["input-mesh"].as<std::string>();
    fs::path outPath = args["output-mesh"].as<std::string>();
    std::cout << "Saving scaled mesh: " << outPath << "\n";
    pgs::ScalePLYMesh(inPath, outPath, stats.summary);
  }

  // Write the landmarks mesh
  if (args.count("save-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving unscaled landmark mesh\n";
    fs::path ldmMesh = args["save-landmarks"].as<std::string>();
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    pgs::WriteMesh(ldmMesh, landmarks);
  }

  // Write the scaled landmarks mesh
  if (args.count("save-scaled-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving scaled landmark mesh\n";
    fs::path ldmMesh = args["save-scaled-landmarks"].as<std::string>();
    ScaleLandmarks(landmarks, stats.summary);
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    pgs::WriteMesh(ldmMesh, landmarks);
  }
  std::cout << "Done.\n";
}
