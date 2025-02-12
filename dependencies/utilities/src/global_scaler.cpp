#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
#include <educelab/core/utils/Filesystem.hpp>
#include <educelab/core/utils/String.hpp>
#include <indicators/progress_bar.hpp>
#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/geometry/Similarity3.hpp>
#include <openMVG/multiview/triangulation_nview.hpp>
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/sfm_data_io.hpp>
#include <openMVG/sfm/sfm_data_transform.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include "RANSAC.hpp"

namespace ar = cv::aruco;
namespace el = educelab;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace openMVG;
using namespace indicators;
using namespace ransac;

namespace pgs {
/** Index number of image in SfM */
using ViewID = openMVG::IndexT;
/** List of a landmark's observations and the view where it was found */
using Observations = std::vector<std::pair<ViewID, Vec2>>;

/** Landmark: A collection of 2D observations and the triangulated 3D point */
struct Landmark {
  Landmark() = default;
  explicit Landmark(std::string id) : id{std::move(id)} {}
  std::string id;
  Observations obs;
  std::optional<Vec3> X;
};

/** Collection of landmarks indexed by ID */
using Landmarks = std::map<std::string, Landmark>;
} // namespace pgs

namespace {
// Known distance maps for the EduceLab Sample Square
/*
using IDPair = std::pair<int, int>;
using DistanceMap = std::map<IDPair, double>;

auto init_distance_map() -> DistanceMap {
    // relative marker positions in cm
    std::array<cv::Vec2d, 4> markerPos = {{
        {0.866666666666667, 0.2},
        {0.2, 0.866666666666667},
        {1.533333333333334, 0.866666666666667},
        {0.866666666666667, 1.533333333333334}
    }};

    DistanceMap res;
    for (std::size_t i = 0; i < 4; i++) {
        for (std::size_t j = 0; j < 4; j++) {
            if (i == j or res.contains({i, j})) {
                continue;
            }
            const auto d = cv::norm(markerPos[j] - markerPos[i]);
            res.insert({{i, j}, d});
            res.insert({{j, i}, d});
            res.insert({{i + 512, j + 512}, d});
            res.insert({{j + 512, i + 512}, d});
        }
    }
    return res;
}

auto DistanceMapCM = init_distance_map();
*/

/** List of ArUco IDs */
using IDList = std::vector<int>;
/** List of a single ArUco marker's corner locations */
using CornersList = std::vector<cv::Point2f>;
/** List of ArUco markers */
using CornersArray = std::vector<CornersList>;
/** Result from running an ArUco detection method */
struct DetectionResult {
    IDList charucoIDs;
    CornersList charucoCorners;
    IDList markerIds;
    CornersArray markerCorners;
    CornersArray rejected;
};

struct RansacObservation {
  using Cam = std::shared_ptr<cameras::IntrinsicBase>;
  RansacObservation() = default;
  RansacObservation(const Vec2 &obs, const Vec3 &pt, const Cam &cam,
                    const geometry::Pose3 &pose)
      : obs{obs}, pt{pt}, cam{cam}, pose{pose} {}
  Vec2 obs;
  Vec3 pt;
  Cam cam;
  geometry::Pose3 pose;
};

auto Triangulate(const std::vector<RansacObservation> &x)
    -> std::pair<bool, Vec3> {
  // Unzip
  std::vector<Vec3> pts;
  std::vector<Mat34> poses;
  pts.reserve(x.size());
  poses.reserve(x.size());
  for (const auto &ro : x) {
    pts.push_back(ro.pt);
    poses.push_back(ro.pose.asMatrix());
  }

  // Fit
  const Map<const Mat3X> mtx(pts[0].data(), 3, pts.size());
  Vec4 Xh;
  if (not TriangulateNViewAlgebraic(mtx, poses, &Xh)) {
    return {false, {}};
  }
  Vec3 X = Xh.hnormalized();

  if (X.hasNaN()) {
    return {false, {}};
  }

  // Test validity (in front of the cameras)
  for (const auto &ro : x) {
    auto chirality = ro.pt.dot(ro.pose(X)) > 0.0;
    if (not chirality) {
      return {false, {}};
    }
  }

  return {true, X};
}

auto EvalTriangulate(const std::vector<RansacObservation> &x, const Vec3 &X)
    -> RANSACResult<RansacObservation, double> {
  using Result = RANSACResult<RansacObservation, double>;
  Result result;
  result.error = 0.;
  constexpr double threshold = 0.1;
  for (const auto &ro : x) {
    // If any views fail chirality, it's a bad model
    const auto chirality = ro.pt.dot(ro.pose(X)) > 0.0;
    if (not chirality) {
      return Result{};
    }
    // Accumulate the residual error
    const auto err = ro.cam->residual(ro.pose(X), ro.obs).norm();
    if (err < threshold) {
      result.error += err;
      result.inliers.push_back(ro);
    }
  }

  // calculate fitness and rmse
  if (result.inliers.size() > 0) {
    result.fitness = static_cast<double>(result.inliers.size()) /
                     static_cast<double>(x.size());
    result.inlier_rmse =
        result.error / std::sqrt(static_cast<double>(result.inliers.size()));
  }
  result.success = true;
  return result;
}

auto TriangulateRansac(const std::vector<RansacObservation> &x)
    -> std::pair<bool, Vec3> {
  constexpr std::size_t nIters = 1000;
  constexpr std::size_t nSamples = 2;
  // fixed seed for reproducibility
  constexpr std::uint_fast32_t seed = 0;
  const auto [X, res] =
      RANSAC(x, Triangulate, EvalTriangulate, nSamples, nIters, seed);
  return {res.success, X};
}

/** Detect ArUco markers */
auto DetectMarkers(const cv::Mat &image, const ar::DetectorParameters &params)
    -> DetectionResult {
  const auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
  const ar::ArucoDetector detector(dict, params);

  DetectionResult res;
  detector.detectMarkers(image, res.markerCorners, res.markerIds, res.rejected);

  return res;
}

/** (EduceLab Sample Square only) Generate a ChArUco board */
auto GenerateBoard(int offset = 0) {
    auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
    dict.bytesList = dict.bytesList({offset, offset + 4}, cv::Range::all());
    auto board = ar::CharucoBoard({3, 3}, 10, 7, dict);
    return board;
}

/** Detect a ChArUco board */
auto DetectBoard(const cv::Mat &image, const ar::CharucoBoard &board,
                 const ar::DetectorParameters &params) -> DetectionResult {
  // Adjust detector scale relative to largest dimension

  ar::CharucoParameters charucoParams;
  charucoParams.tryRefineMarkers = true;

  // Detect the Aruco markers
  const ar::CharucoDetector detector(board, charucoParams, params);
  DetectionResult res;
  detector.detectBoard(image, res.charucoCorners, res.charucoIDs,
                       res.markerCorners, res.markerIds);

  return res;
}

/** Detect the EduceLab Sample Square */
auto DetectSampleSquare(const cv::Mat &image,
                        const ar::DetectorParameters &params)
    -> DetectionResult {
  static const auto boardTop = GenerateBoard();
  auto res = DetectBoard(image, boardTop, params);
  if (res.charucoIDs.empty()) {
    res = DetectionResult();
  }

  static const auto boardBot = GenerateBoard(512);
  const auto res2 = DetectBoard(image, boardBot, params);
  if (res2.charucoIDs.empty()) {
    return res;
  }

  // Merge landmarks and IDs
  for (std::size_t idx = 0; idx < res2.markerIds.size(); ++idx) {
    res.markerIds.push_back(res2.markerIds[idx] + 512);
    res.markerCorners.push_back(res2.markerCorners[idx]);
  }
  for (std::size_t idx = 0; idx < res2.charucoIDs.size(); ++idx) {
    res.charucoIDs.push_back(res2.charucoIDs[idx] + 512);
    res.charucoCorners.push_back(res2.charucoCorners[idx]);
  }

  return res;
}

/**
 * Helper function to build the ID for a specific ArUco marker corner.
 *
 * CornerID:
 *  - 0: TL
 *  - 1: TR
 *  - 2: BR
 *  - 3: BL
 */
auto GetLandmarkID(const int arucoID, const int cornerID) -> std::string {
  return std::to_string(arucoID) + "." + std::to_string(cornerID);
}

/** Undistort an image using cv::undistort */
auto UndistortImage(const cv::Mat &image, cameras::IntrinsicBase *cam)
    -> cv::Mat {
  // Only support pinhole cameras
  if (not cameras::isPinhole(cam->getType())) {
    std::cout << "WARNING: Unsupported camera type! Undistortion skipped\n";
    return image;
  }

  // Basic pinhole has no distortion
  if (cam->getType() == cameras::PINHOLE_CAMERA) {
    return image;
  }

  // Get the intrinsic matrix
  auto pCam = dynamic_cast<cameras::Pinhole_Intrinsic *>(cam);
  cv::Mat mtx;
  cv::eigen2cv(pCam->K(), mtx);

  // Get the distortion parameters
  auto dist = cam->getParams();
  if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL1) {
    dist = {dist[3], 0., 0., 0.};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL3) {
    dist = {dist[3], dist[4], 0., 0., dist[5]};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_BROWN) {
    dist = {dist[3], dist[4], dist[6], dist[7], dist[5]};
  } else if (cam->getType() == cameras::PINHOLE_CAMERA_FISHEYE) {
    dist = {dist[3], dist[4], 0., 0., dist[5], dist[6]};
  }

  // Calculate the new matrix for cv::undistort
  cv::Size size(image.cols, image.rows);
  cv::Rect roi;
  mtx = cv::getOptimalNewCameraMatrix(mtx, dist, size, 0., size, &roi);
  cv::Mat result;
  cv::undistort(image, result, mtx, dist);

  // Crop to the ROI
  cv::Mat ret;
  result(roi).copyTo(ret);

  return ret;
}

void ScaleLandmarks(pgs::Landmarks &ldms, const double scale) {
  for (auto &[_, ldm] : ldms) {
    if (ldm.X) {
      ldm.X.value() *= scale;
    }
  }
}

void WriteOBJ(const fs::path &path, const pgs::Landmarks &ldms) {
  // Open the file
  std::ofstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  }

  // Write vertices
  for (const auto &[_, ldm] : ldms) {
    if (ldm.X) {
      const auto &pt = ldm.X.value();
      file << "v " << pt.x() << " " << pt.y() << " " << pt.z() << "\n";
    }
  }

  // Close file
  file.flush();
  file.close();
  if (file.fail()) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
}

void WritePLY(const fs::path &path, const pgs::Landmarks &ldms) {
  // Iterate the vertices first
  std::size_t numVs{0};
  std::stringstream ss;
  for (const auto &[_, ldm] : ldms) {
    if (ldm.X) {
      ++numVs;
      const auto &pt = ldm.X.value();
      ss << pt.x() << " " << pt.y() << " " << pt.z() << " ";
      ss << 255 << " " << 255 << " " << 0 << "\n";
    }
  }

  // Open the file
  std::ofstream file{path};
  if (not file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + path.string());
  }
  // Write the header
  file << "ply\n";
  file << "format ascii 1.0\n";
  file << "element vertex " << numVs << "\n";
  file << "property float x\n";
  file << "property float y\n";
  file << "property float z\n";
  file << "property uchar red\n";
  file << "property uchar green\n";
  file << "property uchar blue\n";
  file << "end_header\n";

  // Write vertices
  file << ss.rdbuf();

  // Close file
  file.flush();
  file.close();
  if (file.fail()) {
    throw std::runtime_error("Failed to write file: " + path.string());
  }
}

void WriteMesh(const fs::path &path, const pgs::Landmarks &ldms) {
  if (el::is_file_type(path, "obj")) {
    WriteOBJ(path, ldms);
  } else if (el::is_file_type(path, "ply")) {
    WritePLY(path, ldms);
  } else {
    throw std::runtime_error("ERROR: Unrecognized mesh type: " +
                             path.extension().string());
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
}

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
    ("marker-size,s", po::value<double>()->required(), "ArUco marker size in desired world units")
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

  // Get the input and output files
  fs::path sfmPath = args["input-scene"].as<std::string>();

  // Marker size (0.47 cm for the sample square)
  auto markerSize = args["marker-size"].as<double>();

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
  ar::DetectorParameters params;
  params.useAruco3Detection = true;
  params.detectInvertedMarker = args["detect-inverted"].as<bool>();
  auto minMarkerSize = static_cast<double>(args["min-marker-pix"].as<int>());

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
    // Ignore if it doesn't pass the filter
    if (not filter(view)) {
      if (bar) {
        bar->tick();
      }
      continue;
    }

    // Load the image
    auto path = view->s_Img_path;
    auto image = cv::imread(sfmRoot / path);

    // Undistort the images
    if (undistortImages) {
      auto cam = sfmData.intrinsics.at(view->id_intrinsic);
      image = UndistortImage(image, cam.get());
    }

    // Detect markers
    params.minMarkerLengthRatioOriginalImg =
        minMarkerSize / static_cast<double>(std::max(image.rows, image.cols));
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
      for (int i = 0; i < corners.size(); i++) {
        // Get the stored landmark
        pgs::Landmark *ldm;
        const auto cornerID = GetLandmarkID(markerID, i);
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
    if (not args["no-ransac"].as<bool>()) {
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

  // Measure marker sizes
  std::cout << "Measuring landmark distances\n";
  std::vector<double> scales;
  for (const auto &markerID : markerIDs) {
    // Keep a list of observed and expected distances
    for (int i = 0; i < 4; i++) {
      const auto nI = i == 3 ? 0 : i + 1;
      auto c0 = landmarks[GetLandmarkID(markerID, i)].X;
      auto c1 = landmarks[GetLandmarkID(markerID, nI)].X;
      if (not c0 or not c1) {
        continue;
      }

      auto distExpected = markerSize;
      auto distObserved = (c1.value() - c0.value()).norm();
      scales.emplace_back(distExpected / distObserved);
    }
  }
  if (scales.empty()) {
    std::cout << "ERROR: No landmark distances calculated!\n";
    return NO_SCALES;
  }

  // Calculate the scale
  std::cout << "Calculating scale factor from " << scales.size()
            << " distance measurements\n";
  if (scales.size() < 10) {
    std::cout << "WARNING: Final scale factor may be sensitive to noise!\n";
  }
  auto scale =
      std::accumulate(scales.begin(), scales.end(), 0.,
                      [&scales](const auto &a, const auto &b) {
                        return a + b / static_cast<double>(scales.size());
                      });
  std::cout << "Calculated scale factor: " << scale << "\n";

  // Scale and save the scene
  if (args.count("output-scene") > 0) {
    fs::path outPath = args["output-scene"].as<std::string>();
    std::cout << "Saving scaled SfM data\n";
    sfm::ApplySimilarity({{}, scale}, sfmData);
    sfm::Save(sfmData, outPath.string(), sfm::ALL);
  }

  // Write the landmarks mesh
  if (args.count("save-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving unscaled landmark mesh\n";
    fs::path ldmMesh = args["save-landmarks"].as<std::string>();
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    WriteMesh(ldmMesh, landmarks);
  }

  // Write the landmarks mesh
  if (args.count("save-scaled-landmarks") > 0 and not markerIDs.empty()) {
    std::cout << "Saving scaled landmark mesh\n";
    fs::path ldmMesh = args["save-scaled-landmarks"].as<std::string>();
    ScaleLandmarks(landmarks, scale);
    fs::create_directories(fs::weakly_canonical(ldmMesh).parent_path());
    WriteMesh(ldmMesh, landmarks);
  }
  std::cout << "Done.\n";
}
