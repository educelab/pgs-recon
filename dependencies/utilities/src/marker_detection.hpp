#pragma once

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/sfm/sfm_data.hpp>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include "marker_types.hpp"

namespace pgs {

using IDList       = std::vector<int>;
using CornersList  = std::vector<cv::Point2f>;
using CornersArray = std::vector<CornersList>;

struct DetectionResult {
    IDList       charucoIDs;
    CornersList  charucoCorners;
    IDList       markerIds;
    CornersArray markerCorners;
    CornersArray rejected;
};

struct RansacObservation {
    using Cam = std::shared_ptr<openMVG::cameras::IntrinsicBase>;
    RansacObservation() = default;
    RansacObservation(openMVG::Vec2 obs, openMVG::Vec3 pt, Cam cam,
                      openMVG::geometry::Pose3 pose)
        : obs{std::move(obs)}, pt{std::move(pt)}, cam{std::move(cam)},
          pose{std::move(pose)} {}
    openMVG::Vec2 obs;
    openMVG::Vec3 pt;
    Cam cam;
    openMVG::geometry::Pose3 pose;
};

// Reprojection inlier threshold (pixels), shared by RANSAC triangulation and
// reprojection-error reporting so the reported inlier set matches the one RANSAC
// actually fit. Sized as a blunder gate: well above detection/pose noise, well
// below the tens-to-hundreds of px a mis-correspondence reprojects to.
inline constexpr double kReprojThresholdPx = 4.0;

// Triangulation
std::pair<bool, openMVG::Vec3> Triangulate(
    const std::vector<RansacObservation> &x);
std::pair<bool, openMVG::Vec3> TriangulateRansac(
    const std::vector<RansacObservation> &x);

// Detection
DetectionResult DetectMarkers(const cv::Mat &image,
                               const cv::aruco::DetectorParameters &params);
cv::aruco::CharucoBoard GenerateBoard(int offset = 0);
DetectionResult DetectBoard(const cv::Mat &image,
                             const cv::aruco::CharucoBoard &board,
                             const cv::aruco::DetectorParameters &params);
DetectionResult DetectSampleSquare(const cv::Mat &image,
                                   const cv::aruco::DetectorParameters &params);

// Utility
std::string GetLandmarkID(int arucoID, int cornerID);
cv::Mat UndistortImage(const cv::Mat &image,
                        openMVG::cameras::IntrinsicBase *cam);

// Scale statistics
struct ScaleStats {
    std::vector<double> scales;
    double summary{1.};
    std::string summaryLabel{"median"};
};

ScaleStats ComputeUmeyamaScaleStats(const Landmarks &landmarks,
                                     const std::set<int> &markerIDs,
                                     double markerSize);
ScaleStats ComputeEdgeScaleStats(const Landmarks &landmarks,
                                  const std::set<int> &markerIDs,
                                  double markerSize);

} // namespace pgs
