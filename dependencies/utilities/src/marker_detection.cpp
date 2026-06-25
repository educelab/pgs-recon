#include "marker_detection.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <utility>

#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/multiview/triangulation_nview.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "RANSAC.hpp"

namespace ar = cv::aruco;
using namespace openMVG;
using namespace ransac;

namespace {

auto EvalTriangulate(const std::vector<pgs::RansacObservation> &x, const Vec3 &X)
    -> RANSACResult<pgs::RansacObservation, double>
{
    using Result = RANSACResult<pgs::RansacObservation, double>;
    Result result;
    result.error = 0.;
    constexpr double threshold = pgs::kReprojThresholdPx;
    for (const auto &ro : x) {
        if (ro.pt.dot(ro.pose(X)) <= 0.0) return Result{};
        const auto err = ro.cam->residual(ro.pose(X), ro.obs).norm();
        if (err < threshold) {
            result.error += err * err;
            result.inliers.push_back(ro);
        }
    }
    if (!result.inliers.empty()) {
        result.fitness = static_cast<double>(result.inliers.size()) /
                         static_cast<double>(x.size());
        result.inlier_rmse = std::sqrt(result.error /
                             static_cast<double>(result.inliers.size()));
    }
    result.success = true;
    return result;
}

} // namespace

namespace pgs {

std::pair<bool, Vec3> Triangulate(const std::vector<RansacObservation> &x)
{
    if (x.size() < 2) return {false, {}};
    std::vector<Vec3>  pts;
    std::vector<Mat34> poses;
    pts.reserve(x.size());
    poses.reserve(x.size());
    for (const auto &ro : x) {
        pts.push_back(ro.pt);
        poses.push_back(ro.pose.asMatrix());
    }
    const Map<const Mat3X> mtx(pts[0].data(), 3, static_cast<int>(pts.size()));
    Vec4 Xh;
    if (!TriangulateNViewAlgebraic(mtx, poses, &Xh)) return {false, {}};
    Vec3 X = Xh.hnormalized();
    if (X.hasNaN()) return {false, {}};
    for (const auto &ro : x) {
        if (ro.pt.dot(ro.pose(X)) <= 0.0) return {false, {}};
    }
    return {true, X};
}

std::pair<bool, Vec3> TriangulateRansac(const std::vector<RansacObservation> &x)
{
    constexpr std::size_t nIters = 1000;
    constexpr std::size_t nSamples = 2;
    constexpr std::uint_fast32_t seed = 0;
    const auto [X, res] = RANSAC(x, Triangulate, EvalTriangulate, nSamples, nIters, seed);
    return {res.success, X};
}

DetectionResult DetectMarkers(const cv::Mat &image,
                               const ar::DetectorParameters &params)
{
    const auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
    const ar::ArucoDetector detector(dict, params);
    DetectionResult res;
    detector.detectMarkers(image, res.markerCorners, res.markerIds, res.rejected);
    return res;
}

cv::aruco::CharucoBoard GenerateBoard(int offset)
{
    auto dict = ar::getPredefinedDictionary(ar::DICT_ARUCO_ORIGINAL);
    dict.bytesList = dict.bytesList({offset, offset + 4}, cv::Range::all());
    return ar::CharucoBoard({3, 3}, 10, 7, dict);
}

DetectionResult DetectBoard(const cv::Mat &image,
                             const ar::CharucoBoard &board,
                             const ar::DetectorParameters &params)
{
    ar::CharucoParameters charucoParams;
    charucoParams.tryRefineMarkers = true;
    const ar::CharucoDetector detector(board, charucoParams, params);
    DetectionResult res;
    detector.detectBoard(image, res.charucoCorners, res.charucoIDs,
                         res.markerCorners, res.markerIds);
    return res;
}

DetectionResult DetectSampleSquare(const cv::Mat &image,
                                   const ar::DetectorParameters &params)
{
    static const auto boardTop = GenerateBoard();
    auto res = DetectBoard(image, boardTop, params);
    if (res.charucoIDs.empty()) res = DetectionResult();

    static const auto boardBot = GenerateBoard(512);
    const auto res2 = DetectBoard(image, boardBot, params);
    if (res2.charucoIDs.empty()) return res;

    for (std::size_t i = 0; i < res2.markerIds.size(); ++i) {
        res.markerIds.push_back(res2.markerIds[i] + 512);
        res.markerCorners.push_back(res2.markerCorners[i]);
    }
    for (std::size_t i = 0; i < res2.charucoIDs.size(); ++i) {
        res.charucoIDs.push_back(res2.charucoIDs[i] + 512);
        res.charucoCorners.push_back(res2.charucoCorners[i]);
    }
    return res;
}

std::string GetLandmarkID(int arucoID, int cornerID)
{
    return std::to_string(arucoID) + "." + std::to_string(cornerID);
}

cv::Mat UndistortImage(const cv::Mat &image, cameras::IntrinsicBase *cam)
{
    if (!cameras::isPinhole(cam->getType())) {
        std::cout << "WARNING: Unsupported camera type! Undistortion skipped\n";
        return image;
    }
    if (cam->getType() == cameras::PINHOLE_CAMERA) return image;

    auto pCam = dynamic_cast<cameras::Pinhole_Intrinsic *>(cam);
    cv::Mat mtx;
    cv::eigen2cv(pCam->K(), mtx);
    auto dist = cam->getParams();
    if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL1)
        dist = {dist[3], 0., 0., 0.};
    else if (cam->getType() == cameras::PINHOLE_CAMERA_RADIAL3)
        dist = {dist[3], dist[4], 0., 0., dist[5]};
    else if (cam->getType() == cameras::PINHOLE_CAMERA_BROWN)
        dist = {dist[3], dist[4], dist[6], dist[7], dist[5]};
    else if (cam->getType() == cameras::PINHOLE_CAMERA_FISHEYE)
        dist = {dist[3], dist[4], 0., 0., dist[5], dist[6]};

    const cv::Size size(image.cols, image.rows);
    cv::Rect roi;
    mtx = cv::getOptimalNewCameraMatrix(mtx, dist, size, 0., size, &roi);
    cv::Mat result;
    cv::undistort(image, result, mtx, dist);
    cv::Mat ret;
    result(roi).copyTo(ret);
    return ret;
}

ScaleStats ComputeUmeyamaScaleStats(const Landmarks &landmarks,
                                     const std::set<int> &markerIDs,
                                     double markerSize)
{
    struct SW { double scale, weight; };
    std::vector<SW> sws;
    sws.reserve(markerIDs.size());

    const std::array<Vec3, 4> refCorners = {{
        {0.0,        0.0,        0.0},
        {markerSize, 0.0,        0.0},
        {markerSize, markerSize, 0.0},
        {0.0,        markerSize, 0.0}
    }};

    for (const auto &mID : markerIDs) {
        std::vector<Vec3> P_ref, P_obs;
        P_ref.reserve(4);
        P_obs.reserve(4);
        for (int c = 0; c < 4; ++c) {
            auto it = landmarks.find(GetLandmarkID(mID, c));
            if (it != landmarks.end() && it->second.X) {
                P_ref.push_back(refCorners[c]);
                P_obs.push_back(it->second.X.value());
            }
        }
        if (P_obs.size() < 3) continue;

        const auto N = static_cast<int>(P_obs.size());
        Mat3X M_ref(3, N), M_obs(3, N);
        for (int i = 0; i < N; ++i) {
            M_ref.col(i) = P_ref[i];
            M_obs.col(i) = P_obs[i];
        }
        const Eigen::Matrix4d T = Eigen::umeyama(M_obs, M_ref, true);
        const double s = T.block<3, 3>(0, 0).col(0).norm();
        sws.push_back({s, static_cast<double>(N)});
    }

    ScaleStats stats;
    stats.scales.reserve(sws.size());
    for (const auto &[scale, weight] : sws) stats.scales.push_back(scale);
    if (sws.empty()) return stats;

    std::sort(sws.begin(), sws.end(), [](const auto &a, const auto &b) {
        return a.scale < b.scale;
    });
    const auto W = std::accumulate(sws.begin(), sws.end(), 0.0,
                                   [](double l, const SW &r) { return l + r.weight; });
    double cum = 0.0;
    for (const auto &[scale, weight] : sws) {
        cum += weight;
        if (cum >= 0.5 * W) { stats.summary = scale; break; }
    }
    return stats;
}

ScaleStats ComputeEdgeScaleStats(const Landmarks &landmarks,
                                  const std::set<int> &markerIDs,
                                  double markerSize)
{
    ScaleStats stats;
    for (const auto &mID : markerIDs) {
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) % 4;
            const auto c0 = landmarks.at(GetLandmarkID(mID, i)).X;
            const auto c1 = landmarks.at(GetLandmarkID(mID, j)).X;
            if (!c0 || !c1) continue;
            const double dist = (*c1 - *c0).norm();
            if (dist < 1e-9) continue;
            stats.scales.push_back(markerSize / dist);
        }
    }
    if (stats.scales.empty()) return stats;

    std::cout << "Calculating scale factor from " << stats.scales.size()
              << " distance measurements\n";
    auto sorted = stats.scales;
    std::sort(sorted.begin(), sorted.end());
    const auto n = sorted.size();
    stats.summary = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n/2-1] + sorted[n/2]);
    return stats;
}

} // namespace pgs
