#include "localize_camera.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace
{

auto lower(std::string s) -> std::string
{
  for (auto& c : s) {
    c = static_cast<char>(
        std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

}  // namespace

auto pgs::localize::ReadCameraFile(const fs::path& path) -> Camera
{
  std::ifstream f{path};
  if (not f.is_open()) {
    throw BadCamera("cannot read camera file: " + path.string());
  }

  Camera cam;
  bool haveFx{false};
  bool haveFy{false};
  bool haveCx{false};
  bool haveCy{false};
  bool haveW{false};
  bool haveH{false};

  std::string raw;
  while (std::getline(f, raw)) {
    std::istringstream line{raw};
    std::string key;
    if (not(line >> key) or key.empty() or key[0] == '#') {
      continue;
    }
    key = lower(key);

    const auto scalar = [&line, &path, &key]() {
      double v{0.0};
      if (not(line >> v)) {
        throw BadCamera(path.string() + ": key '" + key + "' has no value");
      }
      return v;
    };

    if (key == "fx") {
      cam.fx = scalar();
      haveFx = true;
    } else if (key == "fy") {
      cam.fy = scalar();
      haveFy = true;
    } else if (key == "cx") {
      cam.cx = scalar();
      haveCx = true;
    } else if (key == "cy") {
      cam.cy = scalar();
      haveCy = true;
    } else if (key == "width") {
      cam.width = static_cast<int>(std::lround(scalar()));
      haveW = true;
    } else if (key == "height") {
      cam.height = static_cast<int>(std::lround(scalar()));
      haveH = true;
    } else if (key == "k1" or key == "k2" or key == "k3") {
      cam.disto[static_cast<std::size_t>(key[1] - '1')] = scalar();
      cam.haveDisto = true;
    } else if (key == "p1" or key == "p2") {
      // Read, not ignored: silently dropping a tangential term would localize
      // against a camera model the file does not describe.
      if (scalar() != 0.0) {
        throw BadCamera(path.string() + ": tangential distortion (" + key +
                        ") is not supported; the camera model is radial-only");
      }
    } else if (key == "expected_standoff") {
      cam.expectedStandoff = scalar();
    } else if (key == "standoff_tolerance") {
      cam.standoffTolerance = scalar();
    } else if (key == "pose") {
      std::vector<double> vals;
      double v{0.0};
      while (line >> v) {
        vals.push_back(v);
      }
      if (vals.size() != 16) {
        throw BadCamera(path.string() + ": 'pose' needs 16 values (a 4x4 "
                        "row-major world-to-camera matrix), got " +
                        std::to_string(vals.size()));
      }
      Eigen::Matrix4d m;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          m(r, c) = vals[static_cast<std::size_t>(r * 4 + c)];
        }
      }
      if (std::abs(m(3, 0)) > 1e-9 or std::abs(m(3, 1)) > 1e-9 or
          std::abs(m(3, 2)) > 1e-9 or std::abs(m(3, 3) - 1.0) > 1e-9) {
        throw BadCamera(path.string() +
                        ": 'pose' bottom row is not (0, 0, 0, 1); it is not a "
                        "4x4 rigid transform");
      }
      cam.R = m.block<3, 3>(0, 0);
      cam.t = m.block<3, 1>(0, 3);
      // A non-rigid rotation block is the shape a transposed or scaled pose
      // arrives in, and nothing downstream would notice.
      const Eigen::Matrix3d shouldBeI = cam.R.transpose() * cam.R;
      if ((shouldBeI - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
          1e-6) {
        throw BadCamera(path.string() +
                        ": 'pose' rotation block is not orthonormal");
      }
      if (cam.R.determinant() < 0.0) {
        throw BadCamera(path.string() + ": 'pose' rotation block has a "
                        "negative determinant; it includes a reflection");
      }
      cam.havePose = true;
    }
    // Any other key is a format this tool does not read; leave it be.
  }

  std::string missing;
  const auto want = [&missing](bool have, const char* name) {
    if (not have) {
      missing += (missing.empty() ? "" : ", ");
      missing += name;
    }
  };
  want(haveFx, "fx");
  want(haveCx, "cx");
  want(haveCy, "cy");
  want(haveW, "width");
  want(haveH, "height");
  if (not missing.empty()) {
    throw BadCamera(path.string() + ": missing required key(s): " + missing);
  }
  if (not haveFy) {
    cam.fy = cam.fx;
  }
  if (cam.fx <= 0.0 or cam.fy <= 0.0) {
    throw BadCamera(path.string() + ": focal length must be positive");
  }
  if (cam.width <= 0 or cam.height <= 0) {
    throw BadCamera(path.string() + ": width and height must be positive");
  }
  if (cam.expectedStandoff < 0.0 or cam.standoffTolerance < 0.0) {
    throw BadCamera(path.string() +
                    ": the standoff keys cannot be negative; omit them to "
                    "leave the gate unarmed");
  }
  // Half a gate is not a gate, and the failure mode of letting it through is
  // that the one check independent of the correspondence set silently does
  // nothing.
  if ((cam.expectedStandoff > 0.0) != (cam.standoffTolerance > 0.0)) {
    throw BadCamera(path.string() +
                    ": expected_standoff and standoff_tolerance go together; "
                    "give both or neither");
  }
  return cam;
}

void pgs::localize::WriteCameraFile(const fs::path& path, const Camera& cam)
{
  std::ofstream f{path};
  if (not f.is_open()) {
    throw BadCamera("cannot write camera file: " + path.string());
  }
  // Enough digits to round-trip a double, and a fixed C locale spelling: a
  // camera file read back with a comma decimal separator is not the one written.
  f.imbue(std::locale::classic());
  f << std::setprecision(17);
  f << "fx " << cam.fx << "\n";
  f << "fy " << cam.fy << "\n";
  f << "cx " << cam.cx << "\n";
  f << "cy " << cam.cy << "\n";
  f << "width " << cam.width << "\n";
  f << "height " << cam.height << "\n";
  f << "k1 " << cam.disto[0] << "\n";
  f << "k2 " << cam.disto[1] << "\n";
  f << "k3 " << cam.disto[2] << "\n";
  Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
  m.block<3, 3>(0, 0) = cam.R;
  m.block<3, 1>(0, 3) = cam.t;
  // Carried through so the tool's output stays a valid input: a solved camera
  // seeding the next run keeps the gate armed.
  if (cam.expectedStandoff > 0.0 and cam.standoffTolerance > 0.0) {
    f << "expected_standoff " << cam.expectedStandoff << "\n";
    f << "standoff_tolerance " << cam.standoffTolerance << "\n";
  }
  f << "pose";
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      f << " " << m(r, c);
    }
  }
  f << "\n";
  if (not f) {
    throw BadCamera("short write: " + path.string());
  }
}

auto pgs::localize::RescaleCamera(const Camera& cam, int width, int height)
    -> Camera
{
  if (width <= 0 or height <= 0) {
    throw BadCamera("cannot rescale a camera to a non-positive image size");
  }
  if (cam.width == width and cam.height == height) {
    return cam;
  }
  const auto sx = static_cast<double>(width) / static_cast<double>(cam.width);
  const auto sy = static_cast<double>(height) / static_cast<double>(cam.height);
  if (std::abs(sx - sy) > 1e-6 * std::max(sx, sy)) {
    throw BadCamera("the camera file is " + std::to_string(cam.width) + "x" +
                    std::to_string(cam.height) + " and the query is " +
                    std::to_string(width) + "x" + std::to_string(height) +
                    "; a change of aspect ratio is not a rescale of K");
  }
  Camera out = cam;
  out.fx *= sx;
  out.fy *= sy;
  out.cx *= sx;
  out.cy *= sy;
  out.width = width;
  out.height = height;
  return out;
}

void pgs::localize::DescribeCamera(std::ostream& os, const Camera& cam)
{
  os << "  f = (" << cam.fx << ", " << cam.fy << ") px, pp = (" << cam.cx
     << ", " << cam.cy << "), " << cam.width << "x" << cam.height << "\n";
  if (cam.haveDisto) {
    os << "  radial k = (" << cam.disto[0] << ", " << cam.disto[1] << ", "
       << cam.disto[2] << ")\n";
  }
  if (cam.expectedStandoff > 0.0) {
    os << "  expected standoff " << cam.expectedStandoff << " +/- "
       << cam.standoffTolerance << "\n";
  }
  if (cam.havePose) {
    const Eigen::Vector3d c = cam.center();
    // Both, always: the file stores t and the prose everywhere quotes C, and
    // they differ by a sign and a rotation.
    os << "  camera centre C = (" << c[0] << ", " << c[1] << ", " << c[2]
       << "), world-to-camera t = (" << cam.t[0] << ", " << cam.t[1] << ", "
       << cam.t[2] << ")\n";
  }
}
