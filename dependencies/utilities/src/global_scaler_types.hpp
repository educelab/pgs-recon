#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <openMVG/sfm/sfm_data.hpp>

namespace pgs {
/** Index number of image in SfM */
using ViewID = openMVG::IndexT;
/** List of a landmark's observations and the view where it was found */
using Observations = std::vector<std::pair<ViewID, openMVG::Vec2>>;

/** Landmark: A collection of 2D observations and the triangulated 3D point */
struct Landmark {
  Landmark() = default;
  explicit Landmark(std::string id) : id{std::move(id)} {}
  std::string id;
  Observations obs;
  std::optional<openMVG::Vec3> X;
};

/** Collection of landmarks indexed by ID */
using Landmarks = std::map<std::string, Landmark>;
} // namespace pgs