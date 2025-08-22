#pragma once

#include <filesystem>

#include "global_scaler_types.hpp"

namespace pgs {

void ScalePLYMesh(const std::filesystem::path &inPath,
                  const std::filesystem::path &outPath, double scale);

void WriteOBJ(const std::filesystem::path &path, const Landmarks &ldms);

void WritePLY(const std::filesystem::path &path, const Landmarks &ldms);

void WriteMesh(const std::filesystem::path &path, const Landmarks &ldms);

/**
 * Write an SVG histogram of the given scales, with a red line at `median`.
 */
void WriteScaleHistogram(const std::string &path,
                         const std::vector<double> &scales,
                         const double centerValue,
                         const std::string &centerLabel);
} // namespace pgs
