#pragma once

/**
 * @file
 * @brief Read and write the array-shaped `.npy` files `pgs-localize-joint`
 *        exchanges: an H x W x 3 float32 position map, and N x K float64
 *        tables of correspondences.
 *
 * Beside `Npy.hpp` rather than in it: that header is the 4x4 transform's, and
 * is deliberately narrow. This one is as narrow for its own shapes -- v1.0
 * headers, little-endian, C order, exactly the dtype and rank asked for --
 * and refuses anything else by name, for the reason `Npy.hpp` gives: an array
 * read as the wrong layout is a silent misregistration.
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Npy.hpp"

namespace pgs::npy
{

namespace detail
{

inline void writeHeader(std::ofstream& file, const std::string& descr,
                        const std::string& shape)
{
  const std::string dict = "{'descr': '" + descr +
                           "', 'fortran_order': False, 'shape': " + shape +
                           ", }";
  const std::size_t headerLen = ((dict.size() + 1 + 10 + 63) / 64) * 64 - 10;
  std::string header(headerLen, ' ');
  std::copy(dict.begin(), dict.end(), header.begin());
  header[headerLen - 1] = '\n';
  const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  file.write(reinterpret_cast<const char*>(magic), 8);
  const auto hlen = static_cast<std::uint16_t>(headerLen);
  file.write(reinterpret_cast<const char*>(&hlen), 2);
  file.write(header.data(), static_cast<std::streamsize>(headerLen));
}

/// The header dict of a v1.0 file, leaving @p file at the first data byte.
inline auto readHeader(std::ifstream& file, const std::filesystem::path& path)
    -> std::string
{
  char magic[8]{};
  file.read(magic, 8);
  const char expected[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  if (not file or not std::equal(expected, expected + 6, magic)) {
    throw BadNpy("not a .npy file: " + path.string());
  }
  if (magic[6] != 1) {
    throw BadNpy("unsupported .npy version in " + path.string());
  }
  std::uint16_t len{0};
  file.read(reinterpret_cast<char*>(&len), 2);
  std::string header(len, '\0');
  file.read(header.data(), len);
  if (not file) {
    throw BadNpy("truncated .npy header: " + path.string());
  }
  if (header.find("'fortran_order': False") == std::string::npos) {
    throw BadNpy("expected C order in " + path.string() + ": " + header);
  }
  return header;
}

/// The integers of the header's `shape` tuple.
inline auto shapeOf(const std::string& header, const std::filesystem::path& path)
    -> std::vector<std::size_t>
{
  std::smatch m;
  if (not std::regex_search(header, m, std::regex(R"('shape':\s*\(([^)]*)\))"))) {
    throw BadNpy("no shape in " + path.string() + ": " + header);
  }
  std::vector<std::size_t> dims;
  const std::string inner = m[1];
  const std::regex num(R"(\d+)");
  for (auto it = std::sregex_iterator(inner.begin(), inner.end(), num);
       it != std::sregex_iterator(); ++it) {
    dims.push_back(std::stoull(it->str()));
  }
  return dims;
}

}  // namespace detail

/// Write a CV_32FC3 image as an (H, W, 3) float32 array.
inline void writePositionMap(const std::filesystem::path& path,
                             const cv::Mat& position)
{
  if (position.type() != CV_32FC3) {
    throw BadNpy("a position map is CV_32FC3");
  }
  std::ofstream file(path, std::ios::binary);
  if (not file) {
    throw BadNpy("cannot write: " + path.string());
  }
  detail::writeHeader(file, "<f4",
                      "(" + std::to_string(position.rows) + ", " +
                          std::to_string(position.cols) + ", 3)");
  for (int r = 0; r < position.rows; ++r) {
    file.write(reinterpret_cast<const char*>(position.ptr<cv::Vec3f>(r)),
               static_cast<std::streamsize>(position.cols * 3 * 4));
  }
  if (not file) {
    throw BadNpy("short write: " + path.string());
  }
}

/// Read an (H, W, 3) float32 array as a CV_32FC3 image.
inline auto readPositionMap(const std::filesystem::path& path) -> cv::Mat
{
  std::ifstream file(path, std::ios::binary);
  if (not file) {
    throw BadNpy("cannot read: " + path.string());
  }
  const auto header = detail::readHeader(file, path);
  if (header.find("'<f4'") == std::string::npos) {
    throw BadNpy("expected float32 ('<f4') in " + path.string());
  }
  const auto dims = detail::shapeOf(header, path);
  if (dims.size() != 3 or dims[2] != 3) {
    throw BadNpy("expected an (H, W, 3) array in " + path.string() + ": " +
                 header);
  }
  cv::Mat out(static_cast<int>(dims[0]), static_cast<int>(dims[1]), CV_32FC3);
  for (int r = 0; r < out.rows; ++r) {
    file.read(reinterpret_cast<char*>(out.ptr<cv::Vec3f>(r)),
              static_cast<std::streamsize>(out.cols * 3 * 4));
  }
  if (not file) {
    throw BadNpy("truncated .npy data: " + path.string());
  }
  return out;
}

/// Write rows of equal length as an (N, K) float64 array.
inline void writeTable(const std::filesystem::path& path,
                       const std::vector<std::vector<double>>& rows,
                       const std::size_t columns)
{
  std::ofstream file(path, std::ios::binary);
  if (not file) {
    throw BadNpy("cannot write: " + path.string());
  }
  detail::writeHeader(file, "<f8",
                      "(" + std::to_string(rows.size()) + ", " +
                          std::to_string(columns) + ")");
  for (const auto& row : rows) {
    if (row.size() != columns) {
      throw BadNpy("ragged table written to " + path.string());
    }
    file.write(reinterpret_cast<const char*>(row.data()),
               static_cast<std::streamsize>(columns * 8));
  }
  if (not file) {
    throw BadNpy("short write: " + path.string());
  }
}

}  // namespace pgs::npy
