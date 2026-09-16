#pragma once

/**
 * @file
 * @brief Read and write a 4x4 float64 NumPy `.npy` array.
 *
 * The 4x4 similarity transform is this repo's interchange format for a change of
 * frame: `pgs-sfm-orient --save-transform` writes one, `pgs-center
 * --load-transform` and `pgs-localize --sfm-transform` read one, and the Python
 * side moves them with `numpy.load`/`numpy.save`. Both directions live here so
 * the header dialect a writer emits is the one a reader accepts.
 *
 * Deliberately narrow: v1.0 headers, C-contiguous or Fortran-ordered, little-
 * endian float64, shape exactly (4, 4). Anything else is refused by name rather
 * than reinterpreted -- a transform read as the wrong dtype is a rigid motion
 * that no downstream residual can reveal.
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace pgs::npy
{

/// Thrown for a file that is not a 4x4 float64 `.npy`, naming what it was.
class BadNpy : public std::runtime_error
{
public:
  explicit BadNpy(const std::string& what) : std::runtime_error(what) {}
};

/// Write @p mat as NumPy `.npy` v1.0, C-order, little-endian float64.
inline void write(const std::filesystem::path& path, const Eigen::Matrix4d& mat)
{
  const std::string dictStr =
      "{'descr': '<f8', 'fortran_order': False, 'shape': (4, 4), }";
  const std::size_t minLen = dictStr.size() + 1;  // +1 for \n
  // The 10-byte preamble plus the header must total a multiple of 64.
  const std::size_t headerLen = ((minLen + 10 + 63) / 64) * 64 - 10;

  std::string header(headerLen, ' ');
  std::copy(dictStr.begin(), dictStr.end(), header.begin());
  header[headerLen - 1] = '\n';

  std::ofstream file(path, std::ios::binary);
  if (not file) {
    throw BadNpy("cannot write: " + path.string());
  }
  const std::uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
  file.write(reinterpret_cast<const char*>(magic), 8);
  const auto hlen = static_cast<std::uint16_t>(headerLen);
  file.write(reinterpret_cast<const char*>(&hlen), 2);
  file.write(header.data(), static_cast<std::streamsize>(headerLen));
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      const double v = mat(r, c);
      file.write(reinterpret_cast<const char*>(&v), 8);
    }
  }
  if (not file) {
    throw BadNpy("short write: " + path.string());
  }
}

/// Read a 4x4 float64 `.npy`, honoring `fortran_order`.
inline auto read(const std::filesystem::path& path) -> Eigen::Matrix4d
{
  std::ifstream file(path, std::ios::binary);
  if (not file) {
    throw BadNpy("cannot read: " + path.string());
  }

  char magic[8]{};
  file.read(magic, 8);
  const char expected[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  if (not file or not std::equal(expected, expected + 6, magic)) {
    throw BadNpy("not a .npy file: " + path.string());
  }
  // v1.0 has a 2-byte header length; v2.0+ has 4. Only v1 is written here, and
  // numpy only promotes to v2 for headers over 64 KiB, which a 4x4 never needs.
  if (magic[6] != 1) {
    throw BadNpy("unsupported .npy version " + std::to_string(int(magic[6])) +
                 "." + std::to_string(int(magic[7])) + ": " + path.string());
  }
  std::uint16_t headerLen{0};
  file.read(reinterpret_cast<char*>(&headerLen), 2);
  std::string header(headerLen, '\0');
  file.read(header.data(), headerLen);
  if (not file) {
    throw BadNpy("truncated .npy header: " + path.string());
  }

  // The header is a Python dict literal. Substring tests rather than a parser:
  // numpy writes it with a fixed spelling, and every mismatch is reported as
  // one rather than guessed at.
  const auto has = [&header](const char* needle) {
    return header.find(needle) != std::string::npos;
  };
  if (not(has("'<f8'") or has("'=f8'") or has("'f8'"))) {
    throw BadNpy("expected float64 ('<f8') data in " + path.string() + ": " +
                 header);
  }
  if (not has("(4, 4)") and not has("(4,4)")) {
    throw BadNpy("expected a 4x4 array in " + path.string() + ": " + header);
  }
  const bool fortran = has("'fortran_order': True");

  Eigen::Matrix4d mat;
  for (int i = 0; i < 16; ++i) {
    double v{0.0};
    file.read(reinterpret_cast<char*>(&v), 8);
    if (not file) {
      throw BadNpy("truncated .npy data: " + path.string());
    }
    // C-order fills rows; Fortran order fills columns.
    if (fortran) {
      mat(i % 4, i / 4) = v;
    } else {
      mat(i / 4, i % 4) = v;
    }
  }
  return mat;
}

}  // namespace pgs::npy
