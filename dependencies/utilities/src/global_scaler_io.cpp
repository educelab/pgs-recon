#include "global_scaler_io.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <educelab/core/utils/Filesystem.hpp>

namespace el = educelab;

namespace {
std::uint32_t swap32(const std::uint32_t v) {
    return v >> 24 & 0x000000FF |
           v >>  8 & 0x0000FF00 |
           v <<  8 & 0x00FF0000 |
           v << 24 & 0xFF000000;
}
} // namespace

void pgs::ScalePLYMesh(const std::filesystem::path &inPath,
                       const std::filesystem::path &outPath, double scale) {
  // -- 1) Open input in binary mode, read header
  std::ifstream fin(inPath, std::ios::binary);
  if (!fin)
    throw std::runtime_error("Cannot open input mesh: " + inPath.string());

  enum Format { ASCII, BIN_LE, BIN_BE } format = ASCII;
  std::size_t vertCount = 0, faceCount = 0;
  bool inVertexElement = false;
  struct Property {
    std::string name;
    std::size_t size;
    std::size_t offset;
  };
  std::vector<Property> vertexProps;
  std::vector<std::string> headerLines;

  std::string line;
  std::size_t currentOffset = 0;
  std::streampos bodyStart = 0;

  auto sizeOfType = [&](const std::string &t) {
    if (t == "char" || t == "int8")
      return std::size_t{1};
    if (t == "uchar" || t == "uint8")
      return std::size_t{1};
    if (t == "short" || t == "int16")
      return std::size_t{2};
    if (t == "ushort" || t == "uint16")
      return std::size_t{2};
    if (t == "int" || t == "int32")
      return std::size_t{4};
    if (t == "uint" || t == "uint32")
      return std::size_t{4};
    if (t == "float" || t == "float32")
      return std::size_t{4};
    if (t == "double" || t == "float64")
      return std::size_t{8};
    throw std::runtime_error("Unsupported PLY property type: " + t);
  };

  while (std::getline(fin, line)) {
    headerLines.push_back(line);
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token == "format") {
      std::string fmt;
      iss >> fmt;
      if (fmt == "ascii")
        format = ASCII;
      else if (fmt == "binary_little_endian")
        format = BIN_LE;
      else if (fmt == "binary_big_endian")
        format = BIN_BE;
    } else if (token == "element") {
      std::string name;
      iss >> name;
      if (name == "vertex") {
        inVertexElement = true;
        iss >> vertCount;
      } else {
        // next element; stop collecting vertex props
        inVertexElement = false;
        if (name == "face")
          iss >> faceCount;
      }
    } else if (inVertexElement && token == "property") {
      std::string sub;
      iss >> sub;
      if (sub == "list") {
        // we do not support list‐typed vertex props
        throw std::runtime_error("Unsupported PLY: vertex property is a list");
      }
      // scalar property: <type> <name>
      std::string propName;
      iss >> propName;
      std::size_t sz = sizeOfType(sub);
      vertexProps.push_back({propName, sz, currentOffset});
      currentOffset += sz;
    } else if (line == "end_header") {
      bodyStart = fin.tellg();
      break;
    }
  }

  if (vertCount == 0 || vertexProps.empty())
    throw std::runtime_error("No vertex properties or zero vertices found");

  // find offsets of x,y,z
  auto findOffset = [&](const std::string &n) {
    for (auto &p : vertexProps)
      if (p.name == n)
        return p.offset;
    throw std::runtime_error("PLY has no \"" + n + "\" property");
  };
  std::size_t offX = findOffset("x");
  std::size_t offY = findOffset("y");
  std::size_t offZ = findOffset("z");
  std::size_t recordSize = currentOffset;

  // -- 2) Open output in binary mode, write header
  std::ofstream fout(outPath, std::ios::binary);
  if (!fout)
    throw std::runtime_error("Cannot open output mesh: " + outPath.string());
  for (auto &h : headerLines) {
    fout << h << "\n";
  }

  // -- 3) Seek input to body
  fin.clear();
  fin.seekg(bodyStart);

  if (format == ASCII) {
    // --- ASCII branch ---
    for (std::size_t i = 0; i < vertCount; ++i) {
      std::getline(fin, line);
      if (!fin)
        throw std::runtime_error("Unexpected EOF in ASCII body");
      std::istringstream iss(line);
      double x, y, z;
      iss >> x >> y >> z;
      x *= scale;
      y *= scale;
      z *= scale;
      // reconstruct: x y z + everything else
      std::string tail;
      std::getline(iss, tail);
      fout << x << " " << y << " " << z << tail << "\n";
    }
    // copy faces & any other ASCII data
    while (std::getline(fin, line)) {
      fout << line << "\n";
    }
  } else {
    // --- Binary branch ---
    bool needsSwap = (format == BIN_BE);
    std::vector<char> rec(recordSize);
    for (std::size_t i = 0; i < vertCount; ++i) {
      fin.read(rec.data(), recordSize);
      if (!fin)
        throw std::runtime_error("Unexpected EOF in binary body");
      // decode, swap if BE
      auto adjust = [&](std::size_t off) {
        std::uint32_t iv;
        std::memcpy(&iv, rec.data() + off, sizeof(iv));
        if (needsSwap)
          iv = swap32(iv);
        float f;
        std::memcpy(&f, &iv, sizeof(f));
        f *= static_cast<float>(scale);
        std::memcpy(&iv, &f, sizeof(f));
        if (needsSwap)
          iv = swap32(iv);
        std::memcpy(rec.data() + off, &iv, sizeof(iv));
      };
      adjust(offX);
      adjust(offY);
      adjust(offZ);
      fout.write(rec.data(), recordSize);
    }
    // copy remainder (faces, other elements) verbatim
    constexpr std::size_t BUF = 1 << 20;
    std::vector<char> buf(BUF);
    while (!fin.eof()) {
      fin.read(buf.data(), BUF);
      std::streamsize n = fin.gcount();
      if (n > 0)
        fout.write(buf.data(), n);
    }
  }
}

void pgs::WriteOBJ(const std::filesystem::path &path, const Landmarks &ldms) {
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

void pgs::WritePLY(const std::filesystem::path &path, const Landmarks &ldms) {
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

void pgs::WriteMesh(const std::filesystem::path &path, const Landmarks &ldms) {
  if (el::is_file_type(path, "obj")) {
    WriteOBJ(path, ldms);
  } else if (el::is_file_type(path, "ply")) {
    WritePLY(path, ldms);
  } else {
    throw std::runtime_error("ERROR: Unrecognized mesh type: " +
                             path.extension().string());
  }
}

void pgs::WriteScaleHistogram(const std::string &path,
                              const std::vector<double> &scales,
                              const double centerValue,
                              const std::string &centerLabel) {
  if (scales.empty()) {
    std::cerr << "Warning: no scale samples to histogram\n";
    return;
  }

  // Histogram parameters
  constexpr int nbins = 50;
  const auto minv = *std::min_element(scales.begin(), scales.end());
  const auto maxv = *std::max_element(scales.begin(), scales.end());
  const auto binw = (maxv - minv) / nbins;

  // Bin counts
  std::vector hist(nbins, 0);
  for (const auto &v : scales) {
    const auto b = std::min(static_cast<int>((v - minv) / binw), nbins - 1);
    hist[b]++;
  }
  const auto maxc = *std::max_element(hist.begin(), hist.end());

  // SVG canvas size
  constexpr int SVGW = 800, SVGH = 600;
  // margin
  constexpr int M = 50;
  // plot width
  constexpr int PW = SVGW - 2 * M;
  // plot height
  constexpr int PH = SVGH - 2 * M;

  std::ofstream svg(path);
  if (!svg) {
    throw std::runtime_error("Cannot open histogram SVG for writing");
  }

  // SVG header
  svg << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
         "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
         "  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
         "<svg width=\""
      << SVGW << "\" height=\"" << SVGH
      << "\"\n"
         "     xmlns=\"http://www.w3.org/2000/svg\"\n"
         "     style=\"background-color:white\">\n";

  // Axes
  svg << "<g stroke=\"black\" stroke-width=\"1\">\n"
         "  <line x1=\""
      << M << "\" y1=\"" << (M + PH) << "\" x2=\"" << (M + PW) << "\" y2=\""
      << (M + PH)
      << "\" />\n"
         "  <line x1=\""
      << M << "\" y1=\"" << M << "\" x2=\"" << M << "\" y2=\"" << (M + PH)
      << "\" />\n"
         "</g>\n";

  // Ticks & labels
  constexpr int numXTicks = 10, numYTicks = 5, tickLen = 6;
  svg << "<g fill=\"black\" font-size=\"12\" font-family=\"sans-serif\">\n";
  // X ticks
  for (int i = 0; i <= numXTicks; ++i) {
    double x = M + i * static_cast<double>(PW) / numXTicks;
    double val = minv + i * (maxv - minv) / numXTicks;
    svg << "<line x1=\"" << x << "\" y1=\"" << (M + PH) << "\" x2=\"" << x
        << "\" y2=\"" << (M + PH + tickLen) << "\" stroke=\"black\" />\n";
    svg << "<text x=\"" << x << "\" y=\"" << (M + PH + tickLen + 15)
        << "\" text-anchor=\"middle\">" << std::fixed << std::setprecision(2)
        << val << "</text>\n";
  }
  // Y ticks
  for (int i = 0; i <= numYTicks; ++i) {
    double y = M + PH - i * static_cast<double>(PH) / numYTicks;
    int cnt = static_cast<int>(i * double(maxc) / numYTicks);
    svg << "<line x1=\"" << M << "\" y1=\"" << y << "\" x2=\"" << (M - tickLen)
        << "\" y2=\"" << y << "\" stroke=\"black\" />\n";
    svg << "<text x=\"" << (M - tickLen - 5) << "\" y=\"" << (y + 4)
        << "\" text-anchor=\"end\">" << cnt << "</text>\n";
  }
  svg << "</g>\n";

  // Bars
  double barW = static_cast<double>(PW) / nbins;
  svg << "<g fill=\"steelblue\">\n";
  for (int i = 0; i < nbins; ++i) {
    double x = M + i * barW;
    double h = static_cast<double>(hist[i]) / maxc * PH;
    double y = M + (PH - h);
    svg << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << (barW - 1)
        << "\" height=\"" << h << "\" />\n";
  }
  svg << "</g>\n";

  // Mean/Median line + label
  double lineX = M + (centerValue - minv) / (maxv - minv) * PW;
  svg << "<line x1=\"" << lineX << "\" y1=\"" << M << "\" x2=\"" << lineX
      << "\" y2=\"" << (M + PH) << "\" stroke=\"red\" stroke-width=\"2\" />\n";
  svg << "<text x=\"" << (lineX + 5) << "\" y=\"" << (M + 20)
      << "\" fill=\"red\" font-size=\"14\">" << centerLabel << " = "
      << std::fixed << std::setprecision(2) << centerValue << "</text>\n";

  // SVG footer
  svg << "</svg>\n";
  svg.close();
}
