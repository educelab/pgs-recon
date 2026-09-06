#pragma once

/**
 * @file
 * @brief Error-bounded quadric edge-collapse decimation, and the measurement
 *        that bounds it.
 *
 * The contract is a **measured** distance, not vcglib's collapse metric, which
 * is unitless and under-reports deviation where curvature is high: the threshold
 * is a search variable and the deviation is measured on the result (ADR 0008).
 *
 * Geometry only -- no colour, no texture coordinates, which the CLI drops with a
 * warning, a seam-unaware collapse breaking a parameterization silently.
 */

#include <cstddef>
#include <filesystem>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

namespace pgs::decimate
{

// --- The mesh -------------------------------------------------------------

class DVertex;
class DEdge;
class DFace;

struct DUsedTypes : public vcg::UsedTypes<vcg::Use<DVertex>::AsVertexType,
                                          vcg::Use<DEdge>::AsEdgeType,
                                          vcg::Use<DFace>::AsFaceType> {
};

/**
 * @brief A positional-QEM vertex.
 *
 * `VFAdj`/`Mark` are the collapse's, `Quadric` makes it quadric, and
 * `Normal`/`Quality` are the sampling's. Nothing else -- no colour, no UVs.
 */
class DVertex : public vcg::Vertex<DUsedTypes, vcg::vertex::VFAdj,
                                   vcg::vertex::Coord3f, vcg::vertex::Normal3f,
                                   vcg::vertex::Mark, vcg::vertex::Qualityf,
                                   vcg::vertex::BitFlags>
{
public:
  auto Qd() -> vcg::math::Quadric<double>& { return q_; }

private:
  vcg::math::Quadric<double> q_;
};

class DEdge : public vcg::Edge<DUsedTypes>
{
};

/// `FFAdj` is for the non-manifold counts, which are reported and never repaired.
class DFace : public vcg::Face<DUsedTypes, vcg::face::VFAdj, vcg::face::FFAdj,
                               vcg::face::VertexRef, vcg::face::Normal3f,
                               vcg::face::Mark, vcg::face::BitFlags>
{
};

class Mesh : public vcg::tri::TriMesh<std::vector<DVertex>, std::vector<DFace>>
{
};

using VertexPair = vcg::tri::BasicVertexPair<DVertex>;

class TriEdgeCollapse
    : public vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse,
                                              vcg::tri::QInfoStandard<DVertex>>
{
public:
  using TECQ =
      vcg::tri::TriEdgeCollapseQuadric<Mesh, VertexPair, TriEdgeCollapse,
                                       vcg::tri::QInfoStandard<DVertex>>;
  TriEdgeCollapse(const VertexPair& p, int i, vcg::BaseParameterClass* pp)
      : TECQ(p, i, pp)
  {
  }
};

// --- What the tool was asked for ------------------------------------------

/// Which target decides when the two budgets disagree.
enum class Prefer { Error, Faces };

/// Which bound the result is up against, for the report.
enum class Bound { None, MaxError, MaxFaces, QuadricError };

[[nodiscard]] auto to_string(Bound bound) -> std::string;

/**
 * @brief The geometry knobs, all of them exposed and each negatable.
 *
 * Defaults favour feature preservation. `QualityQuadric` (on) and
 * `ScaleIndependent` (off) are fixed rather than exposed: ADR 0005 makes every
 * flag a permanent surface, the first only shapes collapse order, and the
 * second's normalization would make the reported threshold meaningless.
 */
struct Geometry {
  bool preserveBoundary{true};
  bool preserveTopology{true};
  bool normalCheck{true};
  bool optimalPlacement{true};
  double qualityThreshold{0.3};
};

/// How hard the deviation is measured. `samplesPerFace` counts the faces of the
/// *coarser* mesh, per direction, and is floored at :cpp:var:`kMinSamples`.
struct Measurement {
  int samplesPerFace{10};
  bool curvatureWeighted{true};
};

/// Floor on the uniform sample count per direction, so a mesh coarse enough to
/// ask for only a handful of samples is still measured against something.
constexpr std::size_t kMinSamples = 1000000;

/// The targets, as given. Zero or negative means "not given".
struct Targets {
  double maxError{0.0};
  std::size_t maxFaces{0};
  double quadricError{0.0};
  Prefer prefer{Prefer::Error};

  [[nodiscard]] auto haveError() const -> bool { return maxError > 0.0; }
  [[nodiscard]] auto haveFaces() const -> bool { return maxFaces > 0; }
  [[nodiscard]] auto haveQuadric() const -> bool { return quadricError > 0.0; }
};

// --- What it found --------------------------------------------------------

struct CleanCounts {
  int duplicateVertices{0};
  int unreferencedVertices{0};
  int degenerateFaces{0};
};

struct ManifoldCounts {
  int nonManifoldEdges{0};
  int nonManifoldVertices{0};
};

/// One direction of the symmetric distance, with the sample counts behind it.
struct DirectionStats {
  double max{0.0};
  double mean{0.0};
  double rms{0.0};
  std::size_t uniformSamples{0};
  std::size_t vertexSamples{0};
  std::size_t weightedSamples{0};
};

/**
 * @brief The symmetric sampled Hausdorff distance between two meshes.
 *
 * `max` is the gate: the larger of the two one-sided maxima. Per-direction
 * `mean`/`rms` are the uniform pass alone, so they compare with `metro`'s.
 */
struct Deviation {
  double max{0.0};
  DirectionStats originalToDecimated;
  DirectionStats decimatedToOriginal;
};

/// One decimate-and-measure round. `feasible` means "inside the deviation
/// budget", and is true when no budget was given.
struct Attempt {
  double quadricError{0.0};
  std::size_t faces{0};
  double deviation{0.0};
  bool feasible{false};
};

struct Report {
  std::filesystem::path input;
  std::filesystem::path output;
  Targets targets;
  Geometry geometry;

  std::size_t inputVertices{0};
  std::size_t inputFaces{0};
  std::size_t outputVertices{0};
  std::size_t outputFaces{0};
  std::size_t facesRemoved{0};

  CleanCounts cleaned;
  ManifoldCounts manifold;

  double quadricError{0.0};
  Bound bound{Bound::None};
  std::vector<Attempt> attempts;
  Deviation deviation;

  /// True when `--prefer faces` was honoured at the cost of the error budget.
  bool budgetExceeded{false};
  /// False when a face budget could not be met inside the deviation budget.
  bool faceBudgetMet{true};
  /// True when the mesh was written through because nothing could be collapsed.
  bool passThrough{false};
  /// True when the input carried texture coordinates, which were dropped.
  bool uvsDropped{false};

  std::string reason;
  double elapsedSeconds{0.0};
};

// --- The steps -------------------------------------------------------------

/// A non-triangular face, which the tool refuses rather than fanning.
struct NotTriangular : std::runtime_error {
  explicit NotTriangular(const std::string& what) : std::runtime_error(what) {}
};

/**
 * @brief Read a mesh, refusing anything that is not triangular.
 *
 * A fan is a shape the caller never asked about the deviation of, so a
 * non-triangular face is an error. Throws `std::runtime_error`.
 *
 * @param uvsPresent set when the file carried texture coordinates, then dropped.
 */
void read(const std::filesystem::path& path, Mesh& mesh, bool& uvsPresent);

/// Write a mesh. Format follows the extension (`.ply`, `.obj`).
void write(const std::filesystem::path& path, Mesh& mesh);

/**
 * @brief Remove duplicate, unreferenced and degenerate elements, counting each.
 *
 * Always run: this is the ordinary condition of an MVS mesh, and a search
 * working around it stalls for reasons nothing reports.
 */
auto condition(Mesh& mesh) -> CleanCounts;

/// Count non-manifold edges and vertices. Reported, never repaired.
auto countNonManifold(Mesh& mesh) -> ManifoldCounts;

/**
 * @brief Collapse edges until `quadricError` or `maxFaces` stops it.
 *
 * Either bound may be zero, meaning "not given"; at least one must not be.
 * Returns the resulting face count. Destructive: `mesh` is left compacted.
 */
auto collapse(Mesh& mesh, double quadricError, std::size_t maxFaces,
              const Geometry& geometry) -> std::size_t;

/**
 * @brief The symmetric sampled Hausdorff distance between `original` and
 *        `decimated`.
 *
 * Both meshes are mutated -- normals, quality and topology are rebuilt -- but
 * neither's geometry is touched.
 */
auto measure(Mesh& original, Mesh& decimated, const Measurement& measurement)
    -> Deviation;

/// Called once per search round with the attempt just finished.
using RoundCallback = std::function<void(const Attempt&)>;

/**
 * @brief Decimate `mesh` as far as `targets` allows, writing each improvement
 *        to `output`.
 *
 * Written rather than held, so peak memory stays at the pristine input plus one
 * candidate; the coarsest *feasible* one is what is left on disk.
 *
 * Under `--prefer error` the invariant is unconditional: the file left behind
 * measures no further than `targets.maxError`. Under `--prefer faces` the face
 * budget wins and `Report::budgetExceeded` says so.
 */
void coarsen(Mesh& mesh, const std::filesystem::path& output,
             const Targets& targets, const Geometry& geometry,
             const Measurement& measurement, int maxRounds, Report& report,
             const RoundCallback& onRound = nullptr);

/**
 * @brief Decimate a generated sphere and check the measurement against the
 *        analytic answer.
 *
 * The C++ has no CI coverage, so this checks the measurement without a scan: a
 * sphere's exact distance is `|len(p) - r|`, an answer from outside this code.
 *
 * Returns true if the two agree. Diagnostics go to `out`.
 */
auto selfTest(std::ostream& out) -> bool;

}  // namespace pgs::decimate
