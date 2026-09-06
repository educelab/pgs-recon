/**
 * @file
 * @brief `pgs-decimate`: coarsen a mesh as far as a measured deviation budget
 *        allows, and report what that cost.
 *
 * The pipeline's other three decimations are OpenMVS's, and every one of them
 * takes a face fraction or a face count and states no geometric bound. This one
 * takes a bound in the units of the solved scene and *measures* what it
 * achieved rather than predicting it. See ADR 0008.
 */

#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include <educelab/core/utils/String.hpp>

#include <indicators/progress_bar.hpp>

#include "decimate_core.hpp"

namespace dec = pgs::decimate;
namespace el = educelab;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace indicators;

namespace
{

enum EXIT_CODE {
  SUCCESS = 0,
  HELP = 1,
  BAD_ARG = 2,
  IO_ERROR = 3,
  SELF_TEST_FAILED = 4,
  NOT_TRIANGULAR = 5
};

/// A JSON string body: the escapes a filesystem path can actually contain.
auto escape(const std::string& text) -> std::string
{
  std::string out;
  for (const auto c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

auto quote(const std::string& text) -> std::string
{
  return "\"" + escape(text) + "\"";
}

auto boolean(bool value) -> std::string { return value ? "true" : "false"; }

/// A JSON number. libcore's conversion rather than a stream: locale-independent
/// -- an `LC_NUMERIC` spelling the separator `,` would make the report invalid
/// JSON -- and shortest round-trip, so a threshold read back is the one written.
/// JSON has no infinity or NaN, and an unparseable report is worth less than one
/// admitting a value was not a number.
auto number(double value) -> std::string
{
  return std::isfinite(value) ? el::to_string(value) : "null";
}

/// A JSON number, or `null` for a target never given -- which both budgets
/// spell as a non-positive value, the way `Targets::have*` reads them.
auto numberOrNull(double value) -> std::string
{
  return value <= 0.0 ? "null" : number(value);
}

auto countOrNull(std::size_t value) -> std::string
{
  return value == 0 ? "null" : std::to_string(value);
}

void writeDirection(std::ostream& f, const std::string& name,
                    const dec::DirectionStats& stats, const char* trailing)
{
  f << "      " << quote(name) << ": {\n"
    << "        \"max\": " << number(stats.max) << ",\n"
    << "        \"mean\": " << number(stats.mean) << ",\n"
    << "        \"rms\": " << number(stats.rms) << ",\n"
    << "        \"samples\": {\n"
    << "          \"uniform\": " << stats.uniformSamples << ",\n"
    << "          \"curvature_weighted\": " << stats.weightedSamples << ",\n"
    << "          \"vertices\": " << stats.vertexSamples << "\n"
    << "        }\n"
    << "      }" << trailing << "\n";
}

/**
 * @brief Write the report beside the mesh.
 *
 * Hand-rolled because the report is one flat document of a known shape and the
 * utilities carry no JSON dependency. The keys are the interface: the manifest
 * records the file's path under the `deviation` role and a reader goes to it
 * for the numbers, so renaming one is a breaking change.
 */
void writeReport(const fs::path& path, const dec::Report& report)
{
  std::ofstream f{path};
  if (not f.is_open()) {
    throw std::runtime_error("cannot open report for writing: " +
                             path.string());
  }
  f << "{\n"
    << "  \"tool\": \"pgs-decimate\",\n"
    << "  \"input\": " << quote(report.input.string()) << ",\n"
    << "  \"output\": " << quote(report.output.string()) << ",\n"
    << "  \"targets\": {\n"
    << "    \"max_error\": " << numberOrNull(report.targets.maxError) << ",\n"
    << "    \"max_faces\": " << countOrNull(report.targets.maxFaces) << ",\n"
    << "    \"quadric_error\": " << numberOrNull(report.targets.quadricError)
    << ",\n"
    << "    \"prefer\": "
    << quote(report.targets.prefer == dec::Prefer::Faces ? "faces" : "error")
    << "\n"
    << "  },\n"
    << "  \"geometry\": {\n"
    << "    \"preserve_boundary\": " << boolean(report.geometry.preserveBoundary)
    << ",\n"
    << "    \"preserve_topology\": " << boolean(report.geometry.preserveTopology)
    << ",\n"
    << "    \"normal_check\": " << boolean(report.geometry.normalCheck) << ",\n"
    << "    \"optimal_placement\": " << boolean(report.geometry.optimalPlacement)
    << ",\n"
    << "    \"quality_threshold\": " << number(report.geometry.qualityThreshold)
    << "\n"
    << "  },\n"
    << "  \"input_mesh\": {\"vertices\": " << report.inputVertices
    << ", \"faces\": " << report.inputFaces << "},\n"
    << "  \"output_mesh\": {\"vertices\": " << report.outputVertices
    << ", \"faces\": " << report.outputFaces << "},\n"
    << "  \"faces_removed\": " << report.facesRemoved << ",\n"
    << "  \"cleaned\": {\n"
    << "    \"duplicate_vertices\": " << report.cleaned.duplicateVertices
    << ",\n"
    << "    \"unreferenced_vertices\": " << report.cleaned.unreferencedVertices
    << ",\n"
    << "    \"degenerate_faces\": " << report.cleaned.degenerateFaces << "\n"
    << "  },\n"
    // Reported, never repaired: when PreserveTopology stalls the search short
    // of the target, the reason is in the same file as the shortfall.
    << "  \"non_manifold\": {\n"
    << "    \"edges\": " << report.manifold.nonManifoldEdges << ",\n"
    << "    \"vertices\": " << report.manifold.nonManifoldVertices << "\n"
    << "  },\n"
    << "  \"search\": {\n"
    << "    \"rounds\": " << report.attempts.size() << ",\n"
    << "    \"quadric_error\": " << number(report.quadricError) << ",\n"
    << "    \"bound\": " << quote(dec::to_string(report.bound)) << ",\n"
    << "    \"attempts\": [\n";
  for (std::size_t i = 0; i < report.attempts.size(); ++i) {
    const auto& a = report.attempts[i];
    f << "      {\"quadric_error\": " << number(a.quadricError)
      << ", \"faces\": " << a.faces << ", \"max_deviation\": "
      << number(a.deviation) << ", \"feasible\": " << boolean(a.feasible)
      << "}" << (i + 1 < report.attempts.size() ? "," : "") << "\n";
  }
  f << "    ]\n"
    << "  },\n"
    << "  \"deviation\": {\n"
    << "    \"max\": " << number(report.deviation.max) << ",\n"
    // Both directions, because the asymmetry is the point (ADR 0008 §3). Mean
    // and RMS are the uniform pass alone, so they compare with metro's.
    << "    \"directions\": {\n";
  writeDirection(f, "original_to_decimated", report.deviation.originalToDecimated,
                 ",");
  writeDirection(f, "decimated_to_original", report.deviation.decimatedToOriginal,
                 "");
  f << "    }\n"
    << "  },\n"
    << "  \"budget_exceeded\": " << boolean(report.budgetExceeded) << ",\n"
    << "  \"face_budget_met\": " << boolean(report.faceBudgetMet) << ",\n"
    << "  \"pass_through\": " << boolean(report.passThrough) << ",\n"
    << "  \"uvs_dropped\": " << boolean(report.uvsDropped) << ",\n"
    << "  \"reason\": " << quote(report.reason) << ",\n"
    << "  \"elapsed_s\": " << number(report.elapsedSeconds) << "\n"
    << "}\n";
}

void printSummary(const dec::Report& report)
{
  std::cout << "Faces: " << report.inputFaces << " -> " << report.outputFaces
            << " (" << report.facesRemoved << " removed)\n";
  std::cout << "Measured deviation: max " << report.deviation.max
            << ", mean " << report.deviation.originalToDecimated.mean << " / "
            << report.deviation.decimatedToOriginal.mean
            << ", RMS " << report.deviation.originalToDecimated.rms << " / "
            << report.deviation.decimatedToOriginal.rms << "\n";
  std::cout << "Bound: " << dec::to_string(report.bound) << " after "
            << report.attempts.size() << " round(s); quadric error "
            << report.quadricError << "\n";
  std::cout << report.reason << "\n";
}

}  // namespace

auto main(int argc, char* argv[]) -> int
{
  // clang-format off
  po::options_description required("required arguments");
  required.add_options()
    ("help,h", "print help message")
    ("input-mesh,i", po::value<std::string>(), "input mesh file (ply, obj)")
    ("output-mesh,o", po::value<std::string>(), "output mesh file (ply, obj)")
  ;

  po::options_description targets("targets (at least one required)");
  targets.add_options()
    ("max-error,e", po::value<double>()->default_value(0.0), "deviation budget: the largest distance, in the mesh's own units, that any point of either surface may end up from the other. Measured on the result, not predicted. 0 disables it")
    ("max-faces,f", po::value<long long>()->default_value(0), "face budget. 0 disables it")
    ("quadric-error", po::value<double>()->default_value(0.0), "raw vcglib quadric threshold. UNITLESS and not comparable between meshes; an escape hatch that turns the search off. 0 disables it")
    ("prefer", po::value<std::string>()->default_value("error"), "which target wins when the deviation and face budgets disagree: error, faces. Defaults to error, so the failure mode is a mesh larger than asked for rather than one that lost a feature")
  ;

  po::options_description geometry("geometry options");
  geometry.add_options()
    ("preserve-boundary", po::value<bool>()->default_value(true), "do not collapse boundary edges")
    ("preserve-topology", po::value<bool>()->default_value(true), "refuse collapses that change the topology")
    ("normal-check", po::value<bool>()->default_value(true), "refuse collapses that flip a face normal")
    ("optimal-placement", po::value<bool>()->default_value(true), "place the collapsed vertex where the quadric is minimal, rather than at an endpoint")
    ("quality-threshold", po::value<double>()->default_value(0.3), "penalize collapses producing triangles below this quality, in (0, 0.866]. 0.866 is an equilateral triangle's quality; 0 is not a looser setting but an off switch, and is refused")
  ;

  po::options_description search("search and measurement options");
  search.add_options()
    ("max-rounds", po::value<int>()->default_value(10), "cap on decimate-and-measure rounds")
    ("samples-per-face", po::value<int>()->default_value(10), "uniform samples per face of the coarser mesh, per direction, floored at a million. The measurement is most of the wall clock; lower it to trade sharpness for time")
    ("curvature-samples", po::value<bool>()->default_value(true), "add a curvature-weighted pass at half the uniform count, biasing samples toward where deviation is largest. Counts toward the max only")
    ("report", po::value<std::string>(), "write a JSON report of what was done to this path")
    ("progress,p", po::value<bool>()->default_value(false), "show a progress bar")
    ("self-test", po::bool_switch(), "decimate a generated sphere and check the measured deviation against the analytic answer, then exit")
  ;

  po::options_description parser("options");
  parser.add(required).add(targets).add(geometry).add(search);
  // clang-format on

  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, parser), args);
    if (argc == 1 or args.count("help") > 0) {
      std::cout << parser << "\n";
      return HELP;
    }
    po::notify(args);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return BAD_ARG;
  }

  if (args["self-test"].as<bool>()) {
    return dec::selfTest(std::cout) ? SUCCESS : SELF_TEST_FAILED;
  }

  for (const auto* name : {"input-mesh", "output-mesh"}) {
    if (args.count(name) == 0) {
      std::cerr << "ERROR: --" << name << " is required\n";
      return BAD_ARG;
    }
  }

  dec::Targets wanted;
  wanted.maxError = args["max-error"].as<double>();
  wanted.quadricError = args["quadric-error"].as<double>();
  // `Targets::have*` reads non-positive as "not given", so a negative budget
  // would silently drop the bound the caller asked for -- or, with a second
  // budget alive, coarsen against a target they never set. Signed, and checked
  // here, so `--max-faces -1` cannot arrive as a face budget of 2^64-1 either.
  const auto maxFaces = args["max-faces"].as<long long>();
  if (wanted.maxError < 0.0 or wanted.quadricError < 0.0 or maxFaces < 0) {
    std::cerr << "ERROR: budgets cannot be negative; pass 0 to leave one "
                 "unset\n";
    return BAD_ARG;
  }
  wanted.maxFaces = static_cast<std::size_t>(maxFaces);
  const auto prefer = el::to_lower_copy(args["prefer"].as<std::string>());
  if (prefer == "error") {
    wanted.prefer = dec::Prefer::Error;
  } else if (prefer == "faces") {
    wanted.prefer = dec::Prefer::Faces;
  } else {
    std::cerr << "ERROR: --prefer must be \"error\" or \"faces\"\n";
    return BAD_ARG;
  }
  if (not wanted.haveError() and not wanted.haveFaces() and
      not wanted.haveQuadric()) {
    std::cerr << "ERROR: give at least one of --max-error, --max-faces or "
                 "--quadric-error; there is no default coarsening\n";
    return BAD_ARG;
  }

  dec::Geometry geo;
  geo.preserveBoundary = args["preserve-boundary"].as<bool>();
  geo.preserveTopology = args["preserve-topology"].as<bool>();
  geo.normalCheck = args["normal-check"].as<bool>();
  geo.optimalPlacement = args["optimal-placement"].as<bool>();
  geo.qualityThreshold = args["quality-threshold"].as<double>();
  // Not merely out of range: vcglib clamps a triangle's quality to this and then
  // divides by it, so 0 makes every collapse infinitely expensive and the goal
  // is reached before the first one. The mesh passes through untouched and
  // nothing in the report points at the flag that did it.
  if (geo.qualityThreshold <= 0.0 or geo.qualityThreshold > 0.866) {
    std::cerr << "ERROR: --quality-threshold must be in (0, 0.866]; 0 admits "
                 "no collapse at all and 0.866 is the most an equilateral "
                 "triangle can score\n";
    return BAD_ARG;
  }

  dec::Measurement measurement;
  measurement.samplesPerFace = args["samples-per-face"].as<int>();
  measurement.curvatureWeighted = args["curvature-samples"].as<bool>();
  if (measurement.samplesPerFace < 1) {
    std::cerr << "ERROR: --samples-per-face must be at least 1; the deviation "
                 "is the guarantee, and it cannot be measured with no "
                 "samples\n";
    return BAD_ARG;
  }

  const auto maxRounds = args["max-rounds"].as<int>();
  if (maxRounds < 1) {
    std::cerr << "ERROR: --max-rounds must be at least 1\n";
    return BAD_ARG;
  }

  dec::Report report;
  report.input = fs::absolute(args["input-mesh"].as<std::string>());
  report.output = fs::absolute(args["output-mesh"].as<std::string>());
  report.targets = wanted;
  report.geometry = geo;

  const auto started = std::chrono::steady_clock::now();

  dec::Mesh mesh;
  try {
    // Flushed, not buffered: see the note on the round callback below.
    std::cout << "Loading mesh: " << report.input << std::endl;
    dec::read(report.input, mesh, report.uvsDropped);
  } catch (const dec::NotTriangular& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return NOT_TRIANGULAR;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  if (report.uvsDropped) {
    // Loud: a silently grey mesh is a nasty surprise, and preserving UVs needs
    // seam-aware guards whose failure no deviation measurement would catch.
    std::cout << "WARNING: the input carries texture coordinates and they are "
                 "being dropped. The output is geometry only; re-texture it if "
                 "you need a textured mesh.\n";
  }

  report.cleaned = dec::condition(mesh);
  std::cout << "Cleaned: " << report.cleaned.duplicateVertices
            << " duplicate, " << report.cleaned.unreferencedVertices
            << " unreferenced vertices, " << report.cleaned.degenerateFaces
            << " degenerate faces" << std::endl;
  report.manifold = dec::countNonManifold(mesh);
  if (report.manifold.nonManifoldEdges > 0 or
      report.manifold.nonManifoldVertices > 0) {
    std::cout << "WARNING: mesh is non-manifold ("
              << report.manifold.nonManifoldEdges << " edges, "
              << report.manifold.nonManifoldVertices
              << " vertices). Not repaired; with --preserve-topology 1 this is "
                 "what stalls a search short of its target."
              << std::endl;
  }
  std::cout << "Decimating " << mesh.fn << " faces" << std::endl;

  std::unique_ptr<ProgressBar> bar;
  if (args["progress"].as<bool>()) {
    bar = std::make_unique<ProgressBar>(
        option::BarWidth{50}, option::Start{" ["},
        option::ForegroundColor{Color::unspecified},
        option::MaxProgress{static_cast<std::size_t>(maxRounds)});
  }
  const auto onRound = [&bar](const dec::Attempt& round) {
    if (bar) {
      bar->tick();
    }
    // Flushed: redirected to a log, as the stage runs it, std::cout is fully
    // buffered, and an unflushed log makes a grinding search look like a hang.
    std::cout << "  round: quadric error " << round.quadricError << " -> "
              << round.faces << " faces, max deviation " << round.deviation
              << (round.feasible ? " (within budget)" : " (over budget)")
              << std::endl;
  };

  try {
    dec::coarsen(mesh, report.output, wanted, geo, measurement, maxRounds,
                 report, onRound);
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return IO_ERROR;
  }
  if (bar and not bar->is_completed()) {
    bar->mark_as_completed();
  }

  report.elapsedSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  printSummary(report);

  // `printSummary` has just stated the outcome in words; these add only the
  // severity and the budget it does not carry. Neither is an error.
  if (report.budgetExceeded) {
    std::cout << "WARNING: the measured deviation is over the budget of "
              << wanted.maxError << ".\n";
  }
  if (not report.faceBudgetMet) {
    std::cout << "WARNING: the face budget of " << wanted.maxFaces
              << " was not met.\n";
  }

  if (args.count("report") > 0) {
    try {
      writeReport(fs::absolute(args["report"].as<std::string>()), report);
    } catch (const std::exception& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return IO_ERROR;
    }
  }

  return SUCCESS;
}
