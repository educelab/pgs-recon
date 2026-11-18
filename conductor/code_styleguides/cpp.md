# C++ Style Guide

Applies to `dependencies/utilities/` (`pgs-*` C++ tools). Written from the
existing code; match surrounding code where this is silent.

## Language Standard

- **C++17** by default, set per target with `target_compile_features(<t> PUBLIC cxx_std_17)`.
- **C++20** only where required (`pgs-localize`, for bvh v2's `std::span`).
  Code that must meet openMVG's bundled cereal stays in its own C++17 library
  (ADR 0011).

## CMake

- Consume dependencies as targets: `find_package(...)` / `FetchContent` +
  `target_link_libraries`, with generator expressions where needed. No
  hand-rolled `find_path`/`find_library`.
- New superbuild deps get a `dependencies/cmake/Build<Name>.cmake` and pass
  roots (e.g. `-DVCG_ROOT`) the way existing ones do.
- Output names are `pgs-<tool>`.

## Code

- Namespaces under `pgs::` (e.g. `pgs::decimate`).
- `#pragma once` in headers.
- Includes: standard library, then third-party, then project, in separate blocks.
- Trailing return types are used (`auto f() -> T`); member data has a trailing
  underscore (`q_`).
- `@file`/`@brief` Doxygen comments on headers and public types: why, not what.
  Inline comments 2-3 lines max.
- Errors: throw `std::runtime_error` (or a derived type) with context; the CLI
  catches and returns non-zero. Don't print-and-continue.
- CLI options via Boost.Program_options, matching existing tools.

## Testing

- Each tool with non-trivial math carries a `--self-test` that checks results
  against an analytic or constructed answer (see `pgs-decimate`,
  `pgs-localize`). Extend it when changing behaviour.
- Build and run in the published image; no superbuild needed (see `CLAUDE.md`).

## Recommended tooling (not yet adopted)

No `.clang-format` exists. Recommend a future track to add one that matches the
current style (2-space indent, Allman braces for namespaces/classes, ~80 cols)
and run it as a CI check.
