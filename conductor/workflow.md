# Development Workflow

## Core Principles

1. **plan.md is the source of truth** for task status within a track.
2. **Tests encouraged, not blocking** (moderate TDD).
3. **CI must pass** before merge.
4. **One branch + GitLab MR per track.**

## TDD Policy: Moderate

Write tests first where it's cheap. It's expected, not mandatory, for:

- Planner/stage logic → `tests/test_stages.py`
- New or changed wrapper flags → the `SURFACES` tables in `test_openmvg.py` /
  `test_openmvs.py` (ADR 0005's enforcement)
- Artifact names → `test_layout.py`
- C++ utilities → extend their `--self-test`

Binary-dependent behaviour that the suite can't reach gets verified in the
Docker image instead (see Verification).

## Task Lifecycle

1. **Select** the next `[ ]` task in the track's `plan.md`; check its dependencies.
2. **Mark** it `[~]`.
3. **Test** (where practical): write the failing test.
4. **Implement** to pass it. Follow `code_styleguides/`.
5. **Refactor** without changing behaviour.
6. **Run** `python3 -m unittest discover -s tests`.
7. **Document deviations** from the spec in the spec, or flag them.
8. **Update docs** that the change affects: `CLAUDE.md`, `CONTEXT.md`, an ADR,
   README.
9. **Mark** it `[x]` in `plan.md`.
10. **Commit** (see below).

## Commits

- Descriptive, imperative subject lines; no prefix format.
  e.g. `Give coarsen and decimate matching interfaces`.
- Body explains why when it isn't obvious.
- Commits are signed via 1Password and may need desktop approval.

## Code Review

- **Non-trivial changes:** review required before merge (human or `/code-review`).
- **Trivial fixes** (typos, doc tweaks, one-line obvious fixes): self-review OK.

## Verification

Manual verification happens **once, at track completion**, not per task or
phase. Before declaring a track done:

1. Full test suite passes locally.
2. Anything touching binaries is exercised in the image:
   ```shell
   docker run --rm -v "$PWD":/src -w /src ghcr.io/educelab/pgs-recon:edge \
     bash -lc 'python3 -m pip install . && python3 -m unittest discover -s tests'
   ```
3. C++ changes: rebuild `dependencies/utilities` in the image (see `CLAUDE.md`)
   and run the tool's `--self-test`.
4. Pipeline changes: a `pgs-recon` run (or `--dry-run`) on a representative
   capture set.
5. Present results to the user for sign-off.

## Quality Gates (before merge)

| Gate | Requirement | How |
| ---- | ----------- | --- |
| Tests | Pass | `python3 -m unittest discover -s tests` |
| CI | GitLab pipeline green (incl. `test:in-image`) | MR pipeline |
| Style | Follows `code_styleguides/` | Review |
| Docs | Affected docs/ADRs updated | Review |
| Lint | — (no linter configured yet; see style guides) | — |

## Commands

```shell
# Setup
python3 -m pip install .

# Tests
python3 -m unittest discover -s tests

# C++ utilities, in the image
docker run --rm -v "$PWD":/src -w /src ghcr.io/educelab/pgs-recon:edge \
  bash -lc 'cmake -S dependencies/utilities -B /tmp/bu -DCMAKE_BUILD_TYPE=Release \
            && cmake --build /tmp/bu -j'
```
