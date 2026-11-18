# Implementation Plan: Read PGS datasets through one versioned loader

**Track ID:** pgs-dataset-loader_20261118
**Spec:** [spec.md](./spec.md)
**Created:** 2026-11-18
**Status:** [ ] Not Started

## Overview

Build the loader and v1 reader first, behind tests and fixtures, without changing
any app. Then move the seven readers onto it with behaviour held constant, so
that phase is a pure refactor. Behaviour fixes come third, each against the
loader rather than raw keys. Docs close the track. One branch, one MR.

## Phase 1: `pgs_data` package and the v1 reader

The seam, the v1 reader and its fixtures. No app changes yet beyond the move.

### Tasks

- [ ] Task 1.1: Move `pgs_recon/pgs_data.py` to `pgs_recon/pgs_data/importer.py`
      unchanged; add a light `__init__.py`; point `pgs-import` at
      `pgs_recon.pgs_data.importer:main` and `reconstruct.py`/tests at the new
      path. Move `get_tag_option` to `pgs_recon/utils/exif.py`.
- [ ] Task 1.2: Vendor fixtures into `tests/fixtures/scan/`: `scan-v1.json`, a v1
      pgs-scan golden (record the source commit in a README), a minimal PGS
      Spectral v1 example, plus a merged-positions grid case. Add the schema
      test (skips without `jsonschema`).
- [ ] Task 1.3: Tests first for `dataset.py`: version resolution
      (`schema_version` / `metadata_version` / absent), newer minor accepted,
      unknown major → `UnsupportedSchemaVersion`, `MissingMetadata`,
      `MalformedMetadata` naming the key path.
- [ ] Task 1.4: Implement `dataset.py`: errors, frozen `PGSDataset`, version
      resolution, `{major: reader}` registry, `load_pgs_dataset`.
- [ ] Task 1.5: Tests then `read_v1` in `v1.py`: captures, positions, cameras
      (label from `desc`), grid parameters, derived flag and source, both apps'
      shapes (pgs-scan and PGS Spectral fixtures).
- [ ] Task 1.6: Tests then the v1 grid-cell snap: on-grid, jittered, merged
      position, out-of-tolerance (no cell, warning), two positions in one cell,
      zero step on an axis.
- [ ] Task 1.7: Tests then `expected_images()` and `is_complete()`
      (flag, legacy `success`, missing file, extra file ignored).
- [ ] Task 1.8: Tests then `v1.derive(ds, product, options, ...)`; its output
      validates against `scan-v1.json` (skips without `jsonschema`).
- [ ] Task 1.9: Write ADR 0013 "Read PGS datasets through one versioned loader".

### Verification

- [ ] Suite passes; `python3 -I -c "import pgs_recon.pgs_data"` works with no
      numpy/scipy/exiftool/sfm_utils installed (bare-Python CI job).

## Phase 2: Move the seven readers onto the loader

Refactor only: same output on valid v1 datasets; new errors where the dict
used to `KeyError`.

### Tasks

- [ ] Task 2.1: `importer.py` reads through `load_pgs_dataset` (prefix, ext,
      captures, cameras by `id`, positions); calibration merge by serial.
- [ ] Task 2.2: Move `scan_is_complete` behaviour onto `is_complete()`;
      `list_complete` and `scan_info` use it (removes the app→app import).
- [ ] Task 2.3: `detect_missing` uses `expected_images()`.
- [ ] Task 2.4: `retexture` and `quality_check` use `PGSDataset`.
- [ ] Task 2.5: `convert` reads through the loader (writing still unchanged here).
- [ ] Task 2.6: Error policy: import/retexture stop on `PGSDatasetError` with its
      message; inventory tools report it as a per-directory status and continue
      (`detect_missing` keeps its "missing metadata.json" line).
- [ ] Task 2.7: Grep check: no `['scan']`/`['scanner']` indexing left outside
      `pgs_data/v1.py`.

### Verification

- [ ] Suite passes; `test_pgs_data.py`, `test_retexture.py`, `test_pipeline.py`
      unchanged in what they assert (paths/logger names aside).

## Phase 3: Behaviour fixes

### Tasks

- [ ] Task 3.1: Tests then: import refuses a non-`ROW_*` path at load, before
      exiftool, saying why; fix `init_sfm_pgs`'s docstring.
- [ ] Task 3.2: Tests then: `neighbor_lookup_gridscan` builds from grid cells;
      a merged position no longer shifts its successors' pairs; empty cell logged.
- [ ] Task 3.3: Tests then: EXIF `Make`/`Model`/`FocalLength` fall back to
      `scanner.cameras` (logged); a view with nothing usable is skipped with a
      warning naming the image.
- [ ] Task 3.4: `pgs-list-complete` reports nested `raw/` and `processed/`
      datasets; `pgs-scan-info` marks derived ones (product, source uuid).
- [ ] Task 3.5: `quality_check`: `parse_scan_name`, camera labels, plot titles
      from capture names; unparsable files skipped with a warning.
- [ ] Task 3.6: Tests then: `convert` writes through `v1.derive` (new uuid4,
      `processing` block, commands under `options.commands`, no `enhancements`,
      `complete: false` when filtered); output validates against `scan-v1.json`.

### Verification

- [ ] Suite passes; each Problem item in spec.md has a test or a named manual
      check.

## Phase 4: Docs

### Tasks

- [ ] Task 4.1: `CLAUDE.md` module layout: the `pgs_data` package, `utils/exif.py`,
      the new tests and fixtures.
- [ ] Task 4.2: `docs/migrating-to-2.0.md`: `pgs_data` import paths, convert's
      `processing` block replacing `enhancements`, the import's non-ROW refusal.
- [ ] Task 4.3: Confirm `CONTEXT.md` glossary entries match what shipped.

### Verification

- [ ] Docs reviewed against the code.

## Final Verification

- [ ] All acceptance criteria met
- [ ] Tests passing locally and in `ghcr.io/educelab/pgs-recon:edge`
- [ ] A `pgs-recon -p` run (or `--dry-run`) on a real v1 pgs-scan dataset; the
      view pairs file compared with the pre-change one
- [ ] `pgs-convert`, `pgs-list-complete`, `pgs-scan-info` on a PGS Spectral
      dataset with `processed/`
- [ ] Documentation updated
- [ ] Ready for review

---

_Generated by Conductor. Tasks will be marked [~] in progress and [x] complete._
