# Specification: Read PGS datasets through one versioned loader

**Track ID:** pgs-dataset-loader_20261118
**Type:** Feature
**Created:** 2026-11-18
**Status:** Draft

## Summary

Make every pgs-recon reader of a PGS dataset's `metadata.json` comply with the v1
scan schema (`pgs-scan/docs/scan-schema-v1.md`, `pgs_scan/schemas/scan-v1.json`),
by routing all of them through one loader that dispatches on the **schema
version**. The v2 reader then lands later as an addition, not a rewrite.

## Context

pgs-scan and PGS Spectral now publish a formal dataset schema. v1 is final;
v2 is in progress and will be written by pgs-scan 2.0. Today seven modules
(`pgs_data`, `list_complete`, `detect_missing`, `quality_check`, `scan_info`,
`convert`, `retexture`) each parse the raw dict themselves, none checks the
version, and several diverge from what v1 promises (see Problem). Terms below
are as defined in `CONTEXT.md`: **image set**, **PGS dataset**, **schema
version**, **derived dataset**, **capture**, **capture position**.

## User Story

As an EduceLab researcher reconstructing PGS datasets, I want pgs-recon to read
any v1 dataset the way the schema defines it, and to refuse a version it does not
understand, so that a scan is never silently misread — now, or once pgs-scan 2.0
starts writing v2.

## Problem (v1 gaps found in review)

1. No reader checks `schema_version`; a v2 dataset would be read as v1.
2. v1 merges capture positions that read back alike (#79 upstream), so the
   index-built grid in `neighbor_lookup_gridscan` misplaces every later position
   and produces wrong view pairs. Today it only warns.
3. The importer indexes `EXIF:Make`, `EXIF:Model`, `EXIF:FocalLength`
   directly; v1 writes them only when set, so one image without them fails the
   whole import.
4. A PGS Spectral / `STATIC` dataset fails late with a `KeyError`
   (`is_absolute_pos`) after exiftool has read every image; `init_sfm_pgs`'s
   docstring promises a `None` return it never makes.
5. `scan_is_complete` compares a file *count*, so an extra matching file can mask
   a missing one; `rglob` also finds nested `raw/` and `processed/` datasets
   without saying what they are.
6. `quality_check` splits filenames by hand, reads `desc` unguarded, and assumes
   capture 1 is IR.
7. `pgs-convert` writes a non-schema top-level `enhancements` key instead of a
   derived dataset's `processing` block, keeps the source's uuid, and copies
   `complete: true` onto a filtered subset.
8. `apps/scan_info.py` imports from `apps/list_complete.py` (apps must not import
   each other).

## Acceptance Criteria

- [ ] `pgs_recon/pgs_data/` is a package; `__init__` and `dataset.py`/`v1.py`
      import only the standard library; `importer.py` holds the rig-prior import.
- [ ] Version resolved as `schema_version` → `metadata_version` → `"1.0"`; a newer
      minor loads (debug log); an unknown major raises `UnsupportedSchemaVersion`.
- [ ] `v1.py` is the only module that names v1 metadata keys; no app touches the
      raw dict except through `PGSDataset` (and `convert`, only via `v1.derive`).
- [ ] All seven readers use `load_pgs_dataset`; import and retexture stop on a
      `PGSDatasetError`, inventory tools report it as a status and keep walking.
- [ ] The v1 reader places each capture position on its grid cell by coordinates;
      a merged position no longer shifts its successors' view pairs (tested).
- [ ] Missing EXIF `Make`/`Model`/`FocalLength` falls back to `scanner.cameras`
      (logged); a view with nothing usable is skipped with a warning, not fatal.
- [ ] The import refuses a non-`ROW_*` path before exiftool runs, saying why.
- [ ] Completeness = completion flag **and** every expected filename present,
      shared by `pgs-list-complete` and `pgs-detect-missing`.
- [ ] `pgs-convert` output is a v1 derived dataset that validates against
      `scan-v1.json` (`processing.product == "convert"`, new uuid).
- [ ] Vendored v1 fixtures validate against `scan-v1.json` (skips without
      `jsonschema`); the full suite passes; ADR 0013, `CLAUDE.md` and
      `docs/migrating-to-2.0.md` updated.

## Dependencies

- Existing code: `pgs_recon/pgs_data.py`, `pgs_recon/utils/scan_names.py`, the
  seven readers, `apps/reconstruct.py` (imports `init_sfm_pgs`,
  `get_tag_option`), `tests/test_pgs_data.py`.
- External (read-only): `pgs-scan` repo for `scan-v1.json` and a v1 golden.
- No dependency on the open `py312-noble_20261118` track.

## Out of Scope

- The v2 reader and `v2.derive` — a separate track, started once pgs-scan 2.0.0
  is tagged. This track only builds the seam it plugs into.
- Supporting `STATIC` (PGS Spectral) datasets in the rig-prior import; they stay
  refused until that is added explicitly. Inventory tools and convert do accept
  them.
- Runtime JSON Schema validation (`jsonschema` stays test-only).
- A flag to hide derived datasets from `pgs-list-complete`.
- Changes to the generic (non-PGS) importers beyond moving `get_tag_option`.

## Technical Notes

Decisions settled in design review:

- **Package layout** — `pgs_recon/pgs_data/`:
  - `__init__.py`: re-exports `PGSDataset`, `load_pgs_dataset`, the errors.
    Never imports `importer`, so inventory tools need no numpy/scipy/exiftool.
  - `dataset.py`: frozen `PGSDataset` (prefix, ext, captures, capture positions
    with grid cells, cameras: id/serial/label/pose priors, grid parameters,
    `complete`, derived flag + source, version, `.raw`); version resolution;
    `{major: reader}` registry; `expected_images()`, `is_complete()`.
  - `v1.py`: `read_v1` and `derive` — the only place v1 key names appear.
  - `importer.py`: today's `pgs_data.py` (rig-prior import, neighbour lookup,
    `init_sfm_pgs`, `main`). `pgs-import` → `pgs_recon.pgs_data.importer:main`.
- **Fields are named for concepts, not keys** — e.g. camera `label` from v1
  `desc` (else `""`). Cameras are looked up by `id`, which v1 and v2 both keep
  equal to the list position.
- **Errors** — `PGSDatasetError` ⊃ `MissingMetadata`, `MalformedMetadata` (names
  file and key path), `UnsupportedSchemaVersion`. The reader checks only the
  fields it reads.
- **Grid cells** — the reader gives each capture position a `(layer, row, col)`
  or `None`; `neighbor_lookup_gridscan` consumes cells only. v1 snap: per axis
  with non-zero step, `round((pos − origin) / stepsize)`; reject if farther than
  `max(jitter, stepsize/4)` from the cell centre (warn, no cell, views
  unmatched). Two positions in one cell are both kept and neighbours. A merged
  position's lost images leave an empty cell, logged. (v2 will assign cells
  from the planned index.)
- **EXIF fallback** — `Make`/`Model` from `scanner.cameras[cam]` when non-empty,
  `FocalLength` from `focal length (mm)` when > 0; each use logged.
- **Completeness** — `scan.complete` (or legacy `success`) true **and** every
  `{prefix}{cam:03}_{pos:05}_{cap:02}.{ext}` over positions × captures × that
  capture's cameras exists; extras ignored.
- **Nested datasets** — `pgs-list-complete` reports every PGS dataset found,
  derived and `raw/` included; `pgs-scan-info` marks a derived one with its
  product and source uuid.
- **convert** — via `v1.derive`: new uuid4 (36 chars), `processing =
  {product: "convert", source: {uuid, output_dir}, options: {file_type,
  commands, filters}, datetime_start, datetime_end, frames: {}, failed}`; drop
  top-level `enhancements`; a filtered run writes `complete: false`.
- **`get_tag_option`** moves to `pgs_recon/utils/exif.py`.
- **Fixtures** — `tests/fixtures/scan/`: `scan-v1.json`, a v1 pgs-scan golden
  (record the pgs-scan commit it came from; may be after v1.4.0), and a minimal
  PGS Spectral v1 example. Loader tests are stdlib-only.
- **ADR 0013** — "Read PGS datasets through one versioned loader".
- `CONTEXT.md` glossary changes for this track are already made (uncommitted).

---

_Generated by Conductor. Review and edit as needed._
