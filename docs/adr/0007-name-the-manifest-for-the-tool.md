# Name the run manifest for the tool that owns the directory

**Status: accepted.** Landed in 2.0, alongside the issue #17 series. Directories
written by earlier versions are still read — the rename takes nothing away from
them — see the consequences.

## Context

`pgs-recon` records what a run finished, and with what arguments, in a manifest
at the root of the output directory
([ADR 0004](./0004-staged-resumable-runs.md)). It was called `metadata.json`.

So is something else. An EduceLab **PGS scan directory** — an *input* format,
not ours — describes itself in a `metadata.json`, and seven modules read that
one: `pgs_data.py`, `utils/quality.py`, `apps/scan_info.py`,
`apps/quality_check.py`, `apps/convert.py` (which also *writes* one, for a
converted scan), `apps/detect_missing.py` and
`apps/list_complete.py`. One filename, two unrelated schemas, and nothing but
the containing directory to tell them apart. `pgs-convert -o` produces a
directory holding the scan kind; `pgs-recon -o` produces one holding the run
kind; a person or a script handed a path has to already know which it is.

The generic name also gives a downstream consumer nothing to search for. A
reconstruction that has been copied out of its output directory, or an archive
holding several tools' outputs, carries a `metadata.json` that identifies
neither its producer nor its schema.

## Decision

The manifest is `pgs-recon.json` — named for the tool that owns the directory,
which is the one thing about it that cannot be ambiguous.

`layout.manifest()` writes it. `stages.find_manifest()` *reads* it, falling back
to `metadata.json` when only that is present, so the rename costs an existing
directory nothing: one built by `v1.8.0-alpha.1` resumes with nothing re-run, and
one built by 1.7 is read exactly as well as it was before (which is not very --
it predates stage records, so it has nothing to resume *from*; see
[the migration notes](../migrating-to-2.0.md)). `utils/recon_dir.py` resolves
through the same function, so `pgs-retexture` and `pgs-calibrate` — the two apps
that read a run's manifest — keep working on old directories too.
`pgs-quality-check` is not one of them: it reads a *scan* directory's
`metadata.json`, which is the collision above, not this rename.

The scan `metadata.json` is untouched: it is an input format we do not define.

## Consequences / non-obvious traps

- **This rename is not free the way [ADR 0006](./0006-stage-named-artifacts.md)'s
  was, and for a precise reason.** Artifact names are only ever *written* — the
  manifest locates the artifacts — so changing them cannot invalidate an existing
  directory. The manifest is the one thing located *by name*. Renamed without a
  fallback, every finished output directory would look empty: a resume would
  redo the whole pipeline into a directory full of stale artifacts, and a
  `--from refine` job would fail its prerequisite check instead of running.
  `find_manifest` is not a convenience; it is what makes the rename safe.
- **`find_manifest` is the only filesystem check in `stages.py`, and it does not
  contradict ADR 0004.** That prohibition is about *stage* state — dirtiness must
  come from the records, never from what happens to be on disk, or a
  `mkdir`-in-advance makes a stage permanently un-dirtyable. Which file the
  records live in is a different question, and no record can answer it.
- **The old file is left in place, and goes stale.** The first run to resume a
  pre-2.0 directory reads `metadata.json`, warns once, and records to
  `pgs-recon.json` from then on. Anything still reading the old file sees a
  manifest frozen at the moment of the upgrade — the sharp edge of this
  decision, and the reason the warning names both files. Deleting or renaming
  the old one instead was considered and rejected: a migration should not
  destroy the only record a downstream consumer might still be reading, and
  `--dry-run` would then have to either lie or mutate.
- **A run's own paths never depended on the manifest's name**, so nothing else
  moves: `<name>_recon_config.txt`, `mvg/`, `mvs/` and every artifact in them are
  exactly where they were.
- **The manifest's own `paths` entry is renamed with the file**: `metadata` ->
  `manifest`, so no key inside `pgs-recon.json` still calls it the thing this ADR
  stopped calling it. Safe to do in the same release because `paths` has no
  programmatic consumer -- it is a record for a human, and the same release had
  already reshaped it (ADR 0005 shrank it to the output layout).
- **The per-tool sidecars are renamed with it**: `<stem>_retexture_metadata.json`
  -> `<stem>_retexture.json` (`apps/retexture.py`) and
  `<name>_calibrate_metadata.json` -> `<name>_calibrate.json`
  (`apps/calibrate.py`), each with its in-file `paths` key following the main
  manifest's `metadata` -> `manifest`. An earlier revision of this ADR left them
  alone as already-unambiguous, on the grounds that moving the main record and
  its satellites in one release was more churn than the tidiness was worth. The
  opposite argument won: they are downstream-visible names, so shipping them in
  the release that documents the manifest rename costs one migration note,
  whereas deferring them means a second such note later for a strictly smaller
  reason. Nothing in this project reads either file, and both are still
  tool-prefixed, so the rename is safe in the same sense the manifest's was not —
  no fallback is needed because nothing locates them by name.
