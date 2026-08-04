# Migrating to pgs-recon 2.0

For anyone whose code touches a `pgs-recon` output directory or its exit status:
pipeline orchestrators, cluster submit scripts, downstream ingest, and callers
that import `pgs_recon` as a library.

Nothing here changes what a reconstruction *is*. The deliverable and the frame
everything lives in are unchanged; what moved is a set of names, the manifest's
contents, two spellings on the command line, and what a failed run reports.

**A 1.7 output directory cannot be resumed, only rebuilt.** 2.0 reads its
manifest, but a 1.7 manifest records no per-stage state, so there is nothing to
resume from — see §1. Directories built by the 1.8 pre-release
(`v1.8.0-alpha.1`, the only 1.8 ever tagged — the rest of that line's work ships
here instead) do resume, untouched: they carry stage records, so their 1.7-style
intermediate names are read back from the manifest rather than rebuilt.

## 1. The manifest: renamed, and larger than it was

### 1a. New name

`<output>/metadata.json` → `<output>/pgs-recon.json`
([ADR 0007](./adr/0007-name-the-manifest-for-the-tool.md)). `metadata.json` was
also what an EduceLab *scan* directory calls its descriptor, which is an input
format we do not own, so the one name meant two unrelated things.

* **Reading a finished run:** look for `pgs-recon.json`, fall back to
  `metadata.json`. Both may be present.
* **`pgs-recon` reads whichever name is there** and records to `pgs-recon.json`
  from then on, warning once.
* **The old file is left behind and goes stale.** After a resumed run there may
  be two manifests, and `metadata.json` is frozen at the moment of the upgrade.
  If you read it, you will silently get pre-upgrade state — this is the one
  thing in 2.0 that can mislead rather than fail.

```python
manifest = next((p for p in (recon / 'pgs-recon.json', recon / 'metadata.json')
                 if p.is_file()), None)
```

The other two tools' sidecars move the same way, and unlike the manifest they get
no fallback — nothing locates them by name, so there is nothing to fall back
*for*:

| Tool | 1.7 | 2.0 |
|---|---|---|
| `pgs-retexture` | `<stem>_retexture_metadata.json` | `<stem>_retexture.json` |
| `pgs-calibrate` | `<name>_calibrate_metadata.json` | `<name>_calibrate.json` |

In all three files the `paths` entry pointing at the file itself is now keyed
`manifest` rather than `metadata`.

### 1b. New keys

2.0 added staged, resumable runs
([ADR 0004](./adr/0004-staged-resumable-runs.md)), and with them four top-level
keys. The change is **additive**: every key 1.7 wrote is still written, with the
same meaning, and `commands` keeps its `timestamp → space-joined string` format
(which is why `recon_dir.py` can still recover the solved SfM from a 1.7
manifest by grepping it for `openMVG2openMVS`).

| Key | 1.7 | 2.0 |
|---|---|---|
| `args`, `parsed`, `commands` | ✔ | ✔ unchanged |
| `paths` | ✔ | ✔ **different contents** — see below |
| `stages` | — | per-stage `status`, `inputs`, `outputs`, `args`, timings, `max_rss` |
| `effective_args` | — | the arguments a resumed job inherits |
| `shape` | — | the stages this reconstruction consists of |
| `runs` | — | one entry per invocation: `argv`, `started`, `host`, `range` |

`stages` is the resume record and the supported way to locate an artifact.

**`paths` is no longer a lookup table for artifacts.** In 1.7 it carried the
binaries (`BIN`, `MVS_BIN`, `PATH`, `CAM_DB`) and individual artifacts (`sfm`,
`matches_file`, `mvs_scene`, `view_pairs`); in 2.0 it is the output layout only
(`output`, `mvg`, `matches_dir`, `recon_dir`, `mvs`, `undistorted_images` — 1.7's
`mvs_images` — plus `config`, `manifest` — 1.7's `metadata` — and
`input`/`input_calib` when set). The binaries left because they are resolved per
invocation now (§4); the artifacts left because `stages` records them properly.
Nothing in this project reads `paths` back; it is a record for a human.

### 1c. What a 1.7 directory does when 2.0 runs against it

* Its manifest is found and read — nothing crashes, and `pgs-retexture` /
  `pgs-calibrate` still resolve the mesh and SfM out of it, from the recorded
  `openMVG2openMVS` command and `parsed`. (`pgs-quality-check` reads a PGS *scan*
  directory's `metadata.json`, never a run's manifest, so none of this touches
  it.)
* **It has no `stages`, so every stage is dirty and the whole pipeline re-runs**,
  writing the new artifact names alongside the old ones. `--from` will fail its
  prerequisite check rather than start mid-pipeline.
* **It has no `effective_args`, so nothing is inherited**: `-i` and `--name` have
  to be supplied again, as on a first run.

If that matters, finish or discard 1.7 runs before upgrading; there is no
converter, and inventing stage records for work we cannot verify would be worse
than re-running it.

## 2. Intermediate artifacts are named for the stage that produced them

An intermediate is now `<stage>_<role>.<ext>`, so no filename encodes which
*other* stages ran ([ADR 0006](./adr/0006-stage-named-artifacts.md)):

| Stage | 1.7 | 2.0 |
|---|---|---|
| robust | `mvg/recon_dir/sfm_data_structured.bin` | `mvg/recon_dir/robust_sfm.bin` |
| autoscale | `mvg/recon_dir/sfm_data_structured_scaled.bin` | `mvg/recon_dir/autoscale_sfm.bin` |
| colorize | `mvg/recon_dir/sfm_data_..._colorized.ply` | `mvg/recon_dir/colorize_sfm.ply` |
| convert | `mvs/scene.mvs` | `mvs/convert_scene.mvs` |
| densify | `mvs/scene_dense.mvs` / `.ply` | `mvs/densify.mvs` / `mvs/densify.ply` |
| reconstruct | `mvs/scene_dense_mesh.ply` | `mvs/reconstruct_mesh.ply` |
| refine | `mvs/scene_dense_refine.ply` | `mvs/refine_mesh.ply` |

The 1.7 column depended on the shape, because each name chained off its input's
stem: with `--mvs-densify` off, reconstruct wrote `scene_mesh.ply`; with
`--mvg-robust` off, autoscale wrote `sfm_data_scaled.bin` and colorize
`sfm_data_colorized.ply`. The 2.0 column does not: these are the names for every
shape, which is the point of the change.

`densify` is the exception to `<stage>_<role>`: OpenMVS takes one `-o` and
derives both files from its stem, so the pair is `densify.mvs` / `densify.ply`.

**Unchanged** (deliberately — these were never shape-dependent):
`mvs/<name>.<ext>` (the deliverable), `mvg/sfm_data.json`,
`mvg/recon_dir/sfm_data.bin`, `mvg/matches_dir/` and its `matches[_filtered].bin`,
`mvg/recon_dir/landmarks[_scaled].ply`, `mvs/undistorted_images/`.

**The config file lost its timestamp.** Given `--name`, 1.7 wrote
`<datetime>_<name>_recon_config.txt`, a fresh file per invocation; 2.0 always
writes `<name>_recon_config.txt`, one per reconstruction, because several jobs now
share the directory. (1.7 wrote the un-prefixed name only when `--name` was
*omitted*, since the derived name already started with a timestamp.) A glob for
`*_recon_config.txt` still finds it; a path assembled from a job's own timestamp
does not.

One behavioural note: with `--mvg-recon-method direct`, the solve now lands at
`mvg/recon_dir/sfm_data.bin` — the same path every other engine writes — rather
than borrowing the robust stage's name.

**If you locate artifacts by name, stop.** Every path is recorded in the
manifest, relative to the output directory, and that is the supported way to
find one:

```python
meta = json.loads(manifest.read_text())
mesh = recon / meta['stages']['texture']['outputs']['mesh']
scene = recon / meta['stages']['convert']['inputs']['sfm']   # the solved SfM
```

For the two artifacts downstream tools ask for most, `pgs_recon.utils.recon_dir`
already does this (including for pre-stage-record manifests):

```python
from pgs_recon.utils.recon_dir import resolve_solved_sfm, resolve_textured_mesh
mesh = resolve_textured_mesh(recon).require()
sfm = resolve_solved_sfm(recon).require()
```

## 3. A failed run's exit code is now the child's

`run_command` used to `sys.exit(str)`, so **every** failure exited 1. It now
raises `ToolFailed`, and each `main()` translates it:

| Situation | 1.7 | 2.0 |
|---|---|---|
| Binary exited non-zero | 1 | that exit code |
| Binary killed by a signal | 1 | `128 + signum` (an OOM-killed `RefineMesh` is **137**) |
| Binary not found / not executable | 1 | 127 |

`rc != 0` handling is unaffected. Code that special-cases `rc == 1`, or that
infers "the tool failed" from exactly 1, needs updating — and a Slurm job that
was being retried blindly can now distinguish an OOM (137) from a bad argument.

## 4. Install prefix: `$PGS_RECON_PREFIX`

Binaries are resolved at call time from, in order: `--path`, `$PGS_RECON_PREFIX`,
then `/usr/local/`. `--path` no longer carries a `/usr/local/` default of its
own, which is what makes the environment tier reachable. A missing binary now
names both the path searched and which tier chose the prefix.

Two consequences for a runner:

* Set `PGS_RECON_PREFIX` once in the job environment instead of threading
  `--path` through every invocation. Inside our images neither is needed. `--path`
  is deliberately *not* inherited by a resumed job — a staged run's refine node
  may have a different prefix than the node that ran SfM — so if you do use it,
  pass it to every job.
* A `*_recon_config.txt` written by 2.0 omits arguments that were never set,
  where 1.7 wrote `path = /usr/local/`.

Omitting the unset arguments is also what makes the file loadable at all. **A
1.7 config cannot be fed back in with `-c`** — not to 2.0 and not to 1.7 either:
it spells every unset argument as the literal `None` (`import-pgs-scan = None`,
`focal-length = None`), and the parser rejects the first one it reaches
(`error: Unexpected value for import-pgs-scan: 'None'`). Re-run from the original
command line, or strip the `None` lines. A config written by 2.0 round-trips.

## 5. Two CLI changes

* **`--no-mvs` is deprecated.** It still stops the run after colorize, but it now
  warns, and `--no-mvs` together with any `--to` other than `colorize` is refused
  (exit 1) rather than silently overriding it. Use `--to colorize`. Bare `--mvs`
  warns and does nothing.
* **`--matching-method ANNL2` is gone.** The choices now mirror the matchers
  `openMVG_main_ComputeMatches` actually builds: `HNSWL2`, `HNSWL1` and
  `HNSWHAMMING` were added, `ANNL2` removed. A script still passing it fails at
  argument parsing (exit 2) instead of inside the matches stage.

## 6. Library callers

If you `import pgs_recon`, the wrapper interface changed wholesale in 2.0
([ADR 0005](./adr/0005-wrappers-mirror-the-binary.md)). The wrappers in
`pgs_recon.openmvg` / `pgs_recon.openmvs` took a `paths` dict plus `*_key`
strings and returned the key(s) they had inserted into it; they now take and
return `Path`s, one function per binary, with every flag reachable:

```python
# 1.7
paths = {'scene_key': ..., 'MVS_BIN': ..., 'mvs': ...}
scene_key, cloud_key = mvs_densify(paths, 'scene_key', metadata=meta,
                                   resolution_lvl=2)

# 2.0
from pgs_recon import toolchain
toolchain.configure(prefix=prefix, recorder=recorder)   # once, per process
mvs_densify(scene, output=out, resolution_level=2)
```

Flags now mirror the binary's own names: `mask_value` → `ignore_mask_label`,
`marker_pix` → `min_marker_pix`, `decimation_factor` → `decimate`, `free_space`
→ `free_space_support`, `resolution_lvl` → `resolution_level`, `max_size` →
`max_texture_size`, `file_format` → `export_type`. Output naming lives in
`pgs_recon.layout`, whose functions take the output root and nothing else.
`mvs_reconstruct`/`mvs_refine` no longer return their pass-through scene.

## Checklist for an orchestrator

- [ ] Read `pgs-recon.json`, falling back to `metadata.json`; don't read the old
      one once the new one exists.
- [ ] Drop `_metadata` from the two sidecar names: `<stem>_retexture.json`,
      `<name>_calibrate.json`. No fallback here — the old names are simply gone.
- [ ] Expect `stages`, `effective_args`, `shape` and `runs` in the manifest, and
      a `paths` that no longer names artifacts or binaries.
- [ ] Replace any hardcoded intermediate filename with a manifest lookup.
- [ ] Keep treating `mvs/<name>.<ext>` as the deliverable — it has not moved.
- [ ] Stop treating exit 1 as the only failure; expect 137 for OOM, 127 for a
      missing binary.
- [ ] Set `$PGS_RECON_PREFIX` if you run outside our images.
- [ ] Stop re-loading 1.7 `*_recon_config.txt` files with `-c`, and expect the
      file itself at `<name>_recon_config.txt` — no timestamp prefix, one per
      reconstruction.
- [ ] Replace `--no-mvs` with `--to colorize`, and `--matching-method ANNL2`
      with a matcher the binary builds (`HNSWL2`, `FASTCASCADEHASHINGL2`, …).
- [ ] Keep shape flags (`--mvs-densify`, `--mvg-robust`, `--mvg-autoscale`,
      `--mvs-refine`) on the *first* job of a staged run. Adding one later no
      longer renames anything, but it still moves what the mesh stages consume,
      so they re-run.
