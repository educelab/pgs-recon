# Migrating to pgs-recon 2.0

For anyone whose code touches a `pgs-recon` output directory or its exit status:
pipeline orchestrators, cluster submit scripts, downstream ingest, and callers
that import `pgs_recon` as a library.

Nothing here changes what a reconstruction *is*. The deliverable and the frame
everything lives in are unchanged; what moved is a set of names, the manifest's
contents, some spellings on the command line, and what a failed run reports. The
one thing that does change pixels is how a CIELab capture is decoded — §6.

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

| Tool | pre-2.0 | 2.0 |
|---|---|---|
| `pgs-retexture` | `<stem>_retexture_metadata.json` | `<stem>_retexture.json` |
| `pgs-calibrate` | `<name>_calibrate_metadata.json` | `<name>_calibrate.json` |

(Neither tool shipped in 1.7; the old names are the 1.8 pre-release's.)

In all three files the `paths` entry pointing at the file itself is now keyed
`manifest` rather than `metadata`.

`pgs-retexture`'s `<stem>` also changed, because `-i` now names a scan directory
rather than one camera's images (§5): in capture mode it is
`<scan-dir>_c<capture>`, so the captures of one scan do not overwrite each other,
and it prefixes every scratch artifact and the default deliverable
(`mvs/<stem>.<file-type>`) as before. That sidecar carries two keys beyond
`args`/`parsed`/`commands`: `capture` and `cameras`, the capture textured from and
the camera indices it resolved to. `parsed` records the *resolved* `capture`, so
replaying the config textures the same one even if the scan later grows another;
the camera list is deliberately only in the manifest, since recording it as if it
had been requested would silently narrow a replay.

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

## 5. CLI changes

### 5a. `pgs-recon`

* **`--no-mvs` is deprecated.** It still stops the run after colorize, but it now
  warns, and `--no-mvs` together with any `--to` other than `colorize` is refused
  (exit 1) rather than silently overriding it. Use `--to colorize`. Bare `--mvs`
  warns and does nothing.
* **`--matching-method ANNL2` is gone.** The choices now mirror the matchers
  `openMVG_main_ComputeMatches` actually builds: `HNSWL2`, `HNSWL1` and
  `HNSWHAMMING` were added, `ANNL2` removed. A script still passing it fails at
  argument parsing (exit 2) instead of inside the matches stage.
* **`--decimation-factor` is now `--refine-decimate`.** Same flag, same
  meaning — `RefineMesh`'s face-fraction decimation of the mesh on the way *in* —
  renamed to match the rest of the `--refine-*` group. It is **deleted, not
  aliased**, so a script still passing the old name fails at argument parsing
  (exit 2). The rename exists because 2.0 also adds `--decimate-max-error`, an
  unrelated *deviation budget* for the new `decimate` stage; two flags both
  saying "decimate", one of them silently meaning another stage's, is the kind of
  confusion that costs a cluster allocation. The library keyword is unchanged
  (`mvs_refine(decimate=...)`, §7).
* **`--decimate-max-error` / `--decimate-max-faces` / `--decimate-prefer` are
  new**, and add a fourteenth stage, `decimate`, between `refine` and `texture`
  ([ADR 0008](./adr/0008-error-bounded-decimation.md)). The stage merges
  triangles to make the deliverable smaller — a finished mesh is millions of
  faces, more than the geometry justifies and more than MeshLab opens
  comfortably — and stops while the result is still within
  `--decimate-max-error` of the mesh it started from: the largest distance any
  point of either surface may end up from the other, in the solved scene's
  units. Unlike the three OpenMVS decimations, which take a face fraction and
  state no geometric bound, this one **measures** the deviation on each
  candidate instead of predicting it, so sharp edges and ridges survive and the
  guarantee is a number rather than a hope. It writes
  `mvs/decimate_report.json` beside the mesh with the faces in and out, the
  measured deviation, and which bound stopped it. `--decimate-max-faces` budgets
  faces instead, for a scan that was never scaled to physical units, and
  `--decimate-prefer` says which wins when both are given. The same tool is
  available standalone as `pgs-decimate`, on any mesh. Stating a budget is what
  enables the stage; a budget of `0` disables it on a resume, which is the only
  way to, since arguments are inherited from the manifest — and because `0`
  turns off the *stage*, it drops an inherited second budget with it. Nothing
  changes for a run that states no budget. Two further flags,
  `--decimate-quadric-seed` and `--decimate-min-gain`, tune what the search
  spends getting there and never what it guarantees; both are optional and
  neither enables the stage.
* **`--import-capture n` is new** (`pgs-import --capture/-C` is the same choice for
  the standalone importer). A PGS scan holds every capture position once per
  *capture*, each with its own lighting and camera set; 1.7 hardcoded capture 0
  and warned once per file it skipped. The default is still 0, so no existing
  command changes meaning — but a config file written by 2.0 now carries
  `import-capture`, and the flag warns and is ignored without
  `--import-pgs-scan`.

### 5b. `pgs-retexture` (capture mode only; `--calibration` mode is untouched)

If you drove the 1.8 pre-release's `pgs-retexture`, its default mode changed shape:

| | pre-2.0 | 2.0 |
|---|---|---|
| `-i` | a directory of one camera's modality images | the **PGS scan directory** (needs its `metadata.json`) |
| capture | whatever the directory held | `--capture n`, inferred only when the scan holds one |
| cameras | one, `--camera-index k` or inferred | **every** camera the capture and the solve share; `--camera-index k [k ...]` now *restricts* that set |
| stem | the input directory's name | `<scan-dir>_c<capture>` (§1a) |

`scan.file_prefix` and `scan.format` from the metadata are what select the images,
so two files for one `(camera, position)` in different formats or captures can no
longer be confused for each other. The two image sets need not agree: a solved
view with no modality image is skipped quietly, a modality image with no solved
view is warned about loudly, and a camera `--camera-index` asks for that the
capture lacks is warned about and skipped. Only an empty result is fatal. An
existing artifact path is still overwritten, but now every one of them is listed
in a warning first.

### 5c. `pgs-remove-ground-plane`

1.7 fitted a *plane* and always kept the single largest connected component.
Neither holds in 2.0, so a script that relied on the defaults gets a different
mesh out:

* **The ground is a polynomial surface now, and the default
  `--distance-threshold` is `0.02` rather than `0.1`.** A scan bed is bowed by
  far more than the mesh's own noise, so a plane's inlier band only ever caught
  the strip where the two happened to coincide and left most of the bed
  standing. 2.0 fits a degree-`--surface-degree` (default 2) height field in
  the plane's frame instead and refits it until the inlier set settles. The run
  prints the bed's warp and the fit's rms as a fraction of the threshold, which
  is how you judge whether a threshold suits a capture — a healthy fit lands
  around a tenth to a quarter of it. `--surface-degree 0` asks for 1.7's plane
  back, but a fit whose residuals *fill* the band rather than hugging it, which
  is what a plane over a bowed bed does, is now **refused** (exit 1, nothing
  written) rather than handed on half-removed.
* **Component filtering is a choice.** `--filter-cc largest` is the default and
  is 1.7's behaviour; `--filter-cc none` keeps everything, `--filter-cc N`
  drops components under N faces, and `--filter-cc-area AREA` (mutually
  exclusive with `--filter-cc`) drops those under AREA in the mesh's units
  squared — cm² on an autoscaled reconstruction, and the one measure that
  means the same thing from one scan to the next. An area of `0` reports the
  component inventory without dropping anything, and every filter now prints
  what it kept and dropped.
* **`--drop-below-ground` is new**, and off unless asked for. The scan bed's
  fiducial squares reconstruct as shallow recesses *under* the bed, which
  ground removal cannot take — they are below its band, not inside it — so they
  reach the delivered mesh as 27-29 components per scan. No area threshold
  reaches them: 20-22 of those measured 0.51-2.84 cm², overlapping the real
  fragments the 0.5 cm² floor exists to keep, and only the remainder was
  speckle an area filter would have caught anyway. The flag drops any component
  whose *highest* vertex is below the fitted surface, which separates them
  completely — on three measured captures every island topped out at -0.02
  while the artifact reached +2.5 to +3.0 — and adds no threshold of its own,
  because ground removal has already taken everything within the distance
  threshold of the surface. It composes with the `--filter-cc*` filters rather
  than replacing them; speckle and sub-bed islands are different things. It is
  worth more than tidiness downstream: on one capture the islands inflated the
  kept geometry's bounding box by 1.45x in area, and an orthographic sampling
  frame derived from that bbox spends the difference on empty bed.
* **`--seed` is new** (default 0), so the RANSAC that orients the fit is
  reproducible from one run to the next.

## 6. CIELab captures decode differently, and ImageMagick is gone

`pgs-convert`, `pgs-calibrate` and `pgs-retexture` now read pixels through one
function (`pgs_recon.utils.images.read_srgb`). Nothing shells out for image data,
and **`imagemagick` is no longer installed in the Docker/Apptainer images** — a job
that called `convert`/`magick` inside one of our containers has to bring its own.
`tifffile` is a new Python requirement (a `pip install .` picks it up; a pinned or
vendored environment needs adding to).

What changes on disk is confined to CIELab TIFFs:

* **Their colors change, and were wrong before.** `pgs-convert` used to rescale a
  Lab file's samples as if they were RGB; `pgs-calibrate`/`pgs-retexture` shelled
  out to ImageMagick, which ignores the `WhitePoint` tag and decodes every Lab
  file as D65 — the EduceLab captures are untagged, i.e. TIFF's D50. 2.0 decodes
  against the file's own white point. A texture or converted image derived from a
  Lab capture will not match what any pre-2.0 version produced, and a re-run is
  the only way to get the corrected colors.
* **Everything else is bit-for-bit what it was**: a 16-bit greyscale still becomes
  `round(v / 257)`, an 8-bit RGB still passes through unchanged (both verified
  against ImageMagick in `tests/test_images.py`).
* **`pgs-convert --if-same-type copy|skip` no longer passes a Lab file through.**
  OpenMVG reads sRGB, so no Lab file may reach the output undecoded, which
  disqualifies both whole-dataset shortcuts for a scan holding any — including
  `skip`, which used to exit 0 having written nothing. The colorspace is decided
  per file (a set can mix them), so only the Lab images are converted: with
  `copy`, the rest of the tree (sidecars included) is still copied wholesale; with
  `skip`, the non-Lab images are copied byte-for-byte rather than re-encoded. A
  run with `--filter-cam/-pos/-cap` stays file-by-file either way, so it never
  brings back what it excluded. Cost on an all-sRGB scan is one header read per
  file, threaded.

One build note in the same area: OpenMVG bundles zlib 1.2.3 when either libpng or
libtiff development headers are missing, which aborted a reconstruction with
`libpng error: bad parameters to zlib` the first time it read a mask. A
from-source build now fails *configuration* on that mismatch instead, and the
images install both.

## 7. Library callers

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

`pgs_recon.utils.geometry` carries the ground work behind
`pgs-remove-ground-plane` (§5c). `segment_plane` is still there, but
`segment_ground_surface` is what the app calls, returning a `GroundSurface` and
its inlier vertex indices; `remove_connected_components_below_surface` and
`remove_connected_components_by_area` are the new filters, and every filter
returns a `ComponentInventory` of what it kept and dropped rather than nothing.
A `GroundSurface`'s frame is **oriented**: `signed_distance` is height above the
ground, positive away from it. If you wrote against a 2.0 alpha, that sign used
to fall out of the SVD unconstrained and came out inverted on roughly half of
the captures measured — code that took `abs()` of it to compensate should stop.

Two smaller surfaces moved with the capture work: `pgs_data.import_pgs_scan` and
`pgs_data.init_sfm_pgs` take a `capture` keyword (default 0, the capture 1.7
hardcoded), and the PGS filename convention is parsed by
`pgs_recon.utils.scan_names` (`parse_scan_name`, `parse_view_name`) rather than by
a regex private to each app.

## Checklist for an orchestrator

- [ ] Read `pgs-recon.json`, falling back to `metadata.json`; don't read the old
      one once the new one exists.
- [ ] Drop `_metadata` from the two sidecar names: `<stem>_retexture.json`,
      `<name>_calibrate.json`. No fallback here — the old names are simply gone.
      A capture retexture's `<stem>` now ends in `_c<capture>`.
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
- [ ] Point `pgs-retexture -i` at the PGS scan directory rather than one camera's
      images, pass `--capture` when the scan holds several, and expect every
      shared camera to be textured from unless `--camera-index` narrows it.
- [ ] Pass `--import-capture n` if a run should solve from a capture other than 0
      (the default is unchanged).
- [ ] Rename `--decimation-factor` to `--refine-decimate`; the old spelling is
      gone and fails at argument parsing.
- [ ] Expect `decimate` in `--from`/`--to`'s stage list, and in a manifest's
      `stages` and `shape` whenever `--decimate-max-error`/`--decimate-max-faces`
      was given. Its `deviation` output is `mvs/decimate_report.json`; read the
      measured deviation there rather than from the log.
- [ ] Expect a polynomial ground fit and a `0.02` default `--distance-threshold`
      from `pgs-remove-ground-plane`, and pass `--drop-below-ground` to clear
      the bed's fiducial islands. It is opt-in: without it a run delivers what
      it did before, islands included.
- [ ] Stop calling ImageMagick inside our images, and expect Lab-derived textures
      and conversions to differ from 1.7's — they were miscolored.
- [ ] Keep shape flags (`--mvs-densify`, `--mvg-robust`, `--mvg-autoscale`,
      `--mvs-refine`) on the *first* job of a staged run. Adding one later no
      longer renames anything, but it still moves what the mesh stages consume,
      so they re-run.
