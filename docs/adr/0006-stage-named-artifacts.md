# Name artifacts for the stage that produced them

**Status: proposed — not yet implemented.** Describes MR3 of the issue #17
series; see [the plan](../wrapper-refactor-plan.md). Until it lands, names chain
off the input's stem exactly as described under Context, so an output directory on
disk today still contains `scene_dense_refine.ply` and friends.

## Context

Intermediate filenames were derived by each wrapper appending its own suffix to
its input's stem, so a name accumulated the pipeline's history:
`scene.mvs` → `scene_dense.mvs` → `scene_dense_mesh.ply` →
`scene_dense_refine.ply`. Nobody chose that vocabulary; it fell out of fourteen
wrapper bodies each doing `in_path.stem + '_<tag>'`.

Two costs. Names encode the **pipeline shape**, so adding a stage renames
everything downstream of it — `apptainer/submit_recon_pipeline.sh` documents an
operational constraint that exists only because of this (*"`--mvs-densify`
belongs HERE… densify renames the whole mesh chain (`scene_mesh.ply` →
`scene_dense_mesh.ply`), so adding it on a later job would invalidate stages that
job is not sized to rebuild"*). And the names actively mislead about what a file
*is*: `scene_dense_refine.ply` is a mesh, named from the scene it was refined
against, which is exactly the confusion
[ADR 0003](./0003-portable-mvs-intermediates.md) has to warn about (*"the `-i`
given to refine/texture must be an interface scene, never a mesh-bearing one"*).

The names matter to whoever is reading a half-finished output directory trying to
work out which stage produced what.

## Decision

Name an intermediate `<stage>_<role>.<ext>` — the stage that produced it and the
role it fills. Naming lives in `layout.py`: pure functions, no filesystem, unit
tested.

| Stage | Today | After |
|---|---|---|
| robust | `sfm_data_structured.bin` | `robust_sfm.bin` |
| autoscale | `sfm_data_structured_scaled.bin` | `autoscale_sfm.bin` |
| colorize | `sfm_data_structured_scaled_colorized.ply` | `colorize_sfm.ply` |
| convert | `scene.mvs` | `convert_scene.mvs` |
| densify | `scene_dense.mvs` / `scene_dense.ply` | `densify_scene.mvs` / `densify_cloud.ply` |
| reconstruct | `scene_dense_mesh.ply` | `reconstruct_mesh.ply` |
| refine | `scene_dense_refine.ply` | `refine_mesh.ply` |

**The convention replaces name chaining, so it applies exactly where chaining
occurred.** A name already independent of the shape keeps it: every name a binary
picks for itself (`mvg/sfm_data.json`, `recon_dir/sfm_data.bin`,
`sfm_data_expanded.json`), the already-role-named `matches.bin` /
`matches_filtered.bin` / `matches_dir/`, `pgs-global-scaler`'s
`landmarks[_scaled].ply`, and the deliverable `mvs/<name>.<ext>`.

**Stage-prefixed rather than bare role names** (`mesh.ply`) because roles are
*rebound*: `reconstruct` and `refine` both produce `mesh`, so bare names would
collide and the later stage would clobber the artifact the earlier one wrote —
destroying the per-stage inspectability this exists for, and making a rebinding
invisible to the planner.

**The enabling invariant: `layout` *writes* names; the manifest *locates*
artifacts.** A resumed job never rebuilds a name to find something — bindings
come from the recorded `stages.*.outputs` verbatim (`StageTracker._absorb`), so a
completed stage's filename is never recomputed. This is what makes renaming safe
on existing output directories, and it is the thing to not break.

## Consequences / non-obvious traps

- **Never reconstruct a name to find an artifact, and do not add existence checks
  to compensate.** Both halves matter: rebuilding a name would make this rename
  (and any future one) silently invalidate every output directory, and
  [ADR 0004](./0004-staged-resumable-runs.md) separately forbids `path.exists()`
  because `main()` creates `mvg/matches_dir/` before any check could run, which
  would make `features` permanently un-dirtyable.
- **Clause 4 now catches the direct rebinding and clause 3 propagates it.**
  Previously a shape change renamed everything downstream, so clause 4 ("a
  consumed role is bound to a different path than recorded") fired all the way
  down. Now `refine_mesh.ply` is `refine_mesh.ply` regardless of shape, so
  texture's input no longer *moves* and clause 3 ("a producer it consumes is
  dirty") carries the cascade. Both clauses remain necessary — ADR 0004's
  argument is unchanged — but the division of labour shifted. Two different
  producers can never share a name under this scheme, which is what keeps a
  rebinding always visible to clause 4.
- **Legacy output directories stay resumable, and a verbatim re-run is still a
  no-op.** Completed stages bind from their records, so nothing goes dirty. The
  first stage that *does* run writes the new name and cascades downstream —
  identical to what already happens when that stage re-runs for any other reason,
  since clause 3 fires regardless. The cost is **orphaned old-named files** left
  in `mvg/`/`mvs/`, which `--rerun` already produced.
- **`mvs/<name>.<ext>` is frozen because it is a user-facing contract.** ADR 0004
  depends on it (*"`mvs_texture` always writes `mvs/<name>.<ext>`: a stale final
  mesh sits at exactly the expected filename"*) and
  `recon_dir._mesh_from_commands` reconstructs it from `name` + `file_type` for
  pre-stage-record manifests.
- **`mvg/sfm_data.json` is frozen because one of three importers dictates it.**
  `init_sfm_generic` calls OpenMVG's `SfMInit_ImageListing` with `-o <dir>`, so
  OpenMVG picks the filename; the two Python importers could be renamed, but then
  the import artifact's name would depend on `--new-importer`/`-p`.
- **The submit script's constraint is relaxed, but its advice still stands.**
  Adding `--mvs-densify` on a later job no longer renames the downstream chain,
  but it still moves `reconstruct`'s input, so the cascade still re-runs stages
  that job may not be sized for. Keep shape flags on job 1.
- **This lands as its own MR, separately from
  [ADR 0005](./0005-wrappers-mirror-the-binary.md)'s interface change**, which
  moves naming into `layout.py` while reproducing the *old* names exactly. Split
  on the risk axis on purpose: the interface change is a provable no-op verified
  by a byte-identical artifact tree, and the whole behavioural risk of the rename
  is confined to one file's return values.
