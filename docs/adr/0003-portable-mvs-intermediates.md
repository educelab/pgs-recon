# Keep every MVS intermediate in a Boost-independent format

## Context

A `TextureMesh` run consumed >128 GB of RAM and was OOM-killed while loading a
`.mvs` that had loaded fine in the environment that produced it.

Cause: OpenMVS's default archive type writes a **Boost binary serialization
archive** (`ARCHIVE_BINARY_ZSTD`), deserialized with the `no_header` flag
(`libs/Common/Types.inl`, `SerializeLoad`). Boost's binary format is explicitly
non-portable across Boost versions, and `no_header` strips even the archive's own
signature bytes, so there is no negotiation and no validation. A reader linked
against a different Boost misreads a container length and immediately `resize()`s
a multi-hundred-GB buffer. Because that allocation is satisfiable it never throws
`bad_alloc`, so the `try/catch` around the load does not fire — the process just
balloons and is OOM-killed **during scene load, before any processing**, with no
error message.

The trigger was the container base bump `ubuntu:20.04` → `ubuntu:22.04`
(2024-06-26), i.e. Boost 1.71 → 1.74. Architecture and OS were ruled out by
direct testing: an x86_64 build OOM-killed on the same file that an aarch64 build
did, and the "CentOS" in the logs is the LCC host kernel, not the container
userspace.

The alternative — matching Boost versions across every environment that touches a
run's artifacts — is not viable when jobs move between container generations and
machines, and it is precisely what staged runs ([ADR 0004](./0004-staged-resumable-runs.md))
require.

## Decision

Run the MVS pipeline so **no Boost archive is ever written**. Every scene stays
in the Boost-independent `MVSI` interface format (fixed-width types, its own
versioned header); all geometry travels as `.ply`.

- `--archive-type -1` on all four builders (`mvs_densify`, `mvs_reconstruct`,
  `mvs_refine`, `mvs_texture`).
- The `MVSI` scene from `openMVG2openMVS` (`convert_scene.mvs`) is carried
  forward as `-i` to every stage; meshes and clouds move by `-m`/`-o`/`-p` as
  `.ply`.

Verified end-to-end on OpenMVS v2.4.0 (arm64 / Boost 1.74): densify →
reconstruct → texture produced only `MVSI` scenes plus PLY/PNG, no `MVS\0` file.

## Consequences / non-obvious traps

- **The explicit `-1` is not redundant — do not remove it.** It is already
  OpenMVS's default; it is passed explicitly so an upstream default change cannot
  silently reintroduce Boost binaries.
- **`-1` alone does not make a mesh-bearing scene portable.** `Scene::Save`
  *falls back to Boost binary* when a mesh is embedded. What actually saves us is
  the mesh stages' guard — `if (nArchiveType != ARCHIVE_MVS || sceneType !=
  SCENE_INTERFACE) scene.Save(...)` — so with `-1` **and an interface input
  scene** they skip writing a scene `.mvs` at all and emit only the mesh. The
  `-i` given to refine/texture must therefore be an interface scene
  (`convert_scene.mvs` / `densify.mvs`), never a mesh-bearing one.
- **`-p` is mandatory when densifying.** Under `-1`, `DensifyPointCloud` writes
  the *dense* cloud to `densify.ply` (carrying `view_indices`/`view_weights`,
  which round-trip through `PointCloud::Save`/`Load`) and leaves `densify.mvs`
  holding only the *sparse* cloud. Omit `-p` and `ReconstructMesh` silently
  builds from the sparse cloud. OpenMVS does auto-derive `<input>.ply` under
  `-1`, but it is passed explicitly so the dense cloud can never be silently
  dropped. The pair shares a stem because both names come off the one `-o`,
  which is why densify is the stage
  [ADR 0006](./0006-stage-named-artifacts.md) leaves unable to name its roles.
- **Version-dependent.** Older builds embedded the dense cloud in a Boost-binary
  `.mvs` even with `-1` (the origin of a legacy 7.6 GB `scene_dense.mvs`, as
  the dense scene was then called).
  Re-verify the `-1` behavior whenever the OpenMVS version changes.
- **Refine's output used to be named from the scene, not the mesh.**
  `mvs_refine` built `in_path.stem + '_refine.ply'` from the `-i` scene, so with
  densify the result was `scene_dense_refine.ply`, not
  `scene_dense_mesh_refine.ply`; directories predating the 2024-06 dependency
  update threaded the mesh-named scene forward and so contain the latter. Both
  are history: [ADR 0006](./0006-stage-named-artifacts.md) names it
  `refine_mesh.ply`, from the stage rather than from any input.
- **Legacy Boost-binary artifacts stay unreadable** by a differently-linked
  build; they are not recoverable in the current container. Load them once in a
  Boost-matched build (20.04 → 1.71, 22.04 → 1.74, 24.04 → 1.83) and re-export:
  `TransformScene -i in.mvs -o out.txt.mvs -t identity.txt --archive-type 0`
  (TransformScene needs an operation, hence the identity transform), or export
  the mesh to `.ply` and thereafter texture with
  `TextureMesh -i <scene>.mvs -m mesh.ply` anywhere.
- **Isolation caveat.** The failing readers also ran a different OpenMVS version
  than the writer, so the Boost delta was never isolated from a possible OpenMVS
  serialization change. The Boost gap is sufficient on its own and is the
  documented failure mode; isolating it would need a Boost-1.71 build of the
  *same* OpenMVS version.
- **Upstream fixes, not local patches.** Either stamp `BOOST_VERSION` and type
  sizes into the reserved bytes of OpenMVS's own `.mvs` header and reject on
  mismatch (turns the silent OOM into an actionable error), or extend the
  already-portable `Interface.h` serializer to carry the mesh and texture so
  `-1` becomes fully portable for all scenes.
