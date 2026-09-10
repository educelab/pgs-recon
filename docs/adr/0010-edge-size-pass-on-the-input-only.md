# Force refine's edge-size pass on the input only

**Status: accepted, unvalidated.** The mechanism is read off the pinned OpenMVS
source and the arithmetic is confirmed against a real run's log line; the cost
this removes is **not** measured, and neither is the refined output. What is
recorded below is what the source does, why the coarsen stage cannot avoid it,
and what would settle the size of it. See *Not yet measured*.

Amends [ADR 0009](./0009-coarsen-before-refine.md) in one respect: Decision 4
established that `coarsen` must hand `RefineMesh` `--decimate 1
--ensure-edge-size 2`, because the binary guards the edge-size pass on its own
decimation and "force" is the only value that still reaches it. That is still
true and unchanged. What Decision 4 did not say is that "force" also reaches the
pass at every *later* image scale, where nothing else in the pipeline — and no
stock OpenMVS run — puts it. This ADR carries an OpenMVS patch scoping it back to
the input mesh, and the flags in `stages.REFINE_WITH_COARSEN` do not change.

## Context

Reported by a maintainer running the first refine jobs after `coarsen` landed:
the first iteration proceeded quickly, then the job sat for over an hour in a
silent step between iterations. Nothing in `RefineMesh`'s log names that step.

`Scene::RefineMesh` is a coarse-to-fine loop over image scales — `--scales`,
which its own help text calls "iterations", and which `pgs-recon` sets to 3, so
image scales 0.25, 0.5, 1.0. The first thing each pass does, before any
optimization, is resample the mesh (`SceneRefine.cpp:1303-1315` at the pinned
`ca991d5`):

```cpp
for (unsigned nScale=0; nScale<nScales; ++nScale) {
    const Real scale(POWI(fScaleStep, nScales-nScale-1));
    refine.InitImages(scale, ...);
    refine.ListVertexFacesPre();
    refine.SubdivideMesh(nMaxFaceArea, nScale == 0 ? fDecimateMesh : 1.f, nCloseHoles, nEnsureEdgeSize);
```

`SubdivideMesh` (`SceneRefine.cpp:486`) reprojects every face into every image
pair, splits every face whose projected area exceeds `--max-face-area` (16 px²),
and then runs `Mesh::EnsureEdgeSize` followed by `Mesh::Clean(1.f, ...)`.

**`EnsureEdgeSize` is a full isotropic remeshing.** `Mesh.cpp:1861` imports the
mesh into a CGAL `Surface_mesh` and calls `PMP::isotropic_remeshing` — split,
collapse, flip, relocate over every face — for **10 iterations**
(`Mesh.h:187` defaults `max_iters=50`, clamped by `MINF(max_iters, 10)`).
`Clean` behind it is another CGAL import, hole-close and export. Both are
single-threaded, and both log only on completion (`DEBUG`, one line each, after
the fact), which is what makes a long one indistinguishable from a hang.

**Where it runs is decided by one guard** (`SceneRefine.cpp:556`):

```cpp
if ((nEnsureEdgeSize == 1 && !bNoDecimation) || nEnsureEdgeSize > 1)
```

`fDecimateMesh` is passed only on the first scale; every later scale gets `1.f`,
which sets `bNoDecimation`. So:

| `--ensure-edge-size` | scale 0 | scales ≥ 1 |
|---|---|---|
| `0` (disabled) | never | never |
| `1` (auto, upstream default) | when decimating | **never** |
| `2` (force) | always | **always** |

With every upstream default the pass therefore runs **exactly once**, on the
input mesh. Only "force" crosses into the refinement loop, and "force" is what
`coarsen` has to pass — with `--decimate 1` there is no decimation for "auto" to
attach to, and upstream's tri-state has no value meaning *force on the input
only*.

### Why the later scales are the expensive ones

Two derivations, both from the source rather than from measurement:

* **The mesh is bigger at each later scale.** Subdivision is bounded by
  *projected* face area at the current image scale, and `--scale-step 0.5` makes
  each pass' images 4x the area of the last. At a fixed `--max-face-area` the
  face count after `Subdivide` therefore grows by roughly 4x per scale. Scale 0
  remeshes the coarsened mesh; scale 1 and scale 2 remesh what subdivision has
  grown from it.
* **The target is coarser than the subdivision it follows.**
  `EnsureEdgeSize`'s defaults are `epsilonMin=-0.5`, `epsilonMax=-4`, negative
  meaning multiples of the current mean edge length, and the target is their
  mean: **2.25x the mean edge of the mesh handed in**. CGAL collapses edges below
  4/5 of target, so most of a freshly subdivided mesh is collapse-eligible. ADR
  0009's own measurement confirms the factor from a real log rather than from the
  code: edge target 0.0620153 against mean edge in 0.0275624 is 2.2500.

So at the finer scales the pass remeshes a mesh ~4x larger per scale, toward a
target coarser than the subdivision that just built it, immediately before the
optimizer runs on the result. The one datum in hand is ADR 0009's scale-0
figures, 6m14s and 2m56s on the same fragment prepared two ways — the cheapest
of the three passes, on the smallest mesh of the three.

## Decision

### 1. A patch, not a flag

`dependencies/patches/openMVS-v2.4-ScopeEnsureEdgeSizeToInput.diff`, applied in
`BuildOpenMVS.cmake` beside the other five. The value the pipeline needs does not
exist in the binary, and the alternative was to invent a stand-in for it here:
run `refine` with `--ensure-edge-size 2 --scales 1` three times, or drop to
`--scales 1` and lose the coarse-to-fine schedule. Both would rearrange the
pipeline around a flag's blast radius, and the pipeline is not where this
belongs — `refine` is one binary invocation (ADR 0005), and the wrappers stay a
transcription of the binary's flags rather than a place to work around them.

The patch is two lines of the same shape as the line above them, in both copies
of the loop:

```cpp
refine.SubdivideMesh(nMaxFaceArea, nScale == 0 ? fDecimateMesh : 1.f, nCloseHoles,
    nScale == 0 ? nEnsureEdgeSize : 0);
```

### 2. Scope "force" to the first scale, rather than adding a fourth value

The narrower change would be a `3` meaning "force on the input only", leaving `2`
alone. Rejected: nothing would then ever want `2`. Its only effect over `3` would
be to add remeshing passes inside the optimization loop that no upstream
configuration performs, at the scales where they cost the most — a value kept
live for a behaviour with no caller, in a patch we have to re-apply at every pin
bump.

Scoping `2` is also the smaller behavioural claim. `0` and `1` are **unchanged**,
byte for byte: the guard already excluded the later scales for both, because
`bNoDecimation` is unconditionally true there. The patch moves "force" into the
regime the other two values are already in, and into the regime every stock
OpenMVS run is already in. That is what makes accepting this without an A/B
defensible — the scales ≥ 1 behaviour is not a new configuration, it is the only
one upstream ships.

The file's own `#define MESHOPT_ENSUREEDGESIZE 1 // 0 - at all resolution`
(`SceneRefine.cpp:42`) reads as the same intent: the macro selects first
resolution vs. all resolutions, and it is set to first.

### 3. Both copies of the loop, including CUDA

`Scene::RefineMeshCUDA` (`SceneRefineCUDA.cpp:844-925`) is a duplicate of the
loop with a duplicate `MeshRefineCUDA::SubdivideMesh` and a duplicate guard at
`SceneRefineCUDA.cpp:518`, calling the same `Mesh::EnsureEdgeSize`. Both live in
one binary; `--cuda-device` picks between them, falling back to the CPU path when
`RefineMeshCUDA` returns false (`RefineMesh.cpp:231-242`). A patch to the CPU
path alone would leave the GPU jobs — the ones that hit this — unfixed, and
would do it silently, since neither path announces which it took. The CUDA hunk
is compiled only under `USE_CUDA=ON`, which CI does not build, so it carries the
same "CI misses it" exposure as `FixRefineCUDABounds` beside it.

### 4. `REFINE_WITH_COARSEN` does not change, and neither does the wrapper's surface

`--decimate 1 --ensure-edge-size 2` remains what `coarsen` implies
(`stages.REFINE_WITH_COARSEN`), the derivation stays in `stages.refine_flags`,
and both warnings in `stages.warn_refine_decimation` still say what they said.
The flags are correct; it was their reach inside the binary that was wrong. No
test changes: `tests/test_openmvs.py`'s `SURFACES` table covers the argv `refine`
builds, which is unaffected.

What does change is that `2` now means something different in our build than in
stock OpenMVS, and ADR 0005 makes the wrapper docstring the place that is
recorded — `mvs_refine` says so, and `--refine-ensure-edge-size`'s help says
"force (on the input mesh)" rather than "force".

## Not yet measured

Stated plainly, because this ADR is accepted without it:

* **The cost removed.** No timing exists for the scale 1 and scale 2 passes. The
  argument for their size is the ~4x per-scale growth and the 2.25x target,
  derived above; the reported symptom (>1h silent between iterations, quickly
  through the first) is consistent with it and is not a measurement of it. A
  `-v 3` refine, or a stalled job's `Mesh subdivided:` line and the wall time to
  the following `Ensured edge size`, would settle it.
* **The refined output.** Unchanged-vs-changed has not been compared. The
  mitigating fact is Decision 2's: the patched behaviour at scales ≥ 1 is
  upstream's default behaviour, not a new regime.
* **Whether this is the reported stall.** It is the strongest candidate in the
  scale loop and the only step there whose cost the coarsen flags newly
  introduce, but `SubdivideMesh` also calls `ListCameraFaces` and `ListFaceAreas`
  at each scale, and `Mesh::Subdivide` itself grows. `--refine-scales 2` and a
  higher `--refine-max-face-area` remain the operator-side levers, and a rebuilt
  image is what tests the patch — a published image cannot.

Recorded on **issue #23**, which already carries ADR 0009's deferred refine A/B
and the cluster RSS figure, since the same job answers all three.

## Consequences / non-obvious traps

* **The fix arrives with a rebuilt toolchain, not with a package upgrade.** This
  is the first patch in `dependencies/` whose effect an operator would look for
  in pipeline behaviour, so a `pgs-recon` that is up to date against a published
  image that predates the rebuild will still show the old cost. The image tag is
  the thing to check.
* **`--ensure-edge-size 2` no longer means what upstream's `--help` says it
  means.** Anyone reproducing a refine invocation against stock OpenMVS — as ADR
  0009's *Reproducing* section does, against a published image — gets the
  unscoped behaviour, and a wall-clock comparison against it is a comparison of
  two different pipelines.
* **The scale-0 pass is untouched and still unbounded.** It is the pass ADR 0009
  measured at 6m14s and 2m56s, it is still single-threaded CGAL over the whole
  mesh, and it is still silent until it finishes. This ADR removes two of three
  such passes; it does not make the remaining one visible or bounded.
* **The pin bump has one more patch to re-apply.** `EnsureEdgeSize` is exactly
  the code upstream rewrote onto `cdcseacave/halfmesh` (see
  `FixSpikeRemovalStaleVertex`'s note), so the pin bump that picks that rewrite
  up is likely to need this reworked rather than re-applied. The guard's
  behaviour table above is what to re-establish against, not the hunk.

## Ruled out

**Disabling the pass entirely (`--refine-ensure-edge-size 0`).** The README's
pre-coarsen advice for a stalling refine, and it reintroduces exactly what ADR
0009 Decision 4 measured: refine subdividing straight from 9,761,506 to
9,955,993 faces, 3.9x the intended input, silently. The input pass is the one
that earns its cost.

**`--refine-scales 2`, or a higher `--refine-max-face-area`.** Both reduce this
cost, and both are the right lever for an operator with a job in flight — but
each buys it by changing the refinement itself, one dropping the finest scale and
the other coarsening every scale's subdivision. They trade output for wall clock;
the patch does not.

**Making the pass log its progress instead.** Visibility is the other half of the
complaint, and a `Util::Progress` around the CGAL call would answer "is it
hung?". It also does not make the two later passes worth running. Worth doing on
the scale-0 pass, which stays; not a substitute for this.
