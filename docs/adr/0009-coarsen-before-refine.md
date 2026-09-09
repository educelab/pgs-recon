# Coarsen the mesh ourselves before refining it

**Status: accepted.** Supersedes the working document
`docs/pre-refine-decimation-plan.md`, which was never committed and should be
deleted once this lands; its measurements are reproduced below because they are
the whole argument.

Amends [ADR 0008](./0008-error-bounded-decimation.md) in one respect only: that
ADR named `RefineMesh --decimate` as one of the three OpenMVS decimations we do
not control and said `pgs-decimate` exists as "an alternative to the CGAL one
rather than an inheritance of it". This is the stage that takes it up — the same
binary, a different target, and one stage earlier. Nothing about ADR 0008's
contract changes: the `decimate` stage still states a *measured deviation* bound
on the deliverable. This stage states a *face count* on refine's input, and the
difference is the point.

## Context

Before it refines anything, `RefineMesh` decimates its input mesh by CGAL
Garland–Heckbert edge collapse — `Mesh.cpp:925-945`, called from
`SceneRefine.cpp:508-535` — single-threaded and silent. Silence and
single-threadedness together explain the symptom operators report: the stage log
goes dead after `Mesh loaded`, sixteen requested CPUs sit idle, and nothing
distinguishes a stalled job from a slow one until Slurm kills it. It has killed
seventeen `refine` jobs on the wall clock.

**The target is not what varies.** OpenMVS computes it as

```cpp
MAXF(0.1f, fMedianArea/fMaxArea)
// fMedianArea = 6 × median projected face area
// fMaxArea    = --max-face-area, default 16
```

and across 119 measured `recon-refine` stage logs (2026-07-31 to 2026-09-08) it
resolves to **0.375, every time** — which puts the median projected face area at
exactly 1.0 px², because `densify` reconstructs at one face per pixel. It is a
property of the rig and of `densify_resolution_level`, not of the fragment.

### Cost at that fixed target

| Fragment | Input faces | Ratio | Output | Wall | Faces/s |
|---|---|---|---|---|---|
| PHerc0017Cr1_Osloense | 25,962,930 | 0.375 | 9,736,518 | 8h36m57s | 523 |
| PHerc0009Cr5_Osloense | 22,969,862 | 0.375 | 8,613,752 | 4h50m29s | 824 |
| PHerc0021Cr3_Osloense | 26,047,135 | 0.375 | 9,767,890 | 4h21m33s | 1,037 |
| PHerc0015Cr03 | 32,996,110 | 0.375 | 12,373,544 | 1h43m15s | 3,329 |
| PHerc0021Cr2_Osloense | 26,003,288 | 0.375 | 9,751,495 | 30m53s | 8,775 |
| PHerc0017Cr3_Osloense | 25,126,025 | 0.375 | 9,422,265 | 4m17s | 61,104 |
| PHerc0013Cr04 | 14,883,799 | 0.375 | 5,581,424 | 2m12s | 70,473 |

Input size does not predict cost: the largest mesh in the corpus finishes in a
fifth of the time the worst one takes, at 79% of its size. Two fragments of the
same mount, within 4% of the same face count, differ by **7.2×**. The full
spread is **135×**.

`pgs-decimate --max-faces <target> --max-error 0` does the same reduction at
32,930–57,286 faces/s over three of those meshes — a **1.7× spread** — and does
not preserve CGAL's ordering: the mesh CGAL finds hardest is the one vcglib
finishes fastest. The pathology is in CGAL's implementation, not in the
geometry.

| Fragment | Faces | CGAL (Xeon 8358) | vcglib (M5 Max) |
|---|---|---|---|
| PHerc0006Cr05 | 29,821,605 → 11,183,100 | 286s | 566s |
| PHerc0017Cr3_Osloense | 25,126,025 → 9,422,265 | 257s | 431s |
| PHerc0021Cr2_Osloense | 26,030,684 → 9,761,507 | 1,852s | **284s** |

vcglib is ~2× slower than CGAL on a well-behaved mesh, on faster single-core
hardware, so call it ~4× slower like-for-like in the healthy case. That is
affordable against a 7.2× spike and irrelevant against the 108× one. **What
matters is that the cost is bounded.** Both decimators are single-threaded, so
core count affects neither.

### What the CGAL pass actually buys

| Fragment | Input | After GH | After `EnsureEdgeSize` | Retained |
|---|---|---|---|---|
| PHerc0010Cr02 | 29,715,813 | 11,143,428 | 3,117,232 | 0.280 |
| PHerc0015Cr02 | 34,466,427 | 12,924,908 | 3,514,577 | 0.272 |
| PHerc0016Cr04 | 31,422,974 | 11,783,659 | 3,216,269 | 0.273 |
| PHerc0006Cr05 | 29,821,605 | 11,183,100 | 2,916,175 | 0.261 |
| PHerc0021Cr2_Osloense | 26,003,288 | 9,751,495 | 2,403,220 | 0.246 |

**Refinement never sees the decimated mesh.** `EnsureEdgeSize` immediately
re-reduces it by a further 3.6–4.1×, and *that* is what iteration zero starts
from. The expensive pass produces an intermediate waypoint costing anywhere from
two minutes to eight and a half hours. The retained fraction tracks the post-GH
size rather than normalising it away, so the pre-decimation target still
matters — 0.375 is the value that reproduces today's refine input.

### Validated end to end

`PHerc0021Cr2_Osloense`, both preparation paths:

| | CGAL path (2026-09-04, cluster) | This stage (local) |
|---|---|---|
| Decimation | 30m53s | **4m44s** |
| Faces after decimation | 9,751,495 | 9,761,506 |
| `EnsureEdgeSize` | 6m14s | 2m56s |
| Edge target | 0.0620153 | 0.05988 |
| Mean edge in | 0.0275624 | 0.0266133 |
| **Refine input faces** | **2,403,220** | **2,570,361** |
| Total preparation | 37m07s | **7m40s** |

The refine input lands **7.0% larger**, from an edge target 3.4% finer —
vcglib's vertex placement leaves a slightly shorter mean edge at the same face
count. Refinement costs ~7% more and resolves marginally finer. The coarsened
mesh deviates from the reconstruct mesh by max 0.0259629 cm, well inside the
`decimate` stage's own 0.1 cm budget, on a surface refinement then moves anyway.

The 2026-09-08 run of this same fragment did not finish the first row inside
four hours (`JOB 134086 ... CANCELLED AT 2026-09-08T19:30:33 DUE TO TIME
LIMIT`).

### Verified against the real meshes

The implemented stage, run through `pgs-recon` against
`TEST_FINAL_PHerc0021Cr2_Osloense_recon_20260908_133519` — whose manifest still
has `refine` recorded `running`, being the job that was killed — in
`ghcr.io/educelab/pgs-recon:2.0.0-alpha.6` with the package mounted over the
installed one:

```
Coarsening 26030684 faces to 9761506 (0.3750 of the input)
WARNING: mesh is non-manifold (0 edges, 1 vertices).
  round: quadric error 0 -> 9761506 faces, max deviation 0.0259629 (within budget)
Bound: max_faces after 1 round(s); quadric error 0
target met
coarsen: complete in 328.6s
```

Confirming, on the real artifact rather than on the arithmetic:

* **The target reproduces the CGAL pass's.** 9,761,506 against CGAL's
  9,761,507. The other two meshes on hand read 29,821,605 → 11,183,102
  (CGAL: 11,183,100) and 25,126,025 → 9,422,259 (CGAL: 9,422,265).
* **One round**, as Decision 2 claims, and `target met` exactly.
* **The deviation is the same number the working document measured**,
  0.0259629 — well inside the `decimate` stage's own 0.1 cm budget.
* **The non-manifold vertex is real** and a face budget is indeed unaffected by
  it, which is the hazard Decision 2 cites.
* **328.6s against CGAL's 1,852s on this mesh: 5.6× faster** — in Docker on
  macOS, so with filesystem and VM overhead the native 284s figure does not
  carry. Against the 2026-09-08 run that did not finish in four hours, the
  comparison is not a ratio.
* **Peak RSS 14,241,528 kb (13.6 GiB)**, `max_rss_exact`.
* **The stage really does need only the mesh.** That directory was given the
  manifest and `mvs/reconstruct_mesh.ply` and nothing else — no scene, no
  undistorted images, no depth maps — and `--from coarsen --to coarsen`
  resolved, ran and recorded. Which is Decision 1's argument, tested.

## Decision

### 1. A stage, not a flag on `refine`

`coarsen`, between `reconstruct` and `refine`, declared
`IO(('mesh',), (), ('mesh',))` — the `decimate` stage's shape minus the
`deviation` output. Output is `mvs/coarsen_mesh.ply`, `<stage>_<role>` like
everything else (ADR 0006).

A stage rather than something refine's block does, for the reason ADR 0004
exists: this is the cheapest window in the pipeline to size — one mesh in, one
mesh out, no scene, no images, no depth maps — and refine is the most expensive.
Bundling them would put a bounded single-core reduction inside the allocation
that exists for a 147 GiB optimizer. It also means the reduction is
independently re-runnable, and that its face counts land in a stage record of
their own.

**Named `coarsen`, not `predecimate`.** The vocabulary matters more than the
adjacency: two stages spelled `decimate` and `predecimate`, both driving
`pgs-decimate`, one stating a geometric bound and one a face count, is the
confusion ADR 0008's own naming discipline exists to prevent. `CONTEXT.md`
previously listed "coarsening" as a word to avoid *for* decimation; it is now a
term of its own, and the two entries contrast.

### 2. A face budget, not a deviation budget

`--max-faces <target> --max-error 0`. With a count as the target there is
nothing to search for, so `pgs-decimate` runs one round —
`Bound: max_faces after 1 round(s)`. On a 1.28M-face synthetic mesh: deviation
budget, 4 rounds, 106s; face budget, 1 round, 27s; face budget with sampling
turned down, 10.7s.

It also sidesteps a hazard. Both Osloense meshes tested emit
`WARNING: mesh is non-manifold (0 edges, 1-2 vertices)`, unrepaired, and under
`--preserve-topology 1` that is what stalls a *search* short of its target. A
face budget is unaffected; a deviation budget here could stall.

This is why the stage is not simply "run `decimate` twice with different
budgets". A deviation budget on refine's input would be the wrong contract as
well as the wrong cost: refinement moves this surface, so a geometric guarantee
about the mesh handed *to* it guarantees nothing about what comes out. The
guarantee belongs on the deliverable, which is `decimate`'s job.

### 3. Target = 0.375 × the input mesh's face count

`stages.COARSEN_RATIO`, derived as `6 × 1.0 px² / 16` with that derivation
recorded at the constant. The input count is read from the mesh's PLY header
(`utils.ply.face_count`, `element face N`, inside the first few hundred bytes) —
the one place the pipeline reads a mesh file at all, and a header read rather
than a mesh load. `--coarsen-ratio` overrides it; `--coarsen-max-faces` states
the count outright, for matching a previous run's refine input exactly.

The product is rounded with Python's `round()`, which breaks a tie to even, so a
mesh whose face count times the ratio lands exactly on `.5` gets one face fewer
than half-up would: `PHerc0021Cr2_Osloense`'s 26,030,684 faces give a target of
9,761,506 against CGAL's 9,761,507. One face in twenty-six million is not worth
a tie-breaking rule of our own, but it is worth knowing when comparing a target
against a historical run's.

A header that cannot answer fails the stage, naming both flags that recover.
Inventing a target from nothing is the one outcome worse than stopping: too
small silently throws the mesh away, and too large reinstates the cost this
stage removes.

### 4. Both of refine's preparation flags follow the stage, not the caller

With `coarsen` in the shape, `RefineMesh` runs `--decimate 1
--ensure-edge-size 2`. The guard at `SceneRefine.cpp:556` is

```cpp
if ((nEnsureEdgeSize == 1 && !bNoDecimation) || nEnsureEdgeSize > 1)
```

so `--decimate 1` — which sets `bNoDecimation` — skips the edge-size pass as
well at the default `--ensure-edge-size 1`. Measured directly: with `--decimate
1` alone, `RefineMesh` emitted no `Ensured edge size` line and subdivided
straight from 9,761,506 to **9,955,993** faces, 3.9× the intended refine input,
with nothing in its log to say so. `--ensure-edge-size 2` ("force") is not
optional, and neither flag can be left to an operator remembering the other.

`--refine-decimate` and `--refine-ensure-edge-size` remain explicit overrides,
and both already default to `None`, which is this codebase's "not given" — so no
separate record of explicitness is needed. Two warnings cover the ways this goes
wrong silently: the CGAL pass left on over an already-coarsened mesh, and
`--decimate 1` taking the edge-size pass down with it.

The pair is **derived from the shape, not folded into `args`**. Persisted, a `1`
recorded by a run *with* `coarsen` would be inherited by a later
`--no-mvs-coarsen` run and skip both passes. Refine's dirtiness needs no help
from the recorded flags either way: `coarsen` entering or leaving the shape
rebinds refine's `mesh` input, which the planner already reads as
`inputs changed`.

### 5. On by default whenever `refine` is

Not opt-in. Seventeen killed jobs is the argument, and a fix nobody enables is
not one. `--no-mvs-coarsen` restores the old path. With `--no-mvs-refine` the
stage leaves the shape too — it only ever prepares a mesh for refinement — and
asking for it anyway warns rather than fails, pointing at `decimate` as the
stage that coarsens a deliverable.

Every existing `refine` run changes: the refine input lands ~7% larger and
refinement costs ~7% more, per the table above.

### 6. The face counts are the drift canary

`Decimated faces N (100%, …)` in `RefineMesh`'s log was the only place the 0.375
ratio was ever observable, and disabling the pass removes it. The stage record
therefore carries `input_faces`, `target_faces`, `achieved_faces` and
`achieved_ratio`, via a new `facts=` argument to `StageTracker.end()` —
deliberately not arguments and not paths, so nothing in the planner reads them:
a fact cannot make a stage dirty and binds no role. A rig change, or a
`densify_resolution_level` change, shows up as the achieved ratio moving off
`COARSEN_RATIO` instead of passing unnoticed. Missing the target by more than
5% warns; a face budget with no deviation budget normally lands within a
fraction of a percent, so a larger miss means the collapse stalled.

### 7. The measurement is turned down, and no report is written

`pgs-decimate` measures on every round whatever the target — `attempt()` calls
`measure()` unconditionally — so the measurement cannot be switched off, only
turned down: `--samples-per-face 1 --curvature-samples 0`, which is 10.7s
against 27s on the synthetic mesh.

No `--report`, and no `deviation` role in `STAGE_IO`. A report is only worth
recording where something reads it, and nothing reads a deviation bound on a
surface refinement is about to move; the deviation that matters is
`decimate`'s, on the deliverable. This is the one place the stage is *not*
symmetric with `decimate`, and it is deliberate.

## Consequences / non-obvious traps

* **`pgs-decimate` is now reached by two stages.** Any test or log reader that
  identified a call by tool name has to identify it by what it writes instead.
  `tests/test_pipeline.py` grew `argv_writing()` for exactly this.
* **The fake binaries in `tests/test_pipeline.py` now write PLY headers.** A
  fake `.ply` that declares no `element face N` puts this stage out of reach of
  the end-to-end tests entirely, so `_write()` emits a real header and the
  `pgs-decimate` fake honours `--max-faces`.
* **A resumed pre-2.0 directory gains a stage it never ran.** `coarsen` is dirty
  on such a directory, which cascades into refine and texture. That is correct
  — the recorded refine input was built by the CGAL pass — but it is a re-run a
  1.7 directory did not previously incur.
* **Peak RSS is 13.6 GiB on a 26M-face mesh**, measured (see *Verified against
  the real meshes* above): `max_rss` 14,241,528 kb with `max_rss_exact`, the
  stage being the only one that process ran. That is the number to size a Slurm
  `coarsen` job from, with the usual headroom; it is nowhere near the
  big-memory node `refine` needs, which is the point of the split. It will
  scale with input face count rather than with the target — the pristine input
  plus one candidate is what the collapse holds.
* **The refine A/B has not been run**, and this ADR is accepted without it.
  Everything above validates *preparation*: the reduction and `EnsureEdgeSize`.
  The refined output has not been compared against 2026-09-04's
  `refine_mesh.ply`, because a CPU-only local refine is neither representative
  nor affordable — the stage peaked at 147 GiB RSS on the cluster, and this
  cannot run there until the stage is deployed.

  Deferred to **issue #23** along with the cluster RSS figure, which is what
  would actually size the Slurm stage. Both are recorded there with what has
  already been verified, so the ticket does not repeat it. The stage being on
  by default means a bad A/B result is a regression for every run, so #23 also
  carries the remedy: `--no-mvs-coarsen` restores the old path exactly, which
  is why accepting this ahead of that measurement is recoverable rather than a
  bet.

## Ruled out

Recorded so they are not re-litigated.

**Mesh dirtiness.** `reconstruct`'s own `Cleaned mesh` step removes vertices at
480–1,070 ppm on Osloense meshes against 5–80 ppm elsewhere, which makes it an
excellent *mount* marker and a useless cost predictor. The slowest mesh in the
corpus (PHerc0017Cr1, 31,016s) is the *least* dirty Osloense at 483 ppm;
PHerc0017Cr3 at 973 ppm finishes in 257s; the cleanest mesh anywhere,
PHerc0015Cr03 at 14.5 ppm, still took 6,194s. A cleanup pass would not have
fixed this.

**Input size.** No relationship. 32,996,110 faces in 6,194s against 25,962,930
faces in 31,016s.

**`--mvs-smooth`.** A genuine amplifier, already fixed, not the current cause.
All sixteen historical wall-clock kills ran `--smooth 2`, whose
`PMP::smooth_shape` pass handed `refine` a mesh that decimated 2–37× slower.
Defaulting to 0 cleared it and holds — ten of eleven jobs in the 2026-09-08
batch refined normally, three of them Osloense. The seventeenth kill, and every
measurement above, are at `--smooth 0`.

**Mount type.** Osloense dominates the slow tail but does not explain it.
`PHerc0017Cr3_Osloense` decimates in 4m17s, faster than most non-Osloense
fragments. The pathology is per-fragment.

**Why CGAL stalls on specific fragments is unexplained**, and the fix does not
depend on the explanation. Collapse-validity rejection — topology, normal
flipping, triangle quality — is the obvious place to look, since target, input
size and degenerate-face count are all excluded. Worth knowing only if
OpenMVS's decimator is wanted back for some other reason.

## Reproducing

```sh
# vcglib decimation, face budget, no search
docker run --rm -v "$MVS":/in:ro -v "$OUT":/out \
  --entrypoint pgs-decimate ghcr.io/educelab/pgs-recon:2.0.0-alpha.6 \
  -i /in/reconstruct_mesh.ply -o /out/coarsen_mesh.ply \
  --max-faces 9761506 --max-error 0 \
  --samples-per-face 1 --curvature-samples 0

# RefineMesh preparation, CGAL disabled and edge-size forced
docker run --rm -v "$W":/w \
  --entrypoint /usr/local/bin/OpenMVS/RefineMesh \
  ghcr.io/educelab/pgs-recon:2.0.0-alpha.6 \
  -i densify.mvs -m coarsen_mesh.ply -o refine_test.ply -w /w \
  --archive-type -1 --scales 3 --decimate 1 --ensure-edge-size 2
```

`RefineMesh` is not on `PATH` in the container; it lives at
`/usr/local/bin/OpenMVS/RefineMesh`. Scene image paths in `densify.mvs` are
relative (`undistorted_images/…`), so the working directory needs the scene, the
mesh and that image directory and nothing else — depth maps are not read by this
stage.

Container `ghcr.io/educelab/pgs-recon:2.0.0-alpha.6`, OpenMVS x64 v2.4.0.
