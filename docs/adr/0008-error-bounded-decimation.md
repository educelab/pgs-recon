# Own decimation, bounded by measured deviation

**Status: accepted.** Specified as issue #21 and implemented alongside it.
Supersedes the draft plan
`docs/plans/qecd-decimation-stage.md`, which is deleted with this ADR: two of its
recommendations are now known wrong (see Decision 2) and one of its code snippets
does not compile against the pinned vcglib. Three details of the search
policy in Decision 4 were settled during implementation; they are recorded at the
end.

**Amended** after the first cluster batch (reported by the acquisition-workflow
team against 2.0.0-alpha.6). Nothing about the contract changed and nothing
about it is in question: every run's measured deviation came in under budget.
What changed is the search's **cost**, and the batch's own logs are the reason.
Of the 17 decimate runs in `/Volumes/Supernova/Processing/logs`, **nine
delivered no usable result** — eight cancelled at the scheduler's three-hour
limit, one on an unrelated error — against seven that converged in 6.7 to 28.3
Mface-units. A stage that loses half its jobs to its own search cost is the
problem this amendment exists for. Decisions 4 and 5 and the implementation notes carry the
revision; the superseded mechanism is recorded alongside its replacement rather
than deleted, because its failure is the argument for what replaced it.

## Context

A finished reconstruction's mesh is multiple millions of faces, which makes the
deliverable slow to open and manipulate in MeshLab. These meshes can be coarsened
substantially — but the objects are inscribed surfaces, so **smoothing away sharp
features and edges is the one unacceptable outcome**. What is wanted is the
fewest faces that stay within a stated geometric deviation of the input, not a
target face count.

Decimation already happens in this pipeline, three times over, and none of it is
ours:

- `ReconstructMesh --decimate` / `--target-face-num`,
- `RefineMesh --decimate`, which is a CGAL Garland–Heckbert edge-collapse pass on
  the way *in* (`Mesh.cpp:925-945`, from `SceneRefine.cpp:508-535`), single
  threaded and silent, and the pipeline's wall-clock hog at the pinned revision
  — since taken over by the `coarsen` stage, which drives `pgs-decimate` to a
  face count instead ([ADR 0009](./0009-coarsen-before-refine.md)),
- `TextureMesh --decimate`.

All three are face-fraction or face-count targets. None of them states a
geometric bound, none reports what it cost, and in practice they are unreliable
for mesh conditioning. The secondary motivation for this stage is therefore to
have a decimation path we control and can attest to, and an alternative to the
CGAL one rather than an inheritance of it.

## Decision

### 1. A new in-tree binary, `pgs-decimate`, on vcglib

`dependencies/cmake/BuildVCG.cmake` already installs vcglib 2025.07 headers to
`${PREFIX}/include/vcg` for OpenMVS's benefit.
`vcg::tri::TriEdgeCollapseQuadric` is the canonical Garland–Heckbert QECD, and is
what MeshLab's "Quadric Edge Collapse Decimation" filter drives. Header-only, no
new superbuild entry, no new runtime dependency.

**Not CGAL**, though it is also already in the build: choosing it would
reimplement the exact pass we are trying to stop inheriting, and its stop
predicates are count- and ratio-shaped anyway. **Not libigl** (`igl::qslim` is a
thinner QSlim behind a whole new superbuild entry). **Not pymeshlab / open3d /
trimesh**: the Python layer is subprocess orchestration and does no mesh math.

### 2. The contract is *measured* deviation, not the library's metric

The obvious design is `vcg::LocalOptimization::SetTargetMetric(err)`, which halts
when the cheapest remaining collapse exceeds `err`. That was the draft plan's
recommendation, and reading the pinned header disqualifies it as a *user-facing
bound*:

- the value compared against the target is the heap priority
  `QuadErr / (newQual * MinCos)`
  (`local_optimization/tri_edge_collapse_quadric.h:404-411`), and `QualityCheck`
  **defaults to true**, so it is a quadric error divided by a triangle-quality
  factor in `(0, QualityThr]`;
- under `ScaleIndependent` (default **true**) the quadric is pre-multiplied by
  `ScaleFactor = 1e8 * pow(1.0/bbox.Diag(), 6)` (`:604-608`);
- termination is `currMetric > targetMetric` on that composite
  (`local_optimization.h:289`).

So the threshold has no physical units under any parameter combination we would
actually ship, and a quadric error compounds across composed collapses, which
makes it under-report deviation exactly in the high-curvature regions this exists
to protect.

**Therefore: the quadric threshold is a search variable, and the contract is a
measured distance.** `pgs-decimate` decimates and *measures*, repeatedly, until
the measured deviation is within the stated budget. The budget is in solved-frame
units, which is meaningful because the pipeline autoscales
(`mvg_autoscale` → `pgs-global-scaler`).

The rejected middle option — ship the quadric threshold first under the name
`--decimate-max-error`, and binary-search later behind the same flag — is
rejected specifically because the manifest compares *recorded argument values* to
decide dirtiness. A flag whose meaning changes between versions makes two
incomparable numbers look comparable to the planner.

### 3. What "measured deviation" means, precisely

Sampled Hausdorff distance via `tri::HausdorffSampler` and `SurfaceSampling`
(`point_sampling.h:229-273`), and three choices that are not cosmetic:

- **Symmetric.** The gate is the larger of the two one-sided maxima. The
  asymmetry is the whole point: if a collapse erases a ridge, sampling the
  *decimated* surface finds every sample close to the original, and the loss only
  appears when sampling the *original* surface for the missing ridge.
- **Max, not a percentile.** Max is the only statistic that means "nothing moved
  further than this", which is the requirement. A percentile would quietly permit
  the erased-ridge case. It is a **sampled** max, and therefore a lower bound on
  true Hausdorff distance, which is why every vertex of both meshes is included
  as a sample.
- **Two sampling passes.** A uniform `Montecarlo` pass (ten samples per face of
  the coarser mesh, floored at 1e6), plus a curvature-weighted top-up pass at
  half that, biasing samples toward where deviation is largest.
  `UpdateCurvature::PerVertexAbsoluteMeanAndGaussian` needs only `VFAdjacency`
  and compactness (`update/curvature.h:536`),
  `UpdateQuality::VertexAbsoluteCurvatureFromHGAttribute` maps it onto vertex
  quality (`update/quality.h:366`), and `SurfaceSampling::WeightedMontecarlo`
  samples proportional to it (`point_sampling.h:1224`). No FFAdj, no vertex
  components beyond the `Qualityd` the decimation already needs.

  The weighted pass counts toward the **gating max only**. Reported `mean` and
  `RMS` come from the uniform pass alone, so they stay comparable to what `metro`
  or MeshLab's Hausdorff filter would print for the same mesh pair. Each pass's
  sample count goes in the report.

### 4. Search policy

Search the quadric threshold in log space, aiming each probe from a power-law
fit to the rounds already measured, capped at ten decimate-and-measure rounds,
keeping the **coarsest feasible** result seen rather than the last one tried.

A round is not free and its price is not constant. The measurement samples ten
times per face of the *coarser* mesh (Decision 3), which is always the
candidate, so a probe that barely collapses anything is far dearer than one that
collapses almost everything. But it is not *only* the candidate: every round
copies the pristine input, builds a quadric heap over all of its edges, and — in
the reverse direction of the symmetric measurement — grids, normals and
curvature-maps the whole input again. That part is paid per round whatever the
candidate's size.

Measured rather than assumed, by timing two single-round runs at opposite ends
of the threshold range on a 2,785,306-face input: **29.2 s per round fixed, plus
81 s per million candidate faces**. A round's floor is therefore about 13% of
the input's own face count, and the spread between the cheapest and the dearest
probe is nearer **seven to one than the eighty-five to one** that counting
candidate faces alone implies.

Be careful what that floor is, because it reads like per-round overhead and is
not. It is `kMinSamples`: the cheap probe left 54,042 faces, `10 x 54,042` is
under the million-sample floor, and that round therefore sampled a million
points per direction whatever its candidate's size. The floor is a *measurement*
minimum. What a round actually spends is closest-point queries — a five-round
search on an 11M-face mesh issued on the order of 167 million of them in its
last round alone — and those scale with the candidate, against a mesh that
differs every round. Round *count* still matters, but through the candidates
each round buys rather than through any fixed setup cost, which is small.

The placement of a probe and the decision to stop are therefore both priced, not
only informed; what that means concretely is in the implementation notes.

Deviation is monotone in the threshold in practice but vcglib guarantees
nothing, and tracking the best feasible result makes non-monotonicity a loss of
optimality rather than a correctness bug. The invariant is then unconditional:
**the output's measured deviation is never above the budget.**

Each round re-decimates from a pristine copy, because collapses are destructive,
so peak memory is two copies of the mesh.

**When even the cheapest collapse exceeds the budget, that is an answer, not a
failure**: write the input through unchanged, report `faces_removed: 0` with the
reason, exit 0. A stage that errored there would break a resumed run over an
already-minimal mesh.

### 5. Targets, caps, and precedence

Three target modes, because not every mesh through this pipeline is scaled to
known world units or has the same goal:

| Flag | Means |
|---|---|
| `--max-error` | measured deviation budget, in solved-frame units |
| `--max-faces` | face budget |
| `--quadric-error` | raw threshold, documented as unitless, escape hatch |

`--quadric-seed` is a fourth value of the same kind as `--quadric-error` and is
deliberately not a target: it *starts* the search rather than replacing it, so
the measured guarantee survives it. It exists because the seed is a guess and is
routinely orders of magnitude high (see the implementation notes), and a caller
who has run meshes of a kind before knows better than the formula does.
`--min-gain` is the stop condition's other half, and is documented with it
below.

One is primary; the others act as caps when also present, so
`--max-error 0.5 --max-faces 2000000` reads as "within 0.5 units, and never more
than 2M faces". The two pull in opposite directions and can conflict, so
`--prefer {error,faces}` decides, **defaulting to `error` in both the binary and
the pipeline**. Under `--prefer faces` the report carries
`budget_exceeded: true`, so a deviation that was not honoured can never be
mistaken for one that was. A tool whose default silently violates its own stated
bound is worse than one that needs a flag to do so.

### 6. Enablement: the target *is* the enable flag

`pipeline_shape` has two idioms — a boolean (`args.mvs_refine`) and presence of a
value (`args.mvg_autoscale is not None`). Decimation takes the second: a
`--decimate-max-error` or `--decimate-max-faces` puts the stage in the shape, and
there is no `--mvs-decimate`. This makes "enabled with no target" and "target set
but stage off" unrepresentable.

The gate is **truthiness, not `is not None`**, so `--decimate-max-error 0`
switches the stage off. That is the disable path, and it exists because
`apply_stored` (`stages.py:396-401`) folds the manifest's effective arguments in
as defaults: on a resume, simply *omitting* the flag inherits the recorded value
and the stage runs anyway. Zero is also honest as a value — a zero deviation
budget permits only free collapses — and it must be said in the flag's help text,
because it is not guessable. `--mvg-autoscale` has the same wart with no way out;
a general mechanism for clearing an inherited argument is filed as issue #20 and
is deliberately not in this change.

Because the gate is `any()` over the two budgets, a zero has to clear the budget
*beside* it as well as itself, or a run recorded with both would keep decimating
to the inherited face target — the stage the flag is documented to switch off
surviving on the other budget. `clear_zeroed_budgets()` does that, after
`revert_out_of_range` and before the shape is read off the budgets, and warns
about the one it drops. The exception is a run that names both budgets
explicitly: `--decimate-max-error 0 --decimate-max-faces 2000000` is a caller
saying "not this bound, that one", which is a target change and not a disable.

A **negative** budget is refused outright, in `validate_budgets()` and again in
`pgs-decimate` itself. `Targets::have*` reads non-positive as *not given*, so
`-1` is not a tighter bound: alone it reaches the binary as no target at all and
fails the run after densify, reconstruct and refine have been paid for, and
beside a face budget it silently vanishes and the mesh is coarsened to a target
nobody set. Zero is the off switch and a negative number has no second reading.

### 7. Position, and what it produces

The `decimate` stage sits between `refine` (or `reconstruct`, when refine is off)
and `texture`, consuming `mesh` and rebinding `mesh`, exactly as `refine` does.
Placing it *before* refine — as a replacement for RefineMesh's CGAL prep pass —
is a different contract (feed refine faces that project to about a pixel, not a
world-unit deviation budget) and is deliberately not this stage. That use is
served by running the binary standalone with `--refine-decimate 1`.

**It always writes `mvs/decimate_mesh.ply`, including in the pass-through case**,
where the file is a copy of its input. Recording no output instead would work
mechanically (`StageTracker._absorb` would leave `mesh` bound to refine) but it
puts a data-dependent edge into a graph whose entire value is that the edges are
declared, and it makes "declined to decimate" indistinguishable from "produced
nothing recorded". The duplicated mesh on disk is the accepted cost.

### 8. The measurement is an artifact, not stdout

`StageTracker.end()` records role-to-path bindings only (`stages.py:774`), and
`run_command` does not capture stdout (`utility.py:57`), so the numbers have
nowhere to go today. The binary takes `--report <path>`, `layout` names it
`mvs/decimate_report.json`, and the stage records it as the leaf role
`deviation`, the way `colorize` records `colorized`. No tracker change, no stdout
capture, and the report exists whether or not a pipeline ran the tool — which
matters, because the binary ships first.

The alternative, a `stats=` parameter merging a per-stage-shaped payload into the
stage record, is a new kind of thing in the manifest and would want its own ADR.

### 9. UVs are dropped, loudly

At this point in the pipeline the mesh carries no UVs: reconstruct, refine and
texture emit only geometry, and `TextureMesh` parameterizes *after* this stage.
So plain positional QEM is sufficient, with no attribute-preserving machinery and
no seam handling.

Run standalone against a textured deliverable, the tool warns and writes geometry
only. Preserving UVs would need `WedgeTexCoord` components, seam-aware collapse
guards and a texture-aware quadric; getting that subtly wrong produces texture
swimming that **no deviation measurement would catch**, because the geometry is
fine and the parameterization is what broke. Refusing such input outright was
considered and rejected: manual use is manual, and the two-command workflow
(decimate, then re-texture) is the user's to choose.

## Consequences / non-obvious traps

- **`--decimation-factor` is deleted, not aliased.** It is `RefineMesh`'s
  `--decimate` and is renamed `--refine-decimate`, matching the rest of that
  group. Two flags whose names both say "decimate", one of them silently meaning
  another stage's decimation, is the kind of confusion that costs a cluster
  allocation. We are in alpha on a major release, so it goes in
  `docs/migrating-to-2.0.md` §5a and the orchestrator checklist rather than
  living on as a hidden alias.
- **Disabling the stage on a re-run is already handled by the graph.** Drop the
  target and `texture`'s recorded `mesh` input no longer matches its binding, so
  `_dirty_reason` returns `inputs changed: mesh` (`stages.py:610-613`) and
  texture re-runs against refine's output while everything earlier stays `skip`.
  This is the densify precedent documented at `stages.py:603-606`. The orphaned
  `decimate` record stays in the manifest and plans as `off`.
- **Adding a stage renames nothing downstream.** That is
  [ADR 0006](./0006-stage-named-artifacts.md) being cashed in: under the old
  chained names, inserting a stage here would have renamed the deliverable's
  whole ancestry.
- **The report is a leaf, so nothing may consume it.** If a later stage ever
  reads `deviation`, that is a new edge in `STAGE_IO` and a new dirtiness path;
  it is not a free addition.
- **`work_dir()` does not apply, but the output location still does.** The tool
  resolves absolute paths itself, so it needs no co-location treatment — but its
  output must land in `mvs/` regardless, because `mvs_texture`'s `-m` must
  already be in the working directory.
- **A world-unit budget against an unscaled solve is meaningless**, and the
  pipeline warns when `--decimate-max-error` is given with `autoscale` absent
  from the shape. The binary cannot detect this: it only sees a mesh.
- **Input conditioning is cleaning, never repair.** `RemoveDuplicateVertex`,
  `RemoveUnreferencedVertex` and `RemoveDegenerateFace` always run and their
  counts are reported; a non-triangular face is a hard error rather than being
  silently fanned. Non-manifold input is *reported*
  (`CountNonManifoldEdgeFF`/`VertexFF`) and not repaired, so that when
  `PreserveTopology` stalls the search short of the target, the reason is in the
  same output as the failure.
- **`QualityCheck` can stay at its default now.** With the contract measured
  rather than predicted, the composite metric only shapes collapse *order*, and
  distorting the ordering does not weaken the guarantee.
- **`OptimalPlacement` stays on, which moves vertices off the input surface.**
  That is safe here only because the bound is measured: the measurement catches
  the excursion the metric under-reports. Anyone turning the measurement off has
  also turned the guarantee off.
- **`HausdorffSampler`'s `dist_upper_bound` has to span *both* meshes.** A
  sample landing further than it is discarded outright -- not counted, not
  folded into the max -- so deriving it from the target mesh's own bounding box
  turns the worst outcome the tool exists to catch into a silent pass:
  decimation that deletes a detached fragment shrinks the target's box, and the
  samples stranded on the deleted geometry are exactly the ones over the new
  bound. `measure()` therefore takes the diagonal of the union of the two boxes
  and hands it to both directions.
- **`--quality-threshold 0` is an off switch, not a loose setting, and is
  refused.** vcglib clamps a candidate triangle's quality to `QualityThr` and
  then divides the quadric error by it, so zero makes every collapse cost
  infinity, `GoalReached()` is true before the first one, and the mesh passes
  through untouched with nothing in the report naming the flag. The admissible
  range is `(0, 0.866]`, `0.866` being the most an equilateral triangle scores.
- **The `exit 139` runs were dying of the probe placement, not of the budget.**
  Two local runs at a 0.07 budget on ~11M-face meshes died with `SIGSEGV` after
  11.3 hours, reported as most likely self-inflicted -- nine containers in a
  94 GiB VM, and a tight budget keeping every candidate near full size. The
  signal is real, and `139` rather than `137` is what Linux overcommit does
  rather than an unchecked allocation: `malloc` succeeds against memory the
  kernel cannot back and the fault lands on first touch, which is exactly what
  an 8.5x slowdown against a third run of the same size and budget describes.

  But *why* every candidate was near full size is the search, not the budget.
  Reading the two logs back: `PHerc0009Cr5` spent **three of its four rounds** at
  11,130,905, 11,132,813 and 11,131,877 faces against an input of 11,133,709 --
  99.97% and up, three times over, 33.6 Mface-units of measurement to learn
  almost nothing -- and `PHerc0021Cr2` spent two of five the same way. Peak RSS
  was maximal every round because the search kept *buying* maximal candidates.

  The mechanism is the first refit, which the trust region never covered: it
  needs two prior probes to have a baseline, so after round one there is no
  clamp at all, and a tight budget makes the prior exponent ask for a dive of
  many orders. Both runs took it -- `1.48e-04` to `3.02e-09`, a factor of
  49,000, on the very first correction. The cost bound applies from the first
  refit instead, using the prior face exponent, and would have capped that round
  at 426,428 faces. So the fix for the wall clock is also the fix for the memory
  profile, and neither wants a code change beyond the one already made.

- **The search policy is replayed in simulation, and simulation is where it is
  tuned, not where it is trusted.** The policy is pure arithmetic over
  `(threshold, faces, deviation)` triples, so it can be run outside the binary
  against models of real meshes — a face count following
  `min(input, C * threshold^-k)` and a deviation quantized onto a staircase — at
  no cost. That is what the numbers over "four hundred randomized staircase
  meshes" in these notes come from, and it is the right tool for choosing
  constants: three hand-built cases ranked the growth factor in three different
  orders, so the factor comes from the ensemble's tail rather than from any
  individual run.

  It is emphatically not where the policy is *validated*. The wall above — a
  regression of exactly the kind this revision exists to remove — survived four
  hundred simulated meshes untouched and was caught by the first real one, in
  the first minute of looking at its output. The reason is instructive: the
  generator only ever produced meshes whose answer retained 15-59% of the input,
  so no simulated descent ever came near the share the wall was built from. **A
  generator cannot surprise you with a case it does not generate**, and the
  cases it omits are, reliably, the ones a policy is about to be wrong on. The
  too-tight budget is the standing example: it is the state that walks a search
  into the share region, it was absent from the ensemble, and adding it
  afterwards is worth much less than the real mesh was, because its face curve
  near the input is a guess. Simulate to choose a constant; run a real mesh
  before believing anything.

- **Adding a flag to an in-tree binary turns `test:in-image` red until the image
  is rebuilt, and that is correct behaviour.** That job runs the repo's tests
  inside the published `edge` image, so the *repo's* wrapper is compared against
  the *image's* binaries -- which is the drift check ADR 0005 wants, and it
  duly reports a wrapper that has run ahead of the binary it wraps
  (`test_no_flag_the_wrapper_offers_has_gone_away`, one failure, naming the
  flags). The per-binary skip does not help here: it covers a whole tool the
  image does not carry yet, not a new flag on a tool it does. Do not widen it to
  cover this -- tolerating "the wrapper offers something the binary does not" is
  exactly the condition the test exists to catch. The C++ and the Python ship
  together; the job goes green when `edge` carries both.

- **No CI coverage for the C++.** The only CI job with binaries is
  `test:in-image`, which runs inside a published image and so would not carry a
  newly added tool until after it ships. The binary carries a `--self-test`
  against a generated mesh with a known analytic surface instead, and the
  empirical claim (faces in/out, wall clock, measured max/mean/RMS at two or
  three budgets, over real finished runs) is evidence for the merge request, not
  a test.

## Implementation notes

What Decision 4 left to the implementation, settled empirically rather than
architecturally:

- **Deviation is a very flat power law of the threshold, and the search fits its
  exponent.** The seed is dimensionally right -- a quadric error is a squared
  distance weighted by the area it is measured over, so `budget^2 * mean face
  area` -- but the threshold gates the cheapest *remaining* collapse while the
  budget bounds the deviation accumulated over every collapse composed so far,
  and nothing relates the two in closed form. Measured on the refined meshes this
  pipeline produces, `deviation ~ threshold^0.1`: dividing the threshold by eight
  takes a fifth off the deviation, and reaching a budget half the seed's
  deviation needs the threshold to move by three orders of magnitude. A fixed
  bracketing factor cannot cross that in the rounds available, and the first
  attempt at this -- bracketing by eight -- spent four rounds on three real
  meshes descending 2202, 2171, 1690, 1179 microns against a 1000 micron budget
  and wrote all three inputs through untouched. So each probe is instead placed
  where a two-point fit says the budget lives: a secant step on the power law
  when unbracketed, the same step confined to the bracket -- regula falsi -- once
  one exists, and bisection whenever the fit points at an endpoint and would
  stall. The exponent is a prior of `0.1` for one round and measured from then
  on, which matters because it is not a constant of the problem: on the
  self-test sphere it is `0.38`.
- **The deviation is a staircase, and every other note here is a consequence of
  that.** The reported max is the distance from the single worst *original*
  vertex to the decimated surface, so it holds flat across a wide band of
  thresholds and then steps when a new vertex becomes the worst. Five separate
  runs over one mesh measured `987.9` microns at face counts from 5.8 to 6.4
  million. Two rounds landing on one tread therefore fit a slope of about zero,
  and a slope of about zero licenses a step of many orders of magnitude:
  unguarded, two of three real meshes walked the threshold to the bottom of its
  range on the first refit and spent a round establishing that a mesh nothing
  had been collapsed from was within budget.

  The trust region did not actually stop that, which is worth stating plainly
  because this ADR previously implied it had. A real 11,133,709-face run
  (`PHerc0009Cr5`, budget 0.1, 8 rounds, 5805 s) spent its third round at
  `q=1.48e-10` on a candidate of **11,133,057 faces** — 99.994% of the input,
  the most expensive measurement available and the least informative — and the
  region *permitted* it, because the region is built from the last stride and
  the stride before had been 1429x. A guard that widens in proportion to the
  last jump is no guard at the moment a big jump has just happened.

  **What holds the unbracketed descent is a bound on price, not on
  log-distance.** A probe may not be predicted to leave more than two and a half
  times the last candidate's faces standing, nor more than nine tenths of the
  input's, the prediction coming from a log-log fit of face count against
  threshold over the last two rounds. Two properties earn it the job the first
  attempt's trust region held. It bounds the actual harm rather than a proxy for
  it: the candidate at the bottom of an over-long step is a pass-through, which
  is the most expensive mesh there is to measure and the least informative thing
  to learn, and it is unreachable now by construction. And the curve it is built
  from does not degenerate — two rounds that measured one deviation still
  collapsed different numbers of edges, so the face-count fit stays informative
  exactly where the deviation fit stops being. Geometric growth also bounds the
  whole descent at `g/(g-1)` of its final round and the overshoot past the
  answer at `g`.

  `g` is 2.5, chosen from the ensemble rather than from any single run because
  three hand-built cases ranked candidate factors in three different orders. It
  was re-checked against the *measured* cost model above, since the first sweep
  counted candidate faces only — which makes an extra cheap round look free when
  it really costs 13% of the input. Under the real model the choice is close to
  flat from 2.5 to 8 (medians within 10%, means within 2%), and 2.5 still has
  the best median, the best p90 and the best delivered mesh, so it stands. Read
  the flatness as the ensemble losing resolution there, not as licence to tune
  the constant further on this evidence.

  Measured against that same `PHerc0009Cr5` run, same object and same budget:
  **6 rounds and 14.76 Mface-units against 8 and 39.82**, for a result 7% larger
  (5,770,619 faces at a measured 0.0938, against 5,391,525 at 0.0896). The
  descent is where the saving comes from -- 0.13M, 0.33M, 0.89M, 2.26M, where
  the old run's first three rounds alone were 0.13M, 3.03M and 11.13M. The two
  inputs differ by 896 faces (the mesh still on disk is that run's own near
  pass-through output, not the refine mesh it read), so the treads are not
  identical and the faces-kept figure is approximate; the round count and the
  bill are not.

  The first attempt instead held each step inside the log-distance between the
  two rounds it was fitted from. That is recorded here because its failure is
  instructive rather than merely superseded: the region is built from the last
  two probes, so as they converge it converges with them, and a search that
  lands twice on one tread clamps itself to the distance between two probes that
  are already close. Simulated over four hundred randomized staircase meshes, 24
  of them never bracketed inside the round cap and delivered the input
  unchanged — the region ratcheting the stride down toward zero while the fit,
  reading a flat tread as "nearly arrived", asked for inches. The handoff from
  the first cluster batch reported the mild form of this (a descent clamped to
  decade steps, costing about two rounds); the severe form only appears when the
  budget sits just above a tread the search has already reached.

- **A cost bound is a bias against a first-choice probe and must never become a
  wall.** The share of the input a probe may leave standing is the half of the
  bound that does not scale with the last round, so a descent that reaches it
  asymptotes into it: nine tenths, then a twentieth more, then a hundredth,
  each costing a full measurement. That is the trust region's own failure in a
  new costume, and it was reproduced on the first real mesh this was run
  against — a 300,000-face one whose budget was too tight, where rounds 4
  through 7 moved the candidate 251,750 -> 265,684 -> 269,450 -> 270,030 faces
  against a cap of 270,000.

  So when the bound can no longer buy at least a quarter more faces than the
  round just finished, the probe stops bargaining and goes to the bottom of the
  range. The only question left at that point is whether *anything* below is
  feasible, one probe settles it, and by then that probe costs barely more than
  the crawling one it replaces, because the crawl has already carried the
  candidate to within a fraction of the input. On that mesh it turned 8 rounds
  into 5, for the same — correct — pass-through answer.

- **A repeated deviation is not a datum but the absence of one, and the crawl
  it causes is arithmetically predictable.** `TEST_FLIP_FIX_1` crossed from
  3.0e-04 to 4.6e-08 in one correction and then walked 4.6e-08 -> 3.6e-08 ->
  2.8e-08 -> 2.2e-08 -> 1.7e-08 -> 1.3e-08: six rounds to move 3.5x, at step
  ratios of 0.783, 0.778, 0.786, 0.773, 0.765. That near-constant 0.777 is the
  mechanism's fingerprint. Consecutive rounds on one tread measure *equal*
  deviations, so the log-log slope is exactly zero, `exponent()` rejects it and
  returns the 0.1 prior — and a prior of 0.1 against a deviation 2.5% over
  budget predicts a step of `(1/1.025)^10 = 0.781`. Observed 0.777. The fit is
  not converging; it is reading a flat as "nearly arrived" and asking for
  inches, forever, at a full measurement each. (2.5% over also puts the budget
  in a gap, so the five-percent test cannot end it either.)

  One percent is a generous tolerance for "the same tread", and it is safe to be
  generous because within a single run the measured max is *exactly*
  reproducible: `AllVertex` samples every vertex of both meshes and the max is
  attained at the worst one, so two probes on a single tread return bit-identical
  numbers. That is why the logs show `0.101311` and `0.14844` repeating verbatim
  rather than approximately. Across runs it is not reproducible -- the collapse
  stops on a time-budgeted batch boundary, so two runs of one mesh build
  slightly different candidates and measure slightly different maxima -- which
  is a caveat on comparing runs, not on spotting a tread inside one.

  Two rounds whose measured maxima agree to within one percent in log space are
  therefore on one tread, which means the last stride changed nothing and the
  fit cannot see the riser. Repeating the fit's advice from there just re-reads
  the same number at full measurement cost. So the stride comes from elsewhere,
  and **which elsewhere depends on whether there is a bracket** — the plateau
  reaches both halves of the probe placement, which is not how this fix was
  first written:

  - *Unbracketed*, descending, the probe takes the whole step the cost bound
    above allows; climbing back toward the budget — where going coarser only
    makes the next round cheaper — it doubles the stride that just failed. This
    is what turns the 24 lost runs above into zero, and it improves the
    delivered mesh generally: over the same four hundred, the mean result went
    from 1.37x the coarsest feasible face count to 1.18x.
  - *Bracketed*, it **bisects**. This is the case that was missed first time
    round, and it is the one that kills runs. A flat fit does not point outside
    the bracket, where the pre-existing fallback would have caught it; it points
    a few percent along, comfortably inside, so regula falsi accepts it and
    re-measures the same tread. `PHerc0006Cr05` read `0.101311` on three
    consecutive rounds inside a bracket already only 0.4 decades wide, crawling
    the threshold by 0.903 a round — seven rounds merely to halve the bracket —
    and was still crawling when the scheduler killed it three hours in. Regula
    falsi degenerating into a crawl is the classic failure of the method and
    bisection is the classic answer, closing the bracket geometrically whatever
    the fit believes.

  Note that `--min-gain` does **not** rescue that run: 34% of the faces were
  still formally winnable, so pricing what was left to win correctly told it to
  keep going. The two mechanisms answer different questions — one decides
  *whether* another round is worth buying, the other decides *where* to spend
  it — and `PHerc0006Cr05` needed the second.

- **The trust region is gone from the bracketed case too, and from the safety
  aim.** Both were noted as wrong when this shipped and both are fixed here.
  Once a bracket exists its two measured ends bound the answer and the price
  better than any trust region can, so regula falsi runs unclamped inside it.
  And clamping the last round's safety aim — a mechanism whose whole purpose is
  to overshoot the budget, held down by one whose purpose is to prevent
  overshoot — reduced one real run's shot from the 1497x it asked for to 1.46x.
  The safety aim now keeps only the guard against buying a pass-through, which
  would deliver nothing either.

- **A bracketed probe is kept off the bracket's ends by a small fixed factor,
  not by a share of its log-width.** The end margin exists so a probe cannot
  re-measure a threshold already measured and stall; a fit pointing past an end
  is replaced by the midpoint, because there is nothing left to interpolate.
  Sizing that margin as a share of the log-width looks natural and is wrong: a
  real bracket can span three orders, where five percent of the log-width is a
  factor of one and a half, and probes sitting comfortably inside get thrown out
  for a midpoint two orders away.
- **The seed carries an empirical prefactor of a hundred**, calibrated on the
  sphere. It puts the seed within one step of the answer there. The search exists
  because the seed is a guess, so being wrong here costs rounds and never
  correctness.

  On real meshes it is wrong by a lot: 230 to 3,900 times high across the first
  cluster batch, and up to 16,700 locally. It costs less than that sounds,
  because the refit crosses most of it in one round and because a wildly high
  threshold collapses almost everything, making that first round the cheapest of
  the search. The formula's mesh term earns little too — seeds spanned 1.4x
  across those meshes while the answers spanned 33x. And the scaling is not the
  square the dimensional argument gives: fitting fourteen runs across budgets of
  0.1, 0.15 and 0.2 gives `threshold ~ 484 * budget^9.74`, the exponent being
  the inverse of the `deviation ~ threshold^0.1` above, which is what one should
  expect once the threshold is understood to gate the cheapest *remaining*
  collapse while the budget bounds deviation accumulated over all of them
  composed.

  That exponent is a property of these meshes and not of the problem — the
  self-test sphere's is 0.38, or `threshold ~ budget^2.6` — so it is not baked
  in.

  **What the logs rule out is a near-constant seed, which is what this is.**
  Across 17 runs at one fixed budget the converged thresholds span 255x (35x
  among the runs that delivered), while the seed itself spans 1.42x — 2.21e-04
  to 3.15e-04 — and overshoots by between 370x and 95,000x in *every single
  run*. A quantity that varies by less than half an order cannot track one that
  varies by two and a half, so recalibrating the prefactor is not the fix.

  Note what this does **not** establish. It says nothing about whether some
  formula over mesh features could work; the shipped seed barely tracks face
  count and nothing else, so the experiment that would test a real formula has
  not been run. That question is open, not closed.

  What is known to be predictive is the same object's own previous run. Reruns
  of one fragment span 8.8x across all seven, and 3.6x across the four that
  delivered — roughly an order of magnitude tighter than the between-object
  spread either way you count it. So `--quadric-seed` is the lever available
  today, and the loop closes through the report: `search.quadric_error` in
  `decimate_report.json` is the threshold a run converged at, and it is what to
  pass back on the next run of the *same* object. That the residual is still
  several-fold is exactly why the flag seeds the search rather than replacing it
  — `--quadric-error` would forfeit the measured guarantee for a number good to
  within a factor of four.

  **Do not calibrate a seed on delivered runs only.** Three of the five largest
  seed-to-answer ratios in the batch belong to runs that were killed rather than
  delivered, so conditioning on success drops cases the seed was among its worst
  on. The bias is real and it is weaker than all-five; it is the same selection
  effect that makes converged runs look like they converge high — a high
  threshold means a small candidate means cheap rounds means the run finishes
  first — and the honest spread is the 255x over everything, not the 35x over
  the survivors.

  It buys more than it first appeared to. Simulated against the candidate-only
  cost model, a good seed cut the median round count from six to four but not
  the measurement bill, the rounds it removes being the cheap ones at the top of
  the descent — and that conclusion was an artefact of the model. Once a round
  carries a floor of ~13% of the input (Decision 4), removing rounds *is* the
  saving. Fed its own converged threshold, a real run went from 7 rounds and
  790k face-units to **3 rounds and 429k, 54.7 s to 23.5 s**, and delivered a
  slightly coarser mesh (155,926 faces against 158,176). Where a cost model and
  a stopwatch disagree, believe the stopwatch.

  It remains a per-object number: passing one object's converged threshold as
  another's seed is worse than the formula, since the formula is at least wrong
  in a direction the refit crosses cheaply.
- **The last round aims at half the budget when nothing feasible has been found
  yet.** Every round is a full decimate-and-measure over the whole mesh, so a run
  that ends just outside the budget has spent its entire cost to write the input
  through and deliver nothing. Aiming inside leaves the fit room to be wrong and
  still land a mesh; the faces that caution costs are worth less than the mesh.
- **A feasible result within five percent of the budget ends the search** — when
  it can. The remaining rounds would pay full measurement cost for the last few
  percent of a budget. Stopping there forgoes coarseness, never the bound.

- **That test is unreachable whenever the budget lands in a gap, so what is left
  to win is priced instead.** The staircase again: the test can only fire if a
  tread happens to sit within five percent under the budget, and when the budget
  falls between two treads no probe can ever satisfy it. The search then runs to
  the round cap at full price for whatever face count is still improvable. In
  the first cluster batch this was the common case, not the corner — six of
  eight running jobs were in it, and the two that finished were the two where a
  tread landed two to three percent under. One of them spent four of its six
  rounds, and 27 of its 31.7 million face-units, measuring two distinct
  deviations.

  The bracket answers the question the budget cannot. Faces fall as the
  threshold rises and the answer lies inside the bracket, so the candidate at
  the **infeasible** end is a hard floor on the coarsest feasible mesh any
  further round can find; the difference between it and the best result so far
  is an upper bound on what is left to win. Below `--min-gain` of the best
  result's faces — a tenth by default — another full measurement of a
  multi-million-face mesh is not worth a few percent of it. Like the test above
  it forgoes coarseness and never the bound, and unlike it, it is always
  reachable.

  Measured on a real 300,000-face mesh with the budget placed in a gap (0.16,
  between treads at 0.14844 and 0.1939), against the same binary with the flag
  switched off: 7 rounds and 790k face-units against 10 rounds — the cap — and
  1.25M, for a result 1.5% larger. That is the whole trade, in one controlled
  pair: 37% of the measurement bill for 1.5% of the coarseness. `--max-rounds` is not a substitute: lowering it truncates the
  searches that are converging usefully, which is the opposite of what this
  needs to do.
- **The downward walk stops at the first round that collapses nothing**, rather
  than walking toward a threshold whose deviation is zero. The measurement is
  sampled in single precision, so two copies of one mesh measure a deviation of
  about `1e-7` of their own scale rather than zero: a budget under that floor is
  never satisfiable, and searching for it would spend every round proving so.
  Stopping there reaches Decision 4's answer for that case -- the cheapest
  collapse already exceeds the budget, write the input through, exit 0 -- in two
  rounds.
- **That floor test catches only the sub-noise budget, and the *ordinary*
  too-tight budget is read off `haveBest` instead.** The test is "collapsed
  nothing and is still over budget", which needs the `1e-7` floor above to be
  over the budget too. A budget merely below the cheapest collapse's deviation
  -- microns where the cheapest collapse costs millimetres, the common way to
  ask for too much -- never trips it: probes above the staircase step collapse
  something and blow the budget, probes below collapse nothing and measure the
  floor, which for any ordinary budget is *feasible*. So the search keeps a
  best, removes no faces, and would otherwise be reported as having run out of
  rounds and told to raise `--max-rounds` -- advice that is both wrong and
  useless, since every extra round re-measures a threshold that cannot help.
  `haveBest` with nothing removed is what marks it: every threshold cheap enough
  to stay inside the budget was also too cheap to collapse an edge. Both cases
  get Decision 4's answer, and only a search that never found a feasible round
  is told to raise the cap. The bracket closing to inside the endpoint margin
  ends such a search early, because `nextProbe` can then only re-measure it.

The "peak memory is two copies" of Decision 4 holds *between* rounds, by a route
the decision did not name: the coarsest feasible candidate is written to the
output file as soon as it wins rather than kept in memory, so no third vcglib
mesh is ever live.

It does not hold *during* that write. `write()` builds a complete
`educelab::Mesh3f` of the candidate before serializing it, so a winning round
peaks at the pristine input, the candidate, and a third copy of the candidate in
another representation. It is the smallest of the three — positions and indices,
none of the quadric, adjacency or normal components that dominate a `DVertex` or
a `DFace` — but on a candidate near the input's size it is not nothing, and it
lands at the moment the other two are already at their largest. Streaming the
write would remove it and wants a libcore that can serialize from a callback.
Cost-bounding the probes (above) attacks the same peak from the other side, by
making a near-input candidate something the search no longer buys.
