# Own decimation, bounded by measured deviation

**Status: accepted.** Specified as issue #21 and implemented alongside it.
Supersedes the draft plan
`docs/plans/qecd-decimation-stage.md`, which is deleted with this ADR: two of its
recommendations are now known wrong (see Decision 2) and one of its code snippets
does not compile against the pinned vcglib. Three details of the search
policy in Decision 4 were settled during implementation; they are recorded at the
end.

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
  threaded and silent, and the pipeline's wall-clock hog at the pinned revision,
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
keeping the **coarsest feasible** result seen rather than the last one tried. Deviation is monotone in the threshold in practice but vcglib guarantees
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
- **An extrapolation may not reach further past the last round than the two
  fitted rounds are apart.** The deviation is not a smooth curve but a
  staircase: the reported max is the distance from the single worst *original*
  vertex to the decimated surface, so it holds flat across a wide band of
  thresholds and then steps when a new vertex becomes the worst. Five separate
  runs over one mesh measured `987.9` microns at face counts from 5.8 to 6.4
  million. A fit taken across two adjacent steps therefore reports a slope near
  zero, and a slope near zero licenses a step of many orders of magnitude:
  unguarded, two of three real meshes walked the threshold to the bottom of its
  range on the first refit and spent a round establishing that a mesh nothing
  had been collapsed from was within budget. Holding the step inside the
  baseline the fit was measured over turned that round from a 1.0002x result
  into a 4.1x one, and the whole run from 68 minutes into 14.
  Known refinement, deferred. The trust region is built from the last two
  probes, so once the search has narrowed those the region narrows with them --
  and that happens routinely, because consecutive rounds landing on one step of
  the staircase measure equal deviations, the fit degenerates to the prior, and
  the probes converge. Two effects follow, both observed. Where a bracket
  already exists the region is redundant, since the bracket bounds the answer
  better than any trust region can. Worse, it also clamps the last round's
  safety aim: a mechanism whose whole purpose is to overshoot the budget, held
  down by one whose purpose is to prevent overshoot. On one real run that
  reduced the safety shot from the 1497x it asked for to 1.46x -- which crossed
  the step anyway and produced a mesh, so the harm is a live risk rather than a
  measured failure. Applying the trust region only while unbracketed, and never
  to the safety aim, is the fix. It is recorded rather than taken because the
  search as it stands is better than or equal to the unguarded one on every mesh
  measured, and a third revision would invalidate that evidence for a gain
  nothing has yet demonstrated.
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
- **The last round aims at half the budget when nothing feasible has been found
  yet.** Every round is a full decimate-and-measure over the whole mesh, so a run
  that ends just outside the budget has spent its entire cost to write the input
  through and deliver nothing. Aiming inside leaves the fit room to be wrong and
  still land a mesh; the faces that caution costs are worth less than the mesh.
- **A feasible result within five percent of the budget ends the search.** The
  remaining rounds would pay full measurement cost for the last few percent of a
  budget. Stopping there forgoes coarseness, never the bound.
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

The "peak memory is two copies" of Decision 4 holds, but by a route the decision
did not name: the coarsest feasible candidate is written to the output file as
soon as it wins rather than kept in memory, so the process holds the pristine
input and one candidate and never a third copy.
