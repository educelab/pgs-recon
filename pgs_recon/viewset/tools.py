"""Argv builders for ``pgs-localize-joint``, the C++ half of the view set.

The wrapper convention of :mod:`pgs_recon.openmvg` and :mod:`pgs_recon.openmvs`
(ADR 0005): one function per invocation, ``None`` means *omit the flag* so the
binary's own default wins, and every call ends in :func:`toolchain.run`, which
records it. This module is that binary's complete flag surface; when
``dependencies/utilities/src/localize_joint.cpp`` changes, the two are edited
together.

The binary has three subcommands, one per thing that has to happen in-process
with openMVG, OpenCV and the renderer:

- ``position-map`` -- render the mesh from the overhead pose once, keeping only
  the world position per pixel. Every view's overhead matches lift through it.
- ``view`` -- localize one hand-held image: overhead prior, then render-and-match
  against the IR reference mesh. Emits the one-view calibration and the inlier
  correspondences the adjustment consumes.
- ``adjust`` -- the joint adjustment: one shared intrinsic, every pose, mesh
  points fixed, tie points free, plus the focal cost curve.
- ``qa`` -- how far each view's image sits from where its calibration puts
  it, against the other views, the overhead, and the reference texture.

**Provisional.** The flag names are the ones ADR 0013 implies; nothing has been
run through them yet.
"""
from pathlib import Path
from typing import Optional, Sequence

from pgs_recon.toolchain import resolve_exe, run

#: The installed name, beside ``pgs-localize`` in ``<prefix>/bin``.
BINARY = 'pgs-localize-joint'


def _optional(*flags) -> list:
    """``[flag, value]`` for each pair whose value is not ``None``."""
    command = []
    for flag, value in flags:
        if value is not None:
            command += [flag, value]
    return command


def position_map(calibration: Path, mesh: Path, output: Path,
                 sfm_transform: Optional[Path] = None,
                 threads: Optional[int] = None) -> None:
    """Render ``mesh`` from the pose in ``calibration``; write positions only.

    ``output`` is a ``.npy`` of H x W x 3 float32, NaN where no surface was hit
    (the renderer's own convention, ``localize_render.hpp``).
    """
    command = [resolve_exe(BINARY), 'position-map',
               '--calibration', calibration, '--mesh', mesh, '--output', output]
    command += _optional(('--sfm-transform', sfm_transform),
                         ('--threads', threads))
    run(command)


def localize_view(image: Path, mesh: Path, overhead_image: Path,
                  overhead_calibration: Path, overhead_position_map: Path,
                  camera: Path, output_calibration: Path,
                  output_correspondences: Path, report: Path,
                  mask: Optional[Path] = None,
                  output_camera: Optional[Path] = None,
                  qa_render: Optional[Path] = None,
                  prior_scales: Optional[str] = None,
                  heldout_split: Optional[int] = None,
                  seed: Optional[int] = None,
                  threads: Optional[int] = None) -> None:
    """Localize one hand-held ``image`` against the IR reference ``mesh``.

    ``camera`` supplies the starting K and distortion -- from EXIF, a board
    calibration, or a previous ``adjust`` -- and is required: a K resected from
    one near-planar view is degenerate (ADR 0011). Any pose in it is ignored;
    the prior comes from the overhead image.

    There is no ``sfm_transform``: the position map and ``mesh`` are already in
    the mesh's frame, so every 3D point the view sees is too.

    ``mask`` blanks query pixels that must not match -- the sample square
    (ADR 0013 Decision 5). ``prior_scales`` is the comma-separated ladder of
    query downscales tried against the overhead (``"1,0.5,0.25,0.125"``).
    """
    command = [resolve_exe(BINARY), 'view',
               '--image', image, '--mesh', mesh,
               '--overhead-image', overhead_image,
               '--overhead-calibration', overhead_calibration,
               '--overhead-position-map', overhead_position_map,
               '--camera', camera,
               '--output-calibration', output_calibration,
               '--output-correspondences', output_correspondences,
               '--report', report]
    command += _optional(('--mask', mask),
                         ('--output-camera', output_camera),
                         ('--qa-render', qa_render),
                         ('--prior-scales', prior_scales),
                         ('--heldout-split', heldout_split),
                         ('--seed', seed),
                         ('--threads', threads))
    run(command)


def adjust(calibrations: Sequence[Path], correspondences: Sequence[Path],
           output_calibrations: Sequence[Path], output_scene: Path,
           output_camera: Path, report: Path,
           tie_scenes: Sequence[Path] = (), camera: Optional[Path] = None,
           fix_intrinsic: bool = False, focal_sweep: Optional[str] = None,
           divergence_px: Optional[float] = None,
           threads: Optional[int] = None) -> None:
    """Jointly adjust every localized view under one shared intrinsic.

    ``calibrations[i]``, ``correspondences[i]`` and ``output_calibrations[i]``
    are the same view's; the binary pairs them by position and refuses unequal
    counts. Each output is that view's one-view calibration at its adjusted pose
    -- or at its mesh-only pose, when the two diverge past the gate
    (Decision 5); the report says which. ``tie_scenes`` are the groups' own SfM
    solves, whose tracks become free tie points; a view is matched to its
    tie-scene view by image filename.

    ``fix_intrinsic`` holds K at ``camera`` -- for a board-calibrated camera,
    where the self-calibration would only add noise. ``focal_sweep`` is
    ``"lo:hi:steps"`` in pixels; unset, the binary picks a range around the
    starting focal. The sweep always runs: observability is measured on every
    run, never assumed (Decision 4) -- and when it says the views do not
    determine the focal, no adjusted K is emitted: every view keeps its
    mesh-only calibration, and the report's ``intrinsic_emitted`` is
    ``"start"``.

    ``divergence_px`` is how much worse a view's anchors may reproject under
    the joint solve than under its own before it is emitted mesh-only
    (Decision 5).
    """
    counts = {len(calibrations), len(correspondences), len(output_calibrations)}
    if len(counts) != 1:
        raise ValueError(f'{len(calibrations)} calibrations, '
                         f'{len(correspondences)} correspondence files and '
                         f'{len(output_calibrations)} outputs: one each per view')
    command = [resolve_exe(BINARY), 'adjust']
    for cal, corr, out in zip(calibrations, correspondences,
                              output_calibrations):
        command += ['--calibration', cal, '--correspondences', corr,
                    '--output-calibration', out]
    for scene in tie_scenes:
        command += ['--tie-scene', scene]
    command += ['--output-scene', output_scene,
                '--output-camera', output_camera, '--report', report]
    command += _optional(('--camera', camera), ('--focal-sweep', focal_sweep),
                         ('--divergence-px', divergence_px),
                         ('--threads', threads))
    if fix_intrinsic:
        command.append('--fix-intrinsic')
    run(command)


def qa(calibrations: Sequence[Path], mesh: Path, report: Path,
       overhead_calibration: Optional[Path] = None,
       overhead_image: Optional[Path] = None,
       reference_mesh: Optional[Path] = None,
       threads: Optional[int] = None) -> None:
    """Measure each view's residual misregistration (ADR 0013 Decision 6).

    Each comparison warps one image into another view through ``mesh`` and
    measures the remaining shift, tile by tile, in the target view's pixels.
    The report's ``pairs`` compare the views with each other -- independent of
    the overhead; ``overhead`` compares each with the overhead image -- partly
    independent, since both rest on ``mesh``; ``circular`` compares each with
    ``reference_mesh``, the texture it was registered to, and is never graded.
    """
    command = [resolve_exe(BINARY), 'qa']
    for cal in calibrations:
        command += ['--calibration', cal]
    command += ['--mesh', mesh, '--report', report]
    command += _optional(('--overhead-calibration', overhead_calibration),
                         ('--overhead-image', overhead_image),
                         ('--reference-mesh', reference_mesh),
                         ('--threads', threads))
    run(command)
