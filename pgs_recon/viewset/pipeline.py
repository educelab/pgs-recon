"""The phases of a view-set run, in order (ADR 0013).

Each phase is one function taking the plate, the output root and the options,
and returning what the next phase needs. :func:`run` is nothing but their
sequence, so the order is readable in one place, as ``run_pipeline`` is for
``pgs-recon``.

Phases that are not built yet raise :class:`NotImplementedError` naming the
ADR 0013 task that builds them, so a run stops at the first gap with a pointer
rather than producing a partial result that looks whole.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

from pgs_recon import openmvg, toolchain
from pgs_recon.toolchain import cam_db, run as run_tool
from pgs_recon.viewset import groups as groups_mod
from pgs_recon.viewset import layout, manifest, tools
from pgs_recon.viewset.inputs import PlateInputs, Reference, Strategy

logger = logging.getLogger(__name__)

#: OpenMVS ``--empty-color`` for faces no rig camera saw. Magenta, because the
#: IR reference texture is grey and so a chromatic texel can only be empty --
#: which is what lets the renderer's validity mask (ADR 0013 Decision 2) find
#: them *before* the render is converted to grey, where magenta would be an
#: ordinary mid-grey. 0xFF00FF reads the same in RGB and BGR order.
EMPTY_COLOR = 0xFF00FF

#: OpenMVG's ``-c`` for ``pinhole_radial_k3``: the lens has visible edge
#: distortion, so pinhole is not an option (ADR 0013 Context).
CAMERA_MODEL_RADIAL3 = 3


class Options(NamedTuple):
    """The run's knobs, as the CLI resolved them."""
    camera: Optional[Path] = None   # starting K: EXIF-derived, board, or prior run
    fix_intrinsic: bool = False     # hold K at ``camera`` (board-calibrated)
    focal_sweep: Optional[str] = None
    mask: Optional[Path] = None     # query pixels never to match (sample square)
    threads: Optional[int] = None
    skip_texture: bool = False


def run(plate: PlateInputs, strategy: Strategy, output: Path,
        opts: Options) -> None:
    meta_path = layout.manifest(output)
    meta = manifest.load(meta_path)
    # Every binary this run executes lands in its own manifest's ``commands``,
    # through the same chokepoint pgs-recon records through.
    toolchain.configure(recorder=toolchain.Recorder(
        meta.setdefault('commands', {})))
    meta['strategy'] = {'reference': strategy.reference.value,
                        'prior': strategy.prior.value,
                        'reasons': list(strategy.reasons)}

    reference = build_ir_reference(plate, strategy, output)
    position_map = render_overhead_position_map(plate, reference, output, opts)
    _check_provenance(meta, plate, reference)
    manifest.write(meta_path, meta)

    imported = import_set(plate, output)
    camera = starting_camera(imported, output, opts)

    # Each view is localized against the mesh alone, needing nothing from the
    # groups -- so it runs first, and a set's views are localized even while
    # the group solve below is unbuilt or fails.
    localized = localize_views(plate, reference, position_map, camera, output,
                               opts, meta, meta_path)

    split = reconstruct_groups(imported, output, opts)
    meta['groups'] = {'groups': [list(g) for g in split.groups],
                      'singletons': list(split.singletons)}
    manifest.write(meta_path, meta)

    adjust_views(localized, split, camera, output, opts)
    if not opts.skip_texture:
        texture_views(plate, localized, output)


# --- Phase 0: the IR reference (Decision 2) ---------------------------------


def build_ir_reference(plate: PlateInputs, strategy: Strategy,
                       output: Path) -> Path:
    """The mesh every view is matched against, building it if need be.

    Built through ``pgs-retexture`` as a subprocess -- apps may not import each
    other, and its command line is its contract.
    """
    if strategy.reference is Reference.PROVIDED:
        return plate.ir_reference_mesh

    target = layout.ir_reference_mesh(output)
    if target.exists():
        logger.info(f'IR reference mesh exists, reusing: {target}')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    command = _retexture() + ['-r', plate.recon_dir, '--mesh', plate.mesh,
                              '--output-mesh', target,
                              '--working-dir', layout.reference_dir(output)]
    if strategy.reference is Reference.RIG_IR940:
        # Capture retexture: every rig camera in the IR940 capture contributes.
        command += ['-i', plate.rig_ir_capture.scan_dir,
                    '--capture', plate.rig_ir_capture.capture,
                    '--empty-color', EMPTY_COLOR]
        if plate.sfm_transform is not None:
            command += ['--sfm-transform', plate.sfm_transform]
    else:
        # Fallback: projective texture from the overhead 940 view. Untextured
        # faces carry no UVs at all here, which the renderer already sees.
        command += ['-i', plate.overhead.image,
                    '--calibration', plate.overhead.calibration]
    run_tool(command)
    return target


def render_overhead_position_map(plate: PlateInputs, reference: Path,
                                 output: Path, opts: Options) -> Path:
    """World position per overhead pixel, rendered once for every view."""
    target = layout.overhead_position_map(output)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tools.position_map(plate.overhead.calibration, reference, target,
                           sfm_transform=plate.sfm_transform,
                           threads=opts.threads)
    return target


def _check_provenance(meta: Dict, plate: PlateInputs, reference: Path) -> None:
    """Record what this run localizes through; invalidate every view if any of
    it changed since the manifest was written (Decision 7)."""
    current = {'mesh': manifest.fingerprint(plate.mesh),
               'reference': manifest.fingerprint(reference),
               'overhead_calibration':
                   manifest.fingerprint(plate.overhead.calibration),
               'overhead_image': manifest.fingerprint(plate.overhead.image)}
    if plate.sfm_transform is not None:
        current['sfm_transform'] = manifest.fingerprint(plate.sfm_transform)
    drift = manifest.provenance_drift(meta.get('provenance', {}), current)
    if meta.get('provenance') and drift:
        logger.warning('Run inputs changed since the last run -- every view '
                       'will be redone: '
                       + ', '.join(f'{k} {v}' for k, v in drift.items()))
        for record in meta['views'].values():
            record['status'] = 'stale-provenance'
    meta['provenance'] = current


# --- Phase 2: hand-held structure (Decision 3) ------------------------------


def import_set(plate: PlateInputs, output: Path) -> Path:
    """Import the hand-held images as an OpenMVG scene, once.

    One shared intrinsic (``-g`` left at upstream's grouping default) in the
    radial-k3 model, with its focal from EXIF and the sensor database. That
    intrinsic is the starting K every view is localized with.
    """
    sfm_dir = layout.sfm_dir(output)
    imported = sfm_dir / 'sfm_data.json'  # OpenMVG picks this name
    if not imported.exists():
        sfm_dir.mkdir(parents=True, exist_ok=True)
        openmvg.init_sfm_generic(_common_dir(plate.images), output_dir=sfm_dir,
                                 cam_db=cam_db(),
                                 camera_model=CAMERA_MODEL_RADIAL3)
    return imported


def starting_camera(imported: Path, output: Path, opts: Options) -> Path:
    """The camera file every view starts from: ``--camera``, or the imported
    scene's EXIF-derived intrinsic written out as one."""
    if opts.camera is not None:
        return opts.camera
    target = layout.starting_camera(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(camera_file_from_scene(imported))
    logger.info(f'Starting K from EXIF: {target}')
    return target


def camera_file_from_scene(sfm_json: Path) -> str:
    """The first intrinsic of an OpenMVG ``sfm_data.json`` as a flat camera
    file (``localize_camera.hpp``): K and radial k1..k3, no pose.

    Refuses a scene with no intrinsic -- what OpenMVG's import leaves when the
    images carry no EXIF focal or the camera is not in the sensor database --
    rather than inventing a focal length.
    """
    data = json.loads(Path(sfm_json).read_text())
    intrinsics = data.get('intrinsics') or []
    if not intrinsics:
        sys.exit(f'{sfm_json} has no intrinsic: the images carry no usable '
                 f'EXIF focal length, or the camera is not in the sensor '
                 f'database. Pass --camera with a starting K.')
    value = intrinsics[0]['value']
    d = value['ptr_wrapper']['data']
    f = d['focal_length']
    cx, cy = d['principal_point']
    k = list(d.get('disto_k3', [0.0, 0.0, 0.0]))
    return (f'fx {f}\nfy {f}\ncx {cx}\ncy {cy}\n'
            f'width {d["width"]}\nheight {d["height"]}\n'
            f'k1 {k[0]}\nk2 {k[1]}\nk3 {k[2]}\n')


def reconstruct_groups(imported: Path, output: Path,
                       opts: Options) -> groups_mod.Groups:
    """Describe and match the hand-held set; split it into groups.

    Returns groups of image *stems*. Solving each group's SfM is the second half
    of this phase and not built yet (:func:`_solve_group`).
    """
    matches_dir = layout.matches_dir(output)
    if not layout.filtered_matches(output).exists():
        matches_dir.mkdir(parents=True, exist_ok=True)
        openmvg.compute_features(imported, output_dir=matches_dir,
                                 method='SIFT', preset='HIGH',
                                 threads=opts.threads)
        openmvg.compute_matches(imported, output=layout.putative_matches(output),
                                method='FASTCASCADEHASHINGL2')
        openmvg.geometric_filter(imported,
                                 matches=layout.putative_matches(output),
                                 output=layout.filtered_matches(output),
                                 output_pairs=layout.surviving_pairs(output))

    files = _view_files(imported)
    stems = [Path(files[i]).stem for i in sorted(files)]
    pairs = [(Path(files[a]).stem, Path(files[b]).stem)
             for a, b in groups_mod.read_pairs(layout.surviving_pairs(output))]
    split = groups_mod.split(stems, pairs)
    logger.info(f'{len(stems)} views: {len(split.groups)} groups '
                f'({", ".join(str(len(g)) for g in split.groups) or "none"}), '
                f'{len(split.singletons)} singletons')

    for index, group in enumerate(split.groups):
        _solve_group(imported, group, index, output, opts)
    return split


def _solve_group(imported: Path, group: Sequence[str], index: int,
                 output: Path, opts: Options) -> Path:
    """Subset the imported scene to ``group`` and solve it with OpenMVG.

    Open question to settle while building it: whether ``openMVG_main_SfM``
    accepts a matches file naming views absent from the scene, or the matches
    must be subset too.
    """
    raise NotImplementedError(
        'Per-group SfM (ADR 0013 task 5): subset the imported scene with '
        'utils.sfm_json, then openmvg.mvg_sfm(engine="INCREMENTALV2") into '
        f'{layout.group_dir(output, index)}')


# --- Phase 3: per-view localization (Decision 2) ----------------------------


def localize_views(plate: PlateInputs, reference: Path, position_map: Path,
                   camera: Path, output: Path, opts: Options, meta: Dict,
                   meta_path: Path) -> List[str]:
    """Localize every view against the mesh alone, resuming where possible.

    One view failing does not stop the others (Decision 6: a view that fails
    is reported and not emitted). Returns the stems that succeeded.
    """
    done = []
    for image in plate.images:
        stem = image.stem
        inputs = {'image': manifest.fingerprint(image),
                  'reference': manifest.fingerprint(reference),
                  'position_map': manifest.fingerprint(position_map),
                  'camera': manifest.fingerprint(camera)}
        record = meta['views'].get(stem)
        reason = manifest.stale_reason(record, inputs)
        if reason is None:
            logger.info(f'{stem}: current, skipping')
            done.append(stem)
            continue
        logger.info(f'{stem}: localizing ({reason})')

        layout.view_dir(output, stem).mkdir(parents=True, exist_ok=True)
        record = {'inputs': inputs, 'status': 'running'}
        meta['views'][stem] = record
        manifest.write(meta_path, meta)
        try:
            tools.localize_view(
                image, reference,
                overhead_image=plate.overhead.image,
                overhead_calibration=plate.overhead.calibration,
                overhead_position_map=position_map,
                camera=camera,
                output_calibration=layout.view_calibration(output, stem),
                output_correspondences=layout.view_correspondences(output,
                                                                   stem),
                report=layout.view_report(output, stem),
                mask=opts.mask,
                output_camera=layout.view_camera(output, stem),
                qa_render=layout.view_qa_render(output, stem),
                threads=opts.threads)
        except Exception as e:  # ToolFailed, or a missing binary
            record['status'] = 'failed'
            record['error'] = str(e)
            logger.error(f'{stem}: {e}')
        else:
            record['status'] = 'done'
            done.append(stem)
        manifest.write(meta_path, meta)
    return done


# --- Phase 4: joint adjustment (Decision 4, 5) ------------------------------


def adjust_views(stems: Sequence[str], split: groups_mod.Groups,
                 camera: Path, output: Path, opts: Options) -> None:
    if not stems:
        sys.exit('No view localized; nothing to adjust.')
    layout.joint_dir(output).mkdir(parents=True, exist_ok=True)
    tools.adjust(
        calibrations=[layout.view_calibration(output, s) for s in stems],
        correspondences=[layout.view_correspondences(output, s) for s in stems],
        output_calibrations=[layout.view_adjusted_calibration(output, s)
                             for s in stems],
        output_scene=layout.joint_scene(output),
        output_camera=layout.joint_camera(output),
        report=layout.joint_report(output),
        tie_scenes=[layout.group_dir(output, i) / 'sfm_data.bin'
                    for i in range(len(split.groups))],
        camera=camera, fix_intrinsic=opts.fix_intrinsic,
        focal_sweep=opts.focal_sweep, threads=opts.threads)


# --- Phase 6: texture each view (Decision 7) --------------------------------


def texture_views(plate: PlateInputs, stems: Sequence[str],
                  output: Path) -> None:
    """One projective-UV mesh per view, through ``pgs-retexture --calibration``.

    The deliverable texture is the hand-held image itself, so the mesh is the
    reconstruction's, not the IR reference.
    """
    by_stem = {p.stem: p for p in plate.images}
    for stem in stems:
        target = layout.view_textured_mesh(output, stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        run_tool(_retexture() + [
            '-i', by_stem[stem],
            '--calibration', layout.view_adjusted_calibration(output, stem),
            '-r', plate.recon_dir, '--mesh', plate.mesh,
            '--output-mesh', target,
            '--working-dir', layout.view_dir(output, stem)])


# --- Helpers ----------------------------------------------------------------


def _retexture() -> list:
    """``pgs-retexture``'s argv prefix, run by this interpreter so it is the
    same install's, wherever the console scripts went."""
    return [sys.executable, '-m', 'pgs_recon.apps.retexture']


def _common_dir(images: Sequence[Path]) -> Path:
    """The one directory the set lives in -- OpenMVG imports a directory."""
    dirs = {p.parent for p in images}
    if len(dirs) != 1:
        raise ValueError(f'hand-held images span {len(dirs)} directories; '
                         f'OpenMVG imports one')
    return dirs.pop()


def _view_files(sfm_json: Path) -> Dict[int, str]:
    """View id -> image filename, from an OpenMVG ``sfm_data.json``."""
    data = json.loads(Path(sfm_json).read_text())
    return {v['key']: v['value']['ptr_wrapper']['data']['filename']
            for v in data['views']}
