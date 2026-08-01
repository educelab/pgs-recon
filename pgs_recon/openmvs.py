"""One function per OpenMVS binary.

Same contract as :mod:`pgs_recon.openmvg`: one Python call, one binary
invocation, no ``prefix`` or ``metadata`` parameters, no naming (`ADR 0005
<../docs/adr/0005-wrappers-mirror-the-binary.md>`_). None of these four returns a
path -- OpenMVS never chooses a name the caller did not give it.

Two invariants of `ADR 0003 <../docs/adr/0003-portable-mvs-intermediates.md>`_ run
through this module, and both are deliberately *parameters* rather than
hardcoded:

* ``archive_type`` defaults to ``-1`` and is therefore **always emitted**. 0003's
  concern is the flag being implicit -- an upstream default change would silently
  reintroduce Boost archives, which OOM when read by a differently built
  OpenMVS -- not its being overridable. Do not "tidy" this into the
  ``None``-means-omit convention the other flags follow.
* ``point_cloud`` is nullable, because no dense cloud is a legitimate shape
  rather than a mistake. Passing it whenever densify ran is a *pipeline*
  obligation: without ``-p``, ``ReconstructMesh`` silently builds from the sparse
  cloud and the densification is wasted.

Every stage addresses its scene and geometry by basename against a working
directory, so those files must be co-located; :func:`toolchain.work_dir` derives
that directory from the artifacts themselves and refuses when they disagree.
"""
from pathlib import Path

from pgs_recon.toolchain import MVS_BIN, resolve_exe, run, work_dir


def mvs_densify(scene: Path, output: Path, resolution_level: int = None,
                ignore_mask_label: int = None, archive_type: int = -1) -> None:
    """Densify a scene's point cloud.

    Writes two files: ``output`` (a scene still holding the *sparse* cloud) and
    the dense cloud beside it as ``output``'s stem with a ``.ply`` suffix, which
    OpenMVS pairs by name rather than by any argument
    (:func:`layout.densify_cloud`). This is the only MVS stage that writes a
    scene at all.

    ``ignore_mask_label`` is the label value in each image's mask to exclude;
    ``None`` ignores masks entirely.
    """
    work = work_dir(scene, output)
    command = [
        resolve_exe('DensifyPointCloud', MVS_BIN),
        '-i', Path(scene).name,
        '-o', Path(output).name,
        '-w', work,
        '--archive-type', archive_type,
    ]
    if resolution_level is not None:
        command.extend(['--resolution-level', resolution_level])
    if ignore_mask_label is not None:
        command.extend(['--ignore-mask-label', ignore_mask_label])
    run(command)


def mvs_reconstruct(scene: Path, output: Path, point_cloud: Path = None,
                    free_space_support: bool = False, smooth: int = 2,
                    archive_type: int = -1) -> None:
    """Reconstruct a surface from a scene's point cloud.

    ``point_cloud`` must be given whenever densify ran: the scene densify wrote
    holds the sparse cloud, so without ``-p`` this silently meshes that instead
    of the dense one.
    """
    work = work_dir(scene, output, point_cloud)
    command = [
        resolve_exe('ReconstructMesh', MVS_BIN),
        '-i', Path(scene).name,
        '-o', Path(output).name,
        '-w', work,
        '--archive-type', archive_type,
    ]
    if smooth is not None:
        command.extend(['--smooth', smooth])
    if point_cloud is not None:
        command.extend(['-p', Path(point_cloud).name])
    if free_space_support:
        command.extend(['--free-space-support', '1'])
    run(command)


def mvs_refine(scene: Path, mesh: Path, output: Path, decimate: float = None,
               resolution_level: int = None, min_resolution: int = None,
               scales: int = 3, scale_step: float = None,
               archive_type: int = -1) -> None:
    """Refine a reconstructed mesh against the scene's images.

    The memory hog of the pipeline, and the reason a run can be split into jobs
    (`ADR 0004 <../docs/adr/0004-staged-resumable-runs.md>`_).
    """
    work = work_dir(scene, mesh, output)
    command = [
        resolve_exe('RefineMesh', MVS_BIN),
        '-i', Path(scene).name,
        '-m', Path(mesh).name,
        '-o', Path(output).name,
        '-w', work,
        '--archive-type', archive_type,
    ]
    if decimate is not None:
        command.extend(['--decimate', decimate])
    if resolution_level is not None:
        command.extend(['--resolution-level', resolution_level])
    if min_resolution is not None:
        command.extend(['--min-resolution', min_resolution])
    if scales is not None:
        command.extend(['--scales', scales])
    if scale_step is not None:
        command.extend(['--scale-step', scale_step])
    run(command)


def mvs_texture(scene: Path, mesh: Path, output: Path, export_type: str = None,
                resolution_level: int = None, max_texture_size: int = 0,
                empty_color: int = None, global_seam_leveling: int = None,
                local_seam_leveling: int = None,
                archive_type: int = -1) -> None:
    """Texture a mesh from the scene's images.

    ``export_type`` defaults to ``output``'s extension, which is what OpenMVS
    requires of the pair; pass it only to override. ``mesh`` must already live in
    the working directory -- a caller texturing an externally produced mesh
    stages it there first (see retexture's ``ensure_ply_mesh``).

    Seam leveling and ``empty_color`` are left at OpenMVS defaults unless set.
    Passing ``*_seam_leveling=0`` disables the per-patch brightness normalization
    that hides seams, preserving the source radiometry -- which matters when
    texturing a scientific modality where pixel intensities are the signal.
    """
    work = work_dir(scene, mesh, output)
    if export_type is None:
        export_type = Path(output).suffix.lstrip('.')
    command = [
        resolve_exe('TextureMesh', MVS_BIN),
        '-i', Path(scene).name,
        '-m', Path(mesh).name,
        '-o', Path(output).name,
        '--export-type', export_type.lower(),
        '-w', work,
        '--archive-type', archive_type,
        '--max-texture-size', max_texture_size,
    ]
    if resolution_level is not None:
        command.extend(['--resolution-level', resolution_level])
    if empty_color is not None:
        command.extend(['--empty-color', empty_color])
    if global_seam_leveling is not None:
        command.extend(['--global-seam-leveling', global_seam_leveling])
    if local_seam_leveling is not None:
        command.extend(['--local-seam-leveling', local_seam_leveling])
    run(command, cwd=work)
