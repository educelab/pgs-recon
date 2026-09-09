"""One function per binary of the MVS half of the pipeline.

Same contract as :mod:`pgs_recon.openmvg`: one Python call, one binary
invocation, no ``prefix`` or ``metadata`` parameters, no naming (`ADR 0005
<../docs/adr/0005-wrappers-mirror-the-binary.md>`_). None of these returns a
path -- neither OpenMVS nor ``pgs-decimate`` ever chooses a name the caller did
not give it.

Four of the five wrap an OpenMVS binary; :func:`mvs_decimate` wraps our own
``pgs-decimate``, sitting beside the stages it serves as ``mvg_autoscale``
does. Two stages reach it: ``coarsen`` drives it to a face count on the way into
refine (`ADR 0009 <../docs/adr/0009-coarsen-before-refine.md>`_) and
``decimate`` to a measured deviation budget on the way out (`ADR 0008
<../docs/adr/0008-error-bounded-decimation.md>`_). Which target is which is
``run_pipeline``'s business, not the wrapper's.
Everything below about ``-w`` and the archive type is the OpenMVS four's.

Each function mirrors its binary's **own** options group in the binary's own
order, plus the two generic options that are about the run rather than about
photogrammetry (``--archive-type``, ``--max-threads``). Deliberately omitted, and
the only omissions: ``--help``, ``--config-file``, ``--process-priority``,
``--verbosity`` and ``--cuda-device``, which configure the process rather than
the reconstruction, and each app's undocumented "Hidden options" group, which
upstream does not advertise and churns across pins.

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
``-w`` is derived rather than taken as a parameter because it is also the frame
the scene's *image* paths resolve against: ``openMVG2openMVS`` writes them
relative to the scene file's directory and OpenMVS resolves and re-saves them
relative to ``-w`` (``Scene.cpp:144, 267``), so the two agree only while ``-w``
holds the scene. ADR 0005 records what that does and does not constrain.

Auxiliary path arguments (a mask folder, a view-neighbors list) go absolute
instead: OpenMVS runs every path through ``MAKE_PATH_SAFE``, which keeps an
absolute path verbatim and joins a relative one onto ``-w``.
"""
from pathlib import Path

from pgs_recon.toolchain import MVG_BIN, MVS_BIN, resolve_exe, run, work_dir


def _optional(**flags) -> list:
    """``--flag value`` for every option that is not ``None``.

    Keyword name to flag name is a pure underscore-to-dash mapping for every
    option OpenMVS documents, so the eighty-odd flags below translate mechanically
    rather than as an ``if`` apiece.

    Values are passed through as given: a ``bool`` reaches argv as ``0``/``1``
    (OpenMVS declares these as ``value<bool>`` rather than as presence switches)
    because :func:`toolchain._argv_token` narrows every ``int`` subclass at the
    chokepoint that stringifies argv. Keeping that rule in one place is what stops
    this module and :mod:`pgs_recon.openmvg` from drifting apart on it.
    """
    command = []
    for name, value in flags.items():
        if value is None:
            continue
        command.append(f'--{name.replace("_", "-")}')
        command.append(value)
    return command


def _absolute(path: Path = None):
    """An auxiliary path argument, absolute, or ``None`` left alone.

    For the path-valued flags that are not one of the primary artifacts; see the
    module docstring for why those go absolute.
    """
    return None if path is None else Path(path).resolve()


def mvs_densify(scene: Path, output: Path, point_cloud: Path = None,
                mask_path: Path = None, view_neighbors_file: Path = None,
                output_view_neighbors_file: Path = None,
                resolution_level: int = None, max_resolution: int = None,
                min_resolution: int = None, sub_resolution_levels: int = None,
                number_views: int = None, number_views_fuse: int = None,
                ignore_mask_label: int = None, iters: int = None,
                geometric_iters: int = None, estimate_colors: int = None,
                estimate_normals: int = None, estimate_scale: float = None,
                estimate_segmentation: int = None, sub_scene_area: float = None,
                sample_mesh: float = None, fusion_mode: int = None,
                fusion_filter: int = None,
                fusion_depth_diff_threshold: float = None,
                fusion_reprojection_threshold: float = None,
                postprocess_dmaps: int = None, filter_point_cloud: int = None,
                export_number_views: int = None, roi_border: float = None,
                estimate_roi: float = None, crop_to_roi: bool = None,
                up_axis: int = None, remove_dmaps: bool = None,
                tower_mode: int = None, normalize_coordinates: int = None,
                archive_type: int = -1, max_threads: int = None) -> None:
    """Densify a scene's point cloud.

    Writes two files: ``output`` (a scene still holding the *sparse* cloud) and
    the dense cloud beside it as ``output``'s stem with a ``.ply`` suffix, which
    OpenMVS pairs by name rather than by any argument
    (:func:`layout.densify_cloud`). This is the only MVS stage that writes a
    scene at all, and that pairing is why its two outputs are co-located whatever
    ``-w`` would otherwise accept.

    ``ignore_mask_label`` is the label value in each image's mask to exclude;
    ``None`` ignores masks entirely. ``mask_path`` names a folder of
    ``.mask.png`` files for scenes that do not carry their masks inline.
    """
    work = work_dir(scene, output, point_cloud)
    command = [
        resolve_exe('DensifyPointCloud', MVS_BIN),
        '-i', Path(scene).name,
        '-o', Path(output).name,
        '-w', work,
        '--archive-type', archive_type,
    ]
    if point_cloud is not None:
        command.extend(['-p', Path(point_cloud).name])
    if mask_path is not None:
        command.extend(['-m', _absolute(mask_path)])
    command += _optional(
        view_neighbors_file=_absolute(view_neighbors_file),
        output_view_neighbors_file=_absolute(output_view_neighbors_file),
        resolution_level=resolution_level, max_resolution=max_resolution,
        min_resolution=min_resolution,
        sub_resolution_levels=sub_resolution_levels,
        number_views=number_views, number_views_fuse=number_views_fuse,
        ignore_mask_label=ignore_mask_label, iters=iters,
        geometric_iters=geometric_iters, estimate_colors=estimate_colors,
        estimate_normals=estimate_normals, estimate_scale=estimate_scale,
        estimate_segmentation=estimate_segmentation,
        sub_scene_area=sub_scene_area, sample_mesh=sample_mesh,
        fusion_mode=fusion_mode, fusion_filter=fusion_filter,
        fusion_depth_diff_threshold=fusion_depth_diff_threshold,
        fusion_reprojection_threshold=fusion_reprojection_threshold,
        postprocess_dmaps=postprocess_dmaps,
        filter_point_cloud=filter_point_cloud,
        export_number_views=export_number_views, roi_border=roi_border,
        estimate_roi=estimate_roi, crop_to_roi=crop_to_roi, up_axis=up_axis,
        remove_dmaps=remove_dmaps, tower_mode=tower_mode,
        normalize_coordinates=normalize_coordinates, max_threads=max_threads,
    )
    run(command)


def mvs_reconstruct(scene: Path, output: Path, point_cloud: Path = None,
                    min_point_distance: float = None,
                    integrate_only_roi: bool = None,
                    constant_weight: bool = None,
                    free_space_support: bool = False,
                    thickness_factor: float = None,
                    quality_factor: float = None, decimate: float = None,
                    target_face_num: int = None,
                    remove_spurious: float = None, remove_spikes: bool = None,
                    close_holes: int = None, smooth: int = 2,
                    edge_length: float = None, roi_border: float = None,
                    crop_to_roi: bool = None, export_type: str = None,
                    archive_type: int = -1, max_threads: int = None) -> None:
    """Reconstruct a surface from a scene's point cloud.

    ``point_cloud`` must be given whenever densify ran: the scene densify wrote
    holds the sparse cloud, so without ``-p`` this silently meshes that instead
    of the dense one.

    ``free_space_support`` stays a plain ``bool`` rather than joining the
    ``None``-means-omit convention: ``False`` and omitted mean the same thing to
    the binary, and the flag predates this module.
    """
    work = work_dir(scene, output, point_cloud)
    command = [
        resolve_exe('ReconstructMesh', MVS_BIN),
        '-i', Path(scene).name,
        '-o', Path(output).name,
        '-w', work,
        '--archive-type', archive_type,
    ]
    if point_cloud is not None:
        command.extend(['-p', Path(point_cloud).name])
    if free_space_support:
        command.extend(['--free-space-support', '1'])
    command += _optional(
        min_point_distance=min_point_distance,
        integrate_only_roi=integrate_only_roi, constant_weight=constant_weight,
        thickness_factor=thickness_factor, quality_factor=quality_factor,
        decimate=decimate, target_face_num=target_face_num,
        remove_spurious=remove_spurious, remove_spikes=remove_spikes,
        close_holes=close_holes, smooth=smooth, edge_length=edge_length,
        roi_border=roi_border, crop_to_roi=crop_to_roi,
        export_type=export_type, max_threads=max_threads,
    )
    run(command)


def mvs_refine(scene: Path, mesh: Path, output: Path,
               resolution_level: int = None, min_resolution: int = None,
               max_views: int = None, decimate: float = None,
               close_holes: int = None, ensure_edge_size: int = None,
               max_face_area: int = None, scales: int = 3,
               scale_step: float = None, alternate_pair: int = None,
               regularity_weight: float = None,
               rigidity_elasticity_ratio: float = None,
               gradient_step: float = None, planar_vertex_ratio: float = None,
               reduce_memory: int = None, export_type: str = None,
               archive_type: int = -1, max_threads: int = None) -> None:
    """Refine a reconstructed mesh against the scene's images.

    The memory hog of the pipeline, and the reason a run can be split into jobs
    (`ADR 0004 <../docs/adr/0004-staged-resumable-runs.md>`_). At the pinned
    revision it was also the wall-clock hog, in mesh *preparation* rather
    than in the optimization: before refining, it decimates by CGAL
    Garland-Heckbert edge collapse, single-threaded and silent
    (``Mesh.cpp:925-945``, called from ``SceneRefine.cpp:508-535``), at a cost
    that varies 135x between meshes at an identical target. The ``coarsen``
    stage does that reduction with ``pgs-decimate`` instead and turns this pass
    off (`ADR 0009 <../docs/adr/0009-coarsen-before-refine.md>`_), which is why
    ``run_pipeline`` passes ``decimate=1`` and ``ensure_edge_size=2`` whenever
    that stage is in the shape.

    That pass costs by how far it decimates, not by input size. ``decimate=None``
    leaves OpenMVS at ``0`` (auto), whose target is the mesh's median *projected*
    face area over ``max_face_area``, floored at a tenth of the input. So
    ``max_face_area`` is the denominator: raising it decimates harder, and it
    bounds subdivision rather than this. ``decimate=1`` skips decimation, and with
    ``ensure_edge_size=1`` skips the edge-size pass after it too, via the same
    guard (``SceneRefine.cpp:556``) -- which is why the two travel together;
    ``ensure_edge_size=0`` skips only that pass. None of these are defaults
    here, because each changes the mesh that comes out: the coupling is
    ``run_pipeline``'s, being an invariant of ours rather than the binary's.
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
    command += _optional(
        decimate=decimate, resolution_level=resolution_level,
        min_resolution=min_resolution, scales=scales, scale_step=scale_step,
        max_views=max_views, close_holes=close_holes,
        ensure_edge_size=ensure_edge_size, max_face_area=max_face_area,
        alternate_pair=alternate_pair, regularity_weight=regularity_weight,
        rigidity_elasticity_ratio=rigidity_elasticity_ratio,
        gradient_step=gradient_step, planar_vertex_ratio=planar_vertex_ratio,
        reduce_memory=reduce_memory, export_type=export_type,
        max_threads=max_threads,
    )
    run(command)


def mvs_decimate(mesh: Path, output: Path, report: Path = None,
                 max_error: float = None, max_faces: int = None,
                 quadric_error: float = None, prefer: str = None,
                 preserve_boundary: bool = None,
                 preserve_topology: bool = None, normal_check: bool = None,
                 optimal_placement: bool = None,
                 quality_threshold: float = None, max_rounds: int = None,
                 quadric_seed: float = None, min_gain: float = None,
                 samples_per_face: int = None,
                 curvature_samples: bool = None,
                 progress: bool = None) -> None:
    """Coarsen a mesh as far as a measured deviation budget allows.

    Ours, not OpenMVS's (`ADR 0008
    <../docs/adr/0008-error-bounded-decimation.md>`_), and the only decimation
    here stating a geometric bound: ``max_error`` is a distance in the solved
    frame's units, *measured* on the result rather than predicted. The
    ``decimate=`` parameters elsewhere in this module are face fractions.

    At least one of ``max_error``, ``max_faces`` and ``quadric_error`` is
    required; ``prefer`` decides when the first two disagree. ``quadric_error``
    is vcglib's unitless threshold and turns the search off -- an escape hatch.
    So does a ``max_faces`` given alone, there being nothing to search for
    when the target is a count: that is the ``coarsen`` stage's call, which is
    why it is affordable where the CGAL pass it replaces was not. The
    measurement runs on every round regardless of the target, so it cannot be
    switched off, only turned down (``samples_per_face``,
    ``curvature_samples``).

    ``quadric_seed`` is the same unitless threshold as ``quadric_error`` but
    only *starts* the search, so the measured guarantee survives it.

    ``min_gain`` prices what is left to win: a round costs ten samples per face
    of the candidate, so a bracket that can still remove only a few percent of
    them is not worth another one.

    Paths go absolute and no working directory is derived. The output still
    belongs in ``mvs/``, but that is ``run_pipeline``'s business, because
    ``TextureMesh`` addresses its mesh by basename.
    """
    command = [
        resolve_exe('pgs-decimate', MVG_BIN),
        '-i', _absolute(mesh),
        '-o', _absolute(output),
    ]
    command += _optional(
        report=_absolute(report), max_error=max_error, max_faces=max_faces,
        quadric_error=quadric_error, prefer=prefer,
        preserve_boundary=preserve_boundary,
        preserve_topology=preserve_topology, normal_check=normal_check,
        optimal_placement=optimal_placement,
        quality_threshold=quality_threshold, max_rounds=max_rounds,
        quadric_seed=quadric_seed, min_gain=min_gain,
        samples_per_face=samples_per_face,
        curvature_samples=curvature_samples, progress=progress,
    )
    run(command)


def mvs_texture(scene: Path, mesh: Path, output: Path, export_type: str = None,
                decimate: float = None, close_holes: int = None,
                resolution_level: int = None, min_resolution: int = None,
                outlier_threshold: float = None,
                cost_smoothness_ratio: float = None,
                virtual_face_images: int = None,
                global_seam_leveling: int = None,
                local_seam_leveling: int = None,
                texture_size_multiple: int = None, empty_color: int = None,
                sharpness_weight: float = None,
                orthographic_image_resolution: int = None,
                ignore_mask_label: int = None, max_texture_size: int = 0,
                archive_type: int = -1, max_threads: int = None) -> None:
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
    command += _optional(
        resolution_level=resolution_level, empty_color=empty_color,
        global_seam_leveling=global_seam_leveling,
        local_seam_leveling=local_seam_leveling, decimate=decimate,
        close_holes=close_holes, min_resolution=min_resolution,
        outlier_threshold=outlier_threshold,
        cost_smoothness_ratio=cost_smoothness_ratio,
        virtual_face_images=virtual_face_images,
        texture_size_multiple=texture_size_multiple,
        sharpness_weight=sharpness_weight,
        orthographic_image_resolution=orthographic_image_resolution,
        ignore_mask_label=ignore_mask_label, max_threads=max_threads,
    )
    run(command, cwd=work)
