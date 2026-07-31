"""Where a reconstruction's artifacts go, and what they are called.

Pure functions over paths: no filesystem access, no arguments object, no
``paths`` dict. Naming lives here rather than in the wrappers so that a wrapper
is nothing but a translation from a Python call to a binary invocation (`ADR
0005 <../docs/adr/0005-wrappers-mirror-the-binary.md>`_), and so the whole
vocabulary of an output directory can be read -- and renamed -- in one place.

Every argument is a ``Path``, not a ``PathLike`` -- deliberately narrower than
``toolchain``, which takes strings because a prefix can arrive from the
environment. Nothing reaches this module from outside the process: a caller
holds paths already, and accepting strings would only hide a missing conversion.

The argument convention: **a function takes the output root, unless its name
chains off an input artifact, in which case it takes that artifact.** The
chaining is the *current* scheme, not the intended one: today an intermediate's
name accumulates the pipeline's history (``scene.mvs`` -> ``scene_dense.mvs`` ->
``scene_dense_mesh.ply``), and `ADR 0006
<../docs/adr/0006-stage-named-artifacts.md>`_ replaces it with
``<stage>_<role>.<ext>``. This module reproduces the chained names *exactly* so
that the interface change can be verified against a byte-identical artifact
tree; when 0006 lands, the bodies here change and every function ends up taking
just the output root.

Whatever the scheme, one rule holds: **these names are only ever written.** A
resumed job locates an existing artifact through the manifest's stage records,
never by rebuilding a name -- which is what makes renaming safe on output
directories that already exist.

Not every name is ours to choose. Frozen here, and marked as such below: the
names OpenMVG picks for its own outputs, the already-role-named ``matches*``,
``pgs-global-scaler``'s ``landmarks*``, and the deliverable ``mvs/<name>.<ext>``,
which is a user-facing contract.

``pgs-retexture`` writes into an existing recon's ``mvg/``/``mvs/`` under its own
stem-prefixed convention (see CONTEXT.md); that layout is the app's own and is
not modelled here.
"""
from pathlib import Path
from typing import Tuple

# --- Directories -----------------------------------------------------------


def mvg_dir(output: Path) -> Path:
    """The OpenMVG half of the run."""
    return output / 'mvg'


def matches_dir(output: Path) -> Path:
    """Feature regions and match files. OpenMVG names the per-image regions
    inside it after the images, so the directory is the artifact."""
    return mvg_dir(output) / 'matches_dir'


def recon_dir(output: Path) -> Path:
    """Where the SfM solve and everything derived from it land."""
    return mvg_dir(output) / 'recon_dir'


def mvs_dir(output: Path) -> Path:
    """The OpenMVS half of the run, and the working dir every MVS stage runs
    in: geometry travels beside its scene, so an MVS artifact is referenced by
    basename against this directory."""
    return output / 'mvs'


def undistorted_images(output: Path) -> Path:
    """Undistorted images written by the MVG->MVS conversion, shared by every
    MVS stage."""
    return mvs_dir(output) / 'undistorted_images'


def directories(output: Path) -> Tuple[Path, ...]:
    """Every directory a run needs, outermost first, for one mkdir pass."""
    return (output, mvg_dir(output), matches_dir(output), recon_dir(output),
            mvs_dir(output))


# --- Run records -----------------------------------------------------------


def manifest(output: Path) -> Path:
    """The run's ``metadata.json``."""
    return output / 'metadata.json'


def config(output: Path, name: str) -> Path:
    """The effective arguments, in ``-c``-loadable form. One per
    reconstruction, not per job."""
    return output / f'{name}_recon_config.txt'


# --- MVG artifacts ---------------------------------------------------------


def imported_sfm(output: Path) -> Path:
    """The imported scene. **Frozen:** ``init_sfm_generic`` hands OpenMVG a
    directory and OpenMVG picks this name, so renaming it would make the import
    artifact's name depend on which importer ran."""
    return mvg_dir(output) / 'sfm_data.json'


def matches(output: Path) -> Path:
    """Putative matches. **Frozen:** already named for its role."""
    return matches_dir(output) / 'matches.bin'


def matches_filtered(matches_file: Path) -> Path:
    """Geometrically filtered matches, beside their input.

    **Frozen:** already named for its role. Derived from the input rather than
    spelled out because ``openMVG_main_SfM`` takes it by basename, so the two
    spellings would have to agree.
    """
    return matches_file.parent / (matches_file.stem + '_filtered'
                                  + matches_file.suffix)


def view_pairs(output: Path) -> Path:
    """The importer's view pairs file, limiting matching to spatial neighbors.
    Only a grid scan has one."""
    return matches_dir(output) / 'pgs_view_pairs.txt'


def solved_sfm(output: Path) -> Path:
    """The solved scene. **Frozen:** ``openMVG_main_SfM`` takes an output
    *directory* and names this itself; ``mvg_sfm`` returns the path because the
    binary, not the caller, chose it."""
    return recon_dir(output) / 'sfm_data.bin'


def robust_sfm(output: Path, sfm: Path) -> Path:
    """Robustly re-triangulated scene.

    Note the directory: this lands in ``recon_dir`` even when the input does
    not. The ``direct`` reconstruction method triangulates the *imported* scene
    from ``mvg/``, and its output still belongs with the solve.
    """
    return recon_dir(output) / (sfm.stem + '_structured.bin')


def autoscale_sfm(output: Path, sfm: Path) -> Path:
    """Scene rescaled to physical units by ``pgs-global-scaler``."""
    return recon_dir(output) / (sfm.stem + '_scaled.bin')


def landmarks(output: Path) -> Path:
    """Markers ``pgs-global-scaler`` detected, in the solved frame.
    **Frozen:** already named for its role."""
    return recon_dir(output) / 'landmarks.ply'


def scaled_landmarks(output: Path) -> Path:
    """The same markers after rescaling -- the check that autoscale did what
    was asked. **Frozen:** already named for its role."""
    return recon_dir(output) / 'landmarks_scaled.ply'


def colorize_sfm(sfm: Path) -> Path:
    """Sparse cloud coloured from the images, beside its input scene. A leaf:
    nothing downstream consumes it."""
    return sfm.parent / (sfm.stem + '_colorized.ply')


# --- MVS artifacts ---------------------------------------------------------


def convert_scene(output: Path) -> Path:
    """The interface scene the MVG->MVS conversion writes."""
    return mvs_dir(output) / 'scene.mvs'


def densify_scene(scene: Path) -> Path:
    """The scene densify writes beside its input.

    Densify is the only MVS stage that writes a scene at all -- and the scene
    it writes still holds the *sparse* cloud, the dense one going to
    :func:`densify_cloud`. Reconstruct, refine and texture emit only geometry.
    """
    return scene.parent / (scene.stem + '_dense.mvs')


def densify_cloud(scene: Path) -> Path:
    """The dense cloud. Named from the *scene* densify wrote, so the pair
    differs only by extension, which is how OpenMVS expects to find it."""
    return densify_scene(scene).with_suffix('.ply')


def reconstruct_mesh(scene: Path) -> Path:
    """The surface reconstructed from a scene's cloud."""
    return scene.parent / (scene.stem + '_mesh.ply')


def refine_mesh(scene: Path) -> Path:
    """The refined mesh -- named from the *scene* it was refined against, not
    from the mesh it refines. That is what makes ``scene_dense_refine.ply`` a
    mesh with a scene's name, the confusion ADR 0006 exists to end."""
    return scene.parent / (scene.stem + '_refine.ply')


def final_mesh(output: Path, name: str, file_type: str) -> Path:
    """The deliverable: the textured mesh.

    **Frozen:** a user-facing contract. ADR 0004 depends on it (a stale final
    mesh sits at exactly the expected filename) and ``recon_dir.py``
    reconstructs it from ``name`` + ``file_type`` for manifests predating stage
    records.
    """
    return mvs_dir(output) / f'{name}.{file_type.lower()}'
