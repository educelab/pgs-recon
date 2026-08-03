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

The argument convention: **a function takes the output root, and nothing else**
-- plus, for the two names that are not the pipeline's to choose, the run name
and file type. An intermediate is called ``<stage>_<role>.<ext>``: the stage
that wrote it and the role it fills (`ADR 0006
<../docs/adr/0006-stage-named-artifacts.md>`_). Names no longer chain off their
input's stem, so no name encodes which other stages ran, and nothing here needs
to be handed an input path to work out an output's name.

Whatever the scheme, one rule holds: **these names are only ever written.** A
resumed job locates an existing artifact through the manifest's stage records,
never by rebuilding a name -- which is what makes renaming safe on output
directories that already exist.

Not every name is ours to choose. Frozen here, and marked as such below: the
names OpenMVG picks for its own outputs, the already-role-named ``matches*``,
``pgs-global-scaler``'s ``landmarks*``, and the deliverable ``mvs/<name>.<ext>``,
which is a user-facing contract. ``densify`` is the one stage whose pair of
outputs shares a stem for the same reason -- see :func:`densify_cloud`.

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
    """The run's manifest: what it has finished, and with what arguments.

    Named for the tool that owns the directory. ``metadata.json``, which this
    replaced in 1.8, is also what an EduceLab **scan** directory calls its
    descriptor (see :mod:`pgs_recon.pgs_data`) -- an input format we do not own,
    so the one filename meant two unrelated things.
    """
    return output / 'pgs-recon.json'


def legacy_manifest(output: Path) -> Path:
    """Where runs before 1.8 wrote the manifest.

    Read, never written: :func:`stages.find_manifest` falls back to this so an
    output directory built by an earlier version still resumes, and the next
    write lands on :func:`manifest`.
    """
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
    """The solved scene, whichever engine solved it.

    **Frozen:** ``openMVG_main_SfM`` takes an output *directory* and names this
    itself; ``mvg_sfm`` returns the path because the binary, not the caller,
    chose it. The ``direct`` method, which triangulates known poses instead of
    solving, writes here too: it is the same stage producing the same role, and
    only one of the two branches ever runs.
    """
    return recon_dir(output) / 'sfm_data.bin'


def robust_sfm(output: Path) -> Path:
    """Robustly re-triangulated scene."""
    return recon_dir(output) / 'robust_sfm.bin'


def autoscale_sfm(output: Path) -> Path:
    """Scene rescaled to physical units by ``pgs-global-scaler``."""
    return recon_dir(output) / 'autoscale_sfm.bin'


def landmarks(output: Path) -> Path:
    """Markers ``pgs-global-scaler`` detected, in the solved frame.
    **Frozen:** already named for its role."""
    return recon_dir(output) / 'landmarks.ply'


def scaled_landmarks(output: Path) -> Path:
    """The same markers after rescaling -- the check that autoscale did what
    was asked. **Frozen:** already named for its role."""
    return recon_dir(output) / 'landmarks_scaled.ply'


def colorize_sfm(output: Path) -> Path:
    """Sparse cloud coloured from the images. A leaf: nothing downstream
    consumes it.

    Lands in ``recon_dir`` beside the scene it is coloured from, which every
    shape puts there.
    """
    return recon_dir(output) / 'colorize_sfm.ply'


# --- MVS artifacts ---------------------------------------------------------


def convert_scene(output: Path) -> Path:
    """The interface scene the MVG->MVS conversion writes."""
    return mvs_dir(output) / 'convert_scene.mvs'


def densify_scene(output: Path) -> Path:
    """The scene densify writes.

    Densify is the only MVS stage that writes a scene at all -- and the scene
    it writes still holds the *sparse* cloud, the dense one going to
    :func:`densify_cloud`. Reconstruct, refine and texture emit only geometry.

    The stem is the stage alone, with no role: see :func:`densify_cloud`.
    """
    return mvs_dir(output) / 'densify.mvs'


def densify_cloud(output: Path) -> Path:
    """The dense cloud, the pair to :func:`densify_scene`.

    **Half frozen.** ``DensifyPointCloud`` takes one ``-o`` and writes both
    files from its stem -- the scene as ``.mvs``, the cloud as ``.ply`` -- so
    the two names cannot differ, and the stage cannot spell out two roles.
    Hence ``densify.mvs``/``densify.ply``: the stem names the stage and the
    suffix carries the role, rather than a cloud being named
    ``densify_scene.ply``, which is the class of mislabel ADR 0006 exists to
    end.
    """
    return densify_scene(output).with_suffix('.ply')


def reconstruct_mesh(output: Path) -> Path:
    """The surface reconstructed from a scene's cloud."""
    return mvs_dir(output) / 'reconstruct_mesh.ply'


def refine_mesh(output: Path) -> Path:
    """The refined mesh.

    Named for what it *is*, not for the scene it was refined against -- which
    is what made the old ``scene_dense_refine.ply`` a mesh carrying a scene's
    name.
    """
    return mvs_dir(output) / 'refine_mesh.ply'


def final_mesh(output: Path, name: str, file_type: str) -> Path:
    """The deliverable: the textured mesh.

    **Frozen:** a user-facing contract. ADR 0004 depends on it (a stale final
    mesh sits at exactly the expected filename) and ``recon_dir.py``
    reconstructs it from ``name`` + ``file_type`` for manifests predating stage
    records.
    """
    return mvs_dir(output) / f'{name}.{file_type.lower()}'
