"""What every artifact of a view-set run is called.

Pure functions over the output root, in the convention of
:mod:`pgs_recon.layout`: no filesystem access, every argument a ``Path`` (or a
view's stem), and these names are only ever written -- a resumed run finds an
existing artifact through the manifest, never by rebuilding a name.

A view-set run has its own output root rather than writing into the
reconstruction it reads: the reconstruction is an input here, and an optional
component that wrote into it would be a second owner of a directory
``pgs-recon`` owns.

Per-view artifacts are keyed by the view's **stem** -- its image filename
without extension -- because that is what a person looking for one view's result
has in hand. Two images with the same stem in one set are refused upstream
(:func:`pgs_recon.viewset.inputs.collect_images`).
"""
from pathlib import Path


def manifest(output: Path) -> Path:
    """The run's record: provenance, per-view status, what was skipped."""
    return output / 'pgs-localize-set.json'


# --- The reference: what every view is matched against (Decision 2) --------


def reference_dir(output: Path) -> Path:
    return output / 'reference'


def ir_reference_mesh(output: Path) -> Path:
    """The mesh capture-retextured from the rig's IR940 capture, when this run
    builds it rather than being handed one."""
    return reference_dir(output) / 'ir_reference.obj'


def starting_camera(output: Path) -> Path:
    """The starting K, as a flat camera file, when one is derived from the
    images' EXIF rather than handed in."""
    return reference_dir(output) / 'starting_camera.txt'


def overhead_position_map(output: Path) -> Path:
    """World position per overhead pixel, rendered once from the overhead pose
    and reused to lift every view's overhead matches (Decision 2, item 2)."""
    return reference_dir(output) / 'overhead_position.npy'


# --- Hand-held structure (Decision 3) --------------------------------------


def sfm_dir(output: Path) -> Path:
    """The hand-held set's own OpenMVG import, regions and matches."""
    return output / 'sfm'


def matches_dir(output: Path) -> Path:
    return sfm_dir(output) / 'matches_dir'


def putative_matches(output: Path) -> Path:
    return matches_dir(output) / 'matches.bin'


def filtered_matches(output: Path) -> Path:
    return matches_dir(output) / 'matches_filtered.bin'


def surviving_pairs(output: Path) -> Path:
    """The view pairs that survived geometric filtering: the match graph
    :mod:`.groups` splits."""
    return matches_dir(output) / 'surviving_pairs.txt'


def group_dir(output: Path, index: int) -> Path:
    """One connected group's tie points, triangulated from its views'
    mesh-only poses (``sfm_data.bin``) inside it."""
    return sfm_dir(output) / f'group_{index:03d}'


def group_scene(output: Path, index: int) -> Path:
    """The imported scene cut down to one group's views, ids unchanged."""
    return group_dir(output, index) / 'group_scene.json'


def group_pairs(output: Path, index: int) -> Path:
    """The surviving pairs between one group's views."""
    return group_dir(output, index) / 'pairs.txt'


def group_putative_matches(output: Path, index: int) -> Path:
    """One group's matches. In the shared matches directory, not the group's:
    OpenMVG's matching tools load regions from the directory their output
    file is in."""
    return matches_dir(output) / f'group_{index:03d}_matches.bin'


def group_filtered_matches(output: Path, index: int) -> Path:
    return matches_dir(output) / f'group_{index:03d}_matches_filtered.bin'


# --- Per view ---------------------------------------------------------------


def view_dir(output: Path, stem: str) -> Path:
    return output / 'views' / stem


def view_calibration(output: Path, stem: str) -> Path:
    """The view localized against the mesh alone, before the joint adjustment.

    Kept, not overwritten by :func:`view_adjusted_calibration`: Decision 5 emits
    this pose instead when the two diverge, and measures the divergence from it.
    """
    return view_dir(output, stem) / f'{stem}_mesh_calibration.json'


def view_adjusted_calibration(output: Path, stem: str) -> Path:
    """The deliverable calibration -- the ``*_calibration.json`` of CONTEXT.md's
    *Calibration*, read by ``pgs-retexture --calibration``. The joint pose, or
    the mesh-only one for a view the adjustment flagged as divergent."""
    return view_dir(output, stem) / f'{stem}_calibration.json'


def view_camera(output: Path, stem: str) -> Path:
    """The same pose and K as a flat camera file (``localize_camera.hpp``)."""
    return view_dir(output, stem) / f'{stem}_camera.txt'


def view_correspondences(output: Path, stem: str) -> Path:
    """The view's mesh-anchored 2D<->3D inliers, the joint adjustment's input."""
    return view_dir(output, stem) / f'{stem}_correspondences.npy'


def view_report(output: Path, stem: str) -> Path:
    return view_dir(output, stem) / f'{stem}_report.json'


def view_qa_render(output: Path, stem: str) -> Path:
    return view_dir(output, stem) / f'{stem}_qa.png'


def view_textured_mesh(output: Path, stem: str) -> Path:
    """The deliverable: the mesh textured from this one view (ADR 0002)."""
    return view_dir(output, stem) / 'textured' / f'{stem}.obj'


# --- Joint adjustment (Decision 4) ------------------------------------------


def joint_dir(output: Path) -> Path:
    return output / 'joint'


def joint_scene(output: Path) -> Path:
    """Every localized view at its adjusted pose, sharing one intrinsic."""
    return joint_dir(output) / 'sfm_data.json'


def joint_camera(output: Path) -> Path:
    """The shared K and distortion as a flat camera file."""
    return joint_dir(output) / 'camera.txt'


def joint_report(output: Path) -> Path:
    """The focal cost curve, per-view divergence, and what each gate said."""
    return joint_dir(output) / 'report.json'


# --- QA (Decision 6) --------------------------------------------------------


def qa_report(output: Path) -> Path:
    """Every view's measured misregistration, labelled for independence."""
    return output / 'qa' / 'report.json'
