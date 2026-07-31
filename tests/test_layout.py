"""``layout``: the exact name every stage writes, for every pipeline shape.

Pinned to today's chained names on purpose. The interface refactor (ADR 0005)
moves naming out of the wrappers *without* changing a single filename, and this
file is what makes that provable: it fails if any name moves. When ADR 0006
lands, these expectations become the diff -- ``scene_dense_refine.ply`` ->
``refine_mesh.ply`` -- and the shape permutations below collapse, since a name
will no longer depend on which other stages ran.

The chained names are the interesting ones because they *accumulate*: what
``refine`` writes depends on whether ``densify`` ran, four stages upstream. So
the shapes are walked end to end (:func:`chain`) rather than each function being
poked in isolation -- the same role rebinding ``run_pipeline`` performs.

Pure path arithmetic, no filesystem: nothing here creates or reads a file.
"""
import unittest
from pathlib import Path

from pgs_recon import layout

ROOT = Path('/recon/out')
NAME = 'scroll'


def chain(robust=False, autoscale=False, densify=False, refine=False,
          direct=False, file_type='obj') -> dict:
    """Walk one pipeline shape, returning role -> path at the end of each stage.

    Mirrors ``run_pipeline``'s bindings, including the rebinding: ``sfm`` is
    produced by ``import`` and then rebound by ``sfm``/``robust``/``autoscale``,
    and ``mesh`` by ``reconstruct`` and then ``refine``. Keys are
    ``<stage>.<role>`` so a permutation reads as the history it is.
    """
    out = {}
    sfm = out['import.sfm'] = layout.imported_sfm(ROOT)
    out['matches.matches'] = layout.matches(ROOT)
    out['filter.matches_filtered'] = layout.matches_filtered(
        out['matches.matches'])

    if direct:
        # The 'direct' method triangulates the imported scene in place of a solve
        sfm = out['sfm.sfm'] = layout.robust_sfm(ROOT, sfm)
    else:
        sfm = out['sfm.sfm'] = layout.solved_sfm(ROOT)
    if robust:
        sfm = out['robust.sfm'] = layout.robust_sfm(ROOT, sfm)
    if autoscale:
        sfm = out['autoscale.sfm'] = layout.autoscale_sfm(ROOT, sfm)
    out['colorize.colorized'] = layout.colorize_sfm(sfm)

    scene = out['convert.scene'] = layout.convert_scene(ROOT)
    if densify:
        out['densify.cloud'] = layout.densify_cloud(scene)
        scene = out['densify.scene'] = layout.densify_scene(scene)
    mesh = out['reconstruct.mesh'] = layout.reconstruct_mesh(scene)
    if refine:
        mesh = out['refine.mesh'] = layout.refine_mesh(scene)
    out['texture.mesh'] = layout.final_mesh(ROOT, NAME, file_type)
    # Named so an unused-variable reading of the walk is wrong: the final mesh is
    # textured from whatever 'mesh' ends up bound to.
    out['texture.input'] = mesh
    return out


class TestDirectories(unittest.TestCase):

    def test_the_two_halves_of_a_run(self):
        self.assertEqual(ROOT / 'mvg', layout.mvg_dir(ROOT))
        self.assertEqual(ROOT / 'mvs', layout.mvs_dir(ROOT))

    def test_mvg_subdirectories(self):
        self.assertEqual(ROOT / 'mvg/matches_dir', layout.matches_dir(ROOT))
        self.assertEqual(ROOT / 'mvg/recon_dir', layout.recon_dir(ROOT))

    def test_undistorted_images_are_shared_by_every_mvs_stage(self):
        self.assertEqual(ROOT / 'mvs/undistorted_images',
                         layout.undistorted_images(ROOT))

    def test_directories_are_outermost_first(self):
        # main() mkdirs these in order; a child before its parent would need
        # parents=True to paper over the ordering.
        dirs = layout.directories(ROOT)
        self.assertEqual((ROOT, ROOT / 'mvg', ROOT / 'mvg/matches_dir',
                          ROOT / 'mvg/recon_dir', ROOT / 'mvs'), dirs)


class TestRunRecords(unittest.TestCase):

    def test_manifest(self):
        self.assertEqual(ROOT / 'metadata.json', layout.manifest(ROOT))

    def test_config_is_named_for_the_reconstruction(self):
        self.assertEqual(ROOT / 'scroll_recon_config.txt',
                         layout.config(ROOT, NAME))


class TestFrozenNames(unittest.TestCase):
    """Names that are not ours to choose, and so do not change with the shape."""

    def test_imported_scene(self):
        self.assertEqual(ROOT / 'mvg/sfm_data.json', layout.imported_sfm(ROOT))

    def test_solved_scene(self):
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data.bin',
                         layout.solved_sfm(ROOT))

    def test_matches(self):
        self.assertEqual(ROOT / 'mvg/matches_dir/matches.bin',
                         layout.matches(ROOT))

    def test_filtered_matches_land_beside_their_input(self):
        # openMVG_main_SfM takes this by basename against the regions dir, so it
        # has to be a sibling of the matches it filtered.
        self.assertEqual(ROOT / 'mvg/matches_dir/matches_filtered.bin',
                         layout.matches_filtered(layout.matches(ROOT)))

    def test_filtered_matches_keep_the_input_suffix(self):
        # The suffix is the input's, not a hardcoded .bin: OpenMVG picks the
        # match file's format from its extension.
        self.assertEqual(Path('/m/matches_filtered.txt'),
                         layout.matches_filtered(Path('/m/matches.txt')))

    def test_view_pairs(self):
        self.assertEqual(ROOT / 'mvg/matches_dir/pgs_view_pairs.txt',
                         layout.view_pairs(ROOT))

    def test_landmarks(self):
        self.assertEqual(ROOT / 'mvg/recon_dir/landmarks.ply',
                         layout.landmarks(ROOT))
        self.assertEqual(ROOT / 'mvg/recon_dir/landmarks_scaled.ply',
                         layout.scaled_landmarks(ROOT))

    def test_final_mesh_is_the_run_name(self):
        self.assertEqual(ROOT / 'mvs/scroll.obj',
                         layout.final_mesh(ROOT, NAME, 'obj'))
        self.assertEqual(ROOT / 'mvs/scroll.ply',
                         layout.final_mesh(ROOT, NAME, 'ply'))

    def test_final_mesh_extension_is_lowercased(self):
        # --file-type is already lowercased by the parser; a library caller's is
        # not, and OpenMVS matches the export type case-sensitively.
        self.assertEqual(ROOT / 'mvs/scroll.obj',
                         layout.final_mesh(ROOT, NAME, 'OBJ'))

    def test_final_mesh_does_not_change_with_the_shape(self):
        shapes = [chain(), chain(densify=True), chain(refine=True),
                  chain(robust=True, autoscale=True, densify=True, refine=True)]
        self.assertEqual({ROOT / 'mvs/scroll.obj'},
                         {s['texture.mesh'] for s in shapes})


class TestMvgChain(unittest.TestCase):
    """The SfM chain: every rebinding of the ``sfm`` role adds a suffix."""

    def test_plain_solve(self):
        c = chain()
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data.bin', c['sfm.sfm'])
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data_colorized.ply',
                         c['colorize.colorized'])

    def test_robust_triangulation(self):
        c = chain(robust=True)
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data_structured.bin',
                         c['robust.sfm'])

    def test_autoscale_after_a_plain_solve(self):
        c = chain(autoscale=True)
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data_scaled.bin',
                         c['autoscale.sfm'])

    def test_autoscale_after_robust_carries_both_suffixes(self):
        c = chain(robust=True, autoscale=True)
        self.assertEqual(
            ROOT / 'mvg/recon_dir/sfm_data_structured_scaled.bin',
            c['autoscale.sfm'])

    def test_colorize_follows_the_last_rebinding(self):
        # The ADR 0006 table's example: the name records the whole history.
        c = chain(robust=True, autoscale=True)
        self.assertEqual(
            ROOT / 'mvg/recon_dir/sfm_data_structured_scaled_colorized.ply',
            c['colorize.colorized'])

    def test_colorize_lands_beside_its_input(self):
        # A leaf, so it is the one MVG output whose directory follows the input
        # rather than being recon_dir by construction.
        self.assertEqual(Path('/elsewhere/sfm_data_colorized.ply'),
                         layout.colorize_sfm(Path('/elsewhere/sfm_data.bin')))

    def test_direct_triangulates_the_imported_scene_into_recon_dir(self):
        # The trap robust_sfm's directory argument exists for: the input is in
        # mvg/, the output belongs with the solve in mvg/recon_dir/.
        c = chain(direct=True)
        self.assertEqual(ROOT / 'mvg/sfm_data.json', c['import.sfm'])
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data_structured.bin',
                         c['sfm.sfm'])


class TestMvsChain(unittest.TestCase):
    """The mesh chain, whose names encode whether densify ran."""

    def test_convert(self):
        self.assertEqual(ROOT / 'mvs/scene.mvs', layout.convert_scene(ROOT))

    def test_no_densify_no_refine(self):
        c = chain()
        self.assertEqual(ROOT / 'mvs/scene.mvs', c['convert.scene'])
        self.assertEqual(ROOT / 'mvs/scene_mesh.ply', c['reconstruct.mesh'])
        self.assertEqual(c['reconstruct.mesh'], c['texture.input'])
        self.assertNotIn('densify.cloud', c)

    def test_refine_without_densify(self):
        c = chain(refine=True)
        self.assertEqual(ROOT / 'mvs/scene_refine.ply', c['refine.mesh'])

    def test_densify_writes_a_scene_and_a_cloud(self):
        c = chain(densify=True)
        self.assertEqual(ROOT / 'mvs/scene_dense.mvs', c['densify.scene'])
        self.assertEqual(ROOT / 'mvs/scene_dense.ply', c['densify.cloud'])

    def test_the_dense_cloud_differs_from_its_scene_only_by_extension(self):
        # OpenMVS writes the pair together and finds the cloud back from the
        # scene's name, so this is a requirement rather than a coincidence.
        c = chain(densify=True)
        self.assertEqual(c['densify.scene'].stem, c['densify.cloud'].stem)
        self.assertEqual('.mvs', c['densify.scene'].suffix)
        self.assertEqual('.ply', c['densify.cloud'].suffix)

    def test_densify_renames_the_whole_mesh_chain(self):
        # The operational constraint submit_recon_pipeline.sh documents: adding
        # --mvs-densify on a later job moves artifacts two stages downstream.
        c = chain(densify=True, refine=True)
        self.assertEqual(ROOT / 'mvs/scene_dense_mesh.ply',
                         c['reconstruct.mesh'])
        self.assertEqual(ROOT / 'mvs/scene_dense_refine.ply', c['refine.mesh'])

    def test_refine_is_named_from_the_scene_not_the_mesh(self):
        # ADR 0003: refine's -i is an interface scene, never the mesh-bearing
        # one. Deriving from the mesh would give scene_dense_mesh_refine.ply.
        c = chain(densify=True, refine=True)
        self.assertEqual(c['densify.scene'].parent, c['refine.mesh'].parent)
        self.assertEqual(
            c['densify.scene'].stem + '_refine.ply', c['refine.mesh'].name)

    def test_every_mvs_artifact_stays_in_the_working_dir(self):
        # Every MVS stage runs with -w mvs/ and refers to its inputs by
        # basename, so an artifact outside that directory is unreachable.
        c = chain(densify=True, refine=True)
        for role in ('convert.scene', 'densify.scene', 'densify.cloud',
                     'reconstruct.mesh', 'refine.mesh', 'texture.mesh'):
            self.assertEqual(ROOT / 'mvs', c[role].parent, role)

    def test_no_two_stages_share_a_name(self):
        # What keeps a rebinding visible to the planner: if reconstruct and
        # refine wrote the same name, refine would clobber the artifact and the
        # 'mesh' role's move would be undetectable.
        c = chain(robust=True, autoscale=True, densify=True, refine=True)
        produced = [v for k, v in c.items() if k != 'texture.input']
        self.assertEqual(len(produced), len(set(produced)))


if __name__ == '__main__':
    unittest.main()
