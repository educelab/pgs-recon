"""``layout``: the exact name every stage writes, for every pipeline shape.

These expectations are the record of ADR 0006: an intermediate is
``<stage>_<role>.<ext>``, and a name no longer accumulates the history of the
stages upstream of it (``scene_dense_refine.ply`` -> ``refine_mesh.ply``).

The shapes are still walked end to end (:func:`chain`) rather than each function
being poked in isolation, but for the opposite reason to before: what the walk
now has to show is that no name *moves* when a stage is added or dropped, which
only a permutation can demonstrate. The stage-to-stage threading of paths is
gone with the chaining -- every function takes the output root.

Pure path arithmetic, no filesystem: nothing here creates or reads a file.
"""
import unittest
from pathlib import Path

from pgs_recon import layout

ROOT = Path('/recon/out')
NAME = 'scroll'


def chain(robust=False, autoscale=False, densify=False, refine=False,
          decimate=False, file_type='obj') -> dict:
    """Walk one pipeline shape, returning role -> path at the end of each stage.

    Mirrors ``run_pipeline``'s bindings, including the rebinding: ``sfm`` is
    produced by ``import`` and then rebound by ``sfm``/``robust``/``autoscale``,
    and ``mesh`` by ``reconstruct`` and then ``refine``. Keys are
    ``<stage>.<role>`` so a permutation reads as the history it is.
    """
    out = {}
    out['import.sfm'] = layout.imported_sfm(ROOT)
    out['matches.matches'] = layout.matches(ROOT)
    out['filter.matches_filtered'] = layout.matches_filtered(
        out['matches.matches'])

    # Not a shape parameter any more: the 'direct' method triangulates the
    # imported scene in place of a solve, and writes it where a solve would.
    out['sfm.sfm'] = layout.solved_sfm(ROOT)
    if robust:
        out['robust.sfm'] = layout.robust_sfm(ROOT)
    if autoscale:
        out['autoscale.sfm'] = layout.autoscale_sfm(ROOT)
    out['colorize.colorized'] = layout.colorize_sfm(ROOT)

    out['convert.scene'] = layout.convert_scene(ROOT)
    if densify:
        out['densify.cloud'] = layout.densify_cloud(ROOT)
        out['densify.scene'] = layout.densify_scene(ROOT)
    mesh = out['reconstruct.mesh'] = layout.reconstruct_mesh(ROOT)
    if refine:
        mesh = out['refine.mesh'] = layout.refine_mesh(ROOT)
    if decimate:
        mesh = out['decimate.mesh'] = layout.decimate_mesh(ROOT)
        out['decimate.deviation'] = layout.decimate_report(ROOT)
    out['texture.mesh'] = layout.final_mesh(ROOT, NAME, file_type)
    # Named so an unused-variable reading of the walk is wrong: the final mesh is
    # textured from whatever 'mesh' ends up bound to.
    out['texture.input'] = mesh
    return out


SHAPES = {
    'minimal': chain(),
    'densified': chain(densify=True),
    'refined': chain(refine=True),
    'decimated': chain(decimate=True),
    'everything': chain(robust=True, autoscale=True, densify=True,
                        refine=True, decimate=True),
}


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

    def test_manifest_is_named_for_the_tool_that_owns_the_directory(self):
        # Not metadata.json: that is what an EduceLab *scan* directory calls its
        # descriptor, an input format we do not own.
        self.assertEqual(ROOT / 'pgs-recon.json', layout.manifest(ROOT))

    def test_the_pre_1_8_manifest_name_is_still_spelled_out(self):
        # Read, never written -- stages.find_manifest falls back to it.
        self.assertEqual(ROOT / 'metadata.json', layout.legacy_manifest(ROOT))

    def test_config_is_named_for_the_reconstruction(self):
        self.assertEqual(ROOT / 'scroll_recon_config.txt',
                         layout.config(ROOT, NAME))


class TestFrozenNames(unittest.TestCase):
    """Names that are not ours to choose."""

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


class TestMvgChain(unittest.TestCase):
    """The SfM chain: each rebinding of the ``sfm`` role names its own stage."""

    def test_plain_solve(self):
        c = chain()
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data.bin', c['sfm.sfm'])
        self.assertEqual(ROOT / 'mvg/recon_dir/colorize_sfm.ply',
                         c['colorize.colorized'])

    def test_robust_triangulation(self):
        c = chain(robust=True)
        self.assertEqual(ROOT / 'mvg/recon_dir/robust_sfm.bin', c['robust.sfm'])

    def test_autoscale(self):
        c = chain(autoscale=True)
        self.assertEqual(ROOT / 'mvg/recon_dir/autoscale_sfm.bin',
                         c['autoscale.sfm'])

    def test_autoscale_after_robust_is_the_same_name(self):
        # The ADR 0006 diff: this used to be sfm_data_structured_scaled.bin.
        self.assertEqual(chain(autoscale=True)['autoscale.sfm'],
                         chain(robust=True, autoscale=True)['autoscale.sfm'])

    def test_colorize_does_not_follow_the_last_rebinding(self):
        # It used to: sfm_data_structured_scaled_colorized.ply recorded which
        # stages had run upstream of a leaf that consumed all of them.
        self.assertEqual(
            {ROOT / 'mvg/recon_dir/colorize_sfm.ply'},
            {c['colorize.colorized'] for c in SHAPES.values()})

    def test_direct_writes_the_solve_where_a_solve_goes(self):
        # The 'direct' method is the same stage producing the same role by
        # another means, so it writes solved_sfm, not robust_sfm. It has to:
        # --mvg-robust can be set alongside it, and sharing a name would have
        # robust reading and writing one file.
        self.assertNotEqual(layout.solved_sfm(ROOT), layout.robust_sfm(ROOT))
        self.assertEqual(ROOT / 'mvg/recon_dir/sfm_data.bin',
                         layout.solved_sfm(ROOT))


class TestMvsChain(unittest.TestCase):
    """The mesh chain, whose names no longer encode whether densify ran."""

    def test_convert(self):
        self.assertEqual(ROOT / 'mvs/convert_scene.mvs',
                         layout.convert_scene(ROOT))

    def test_no_densify_no_refine(self):
        c = chain()
        self.assertEqual(ROOT / 'mvs/convert_scene.mvs', c['convert.scene'])
        self.assertEqual(ROOT / 'mvs/reconstruct_mesh.ply', c['reconstruct.mesh'])
        self.assertEqual(c['reconstruct.mesh'], c['texture.input'])
        self.assertNotIn('densify.cloud', c)

    def test_refine(self):
        c = chain(refine=True)
        self.assertEqual(ROOT / 'mvs/refine_mesh.ply', c['refine.mesh'])
        self.assertEqual(c['refine.mesh'], c['texture.input'])

    def test_densify_writes_a_scene_and_a_cloud(self):
        c = chain(densify=True)
        self.assertEqual(ROOT / 'mvs/densify.mvs', c['densify.scene'])
        self.assertEqual(ROOT / 'mvs/densify.ply', c['densify.cloud'])

    def test_the_dense_cloud_differs_from_its_scene_only_by_extension(self):
        # DensifyPointCloud takes one -o and writes both files from its stem, so
        # this is a requirement rather than a coincidence -- and the reason the
        # densify pair is the one stem that names no role.
        c = chain(densify=True)
        self.assertEqual(c['densify.scene'].stem, c['densify.cloud'].stem)
        self.assertEqual('.mvs', c['densify.scene'].suffix)
        self.assertEqual('.ply', c['densify.cloud'].suffix)

    def test_densify_no_longer_renames_the_mesh_chain(self):
        # The operational constraint submit_recon_pipeline.sh documented: adding
        # --mvs-densify on a later job used to move artifacts two stages
        # downstream (scene_mesh.ply -> scene_dense_mesh.ply).
        plain, dense = chain(refine=True), chain(densify=True, refine=True)
        self.assertEqual(plain['reconstruct.mesh'], dense['reconstruct.mesh'])
        self.assertEqual(plain['refine.mesh'], dense['refine.mesh'])

    def test_refine_is_named_for_the_mesh_it_is(self):
        # ADR 0003: refine's -i is an interface scene, never the mesh-bearing
        # one -- which is why the old name came off the scene's stem and read as
        # a scene. The stage name replaces that.
        c = chain(densify=True, refine=True)
        self.assertEqual(ROOT / 'mvs/refine_mesh.ply', c['refine.mesh'])
        self.assertNotIn('scene', c['refine.mesh'].name)

    def test_decimate(self):
        c = chain(refine=True, decimate=True)
        self.assertEqual(ROOT / 'mvs/decimate_mesh.ply', c['decimate.mesh'])
        self.assertEqual(ROOT / 'mvs/decimate_report.json',
                         c['decimate.deviation'])
        self.assertEqual(c['decimate.mesh'], c['texture.input'])

    def test_decimate_takes_the_mesh_from_whatever_owns_it(self):
        # With refine off it coarsens reconstruct's mesh, under the same name:
        # a stage's output name never records which stages ran before it.
        with_refine = chain(refine=True, decimate=True)
        without = chain(decimate=True)
        self.assertEqual(with_refine['decimate.mesh'],
                         without['decimate.mesh'])
        self.assertEqual(without['decimate.mesh'], without['texture.input'])

    def test_every_mvs_artifact_stays_in_the_working_dir(self):
        # Every MVS stage runs with -w mvs/ and names inputs by basename. The
        # decimated mesh is here because TextureMesh addresses it that way.
        c = chain(densify=True, refine=True, decimate=True)
        for role in ('convert.scene', 'densify.scene', 'densify.cloud',
                     'reconstruct.mesh', 'refine.mesh', 'decimate.mesh',
                     'decimate.deviation', 'texture.mesh'):
            self.assertEqual(ROOT / 'mvs', c[role].parent, role)


class TestNamesAreShapeIndependent(unittest.TestCase):
    """What ADR 0006 bought: the property the permutations exist to prove."""

    def test_a_role_a_stage_produces_has_one_name_across_every_shape(self):
        # Before, half of these moved with the shape -- which is what made
        # adding --mvs-densify on a later job invalidate the stages downstream.
        for key in ('import.sfm', 'sfm.sfm', 'robust.sfm', 'autoscale.sfm',
                    'colorize.colorized', 'convert.scene', 'densify.scene',
                    'densify.cloud', 'reconstruct.mesh', 'refine.mesh',
                    'decimate.mesh', 'decimate.deviation', 'texture.mesh'):
            names = {c[key] for c in SHAPES.values() if key in c}
            self.assertEqual(1, len(names), f'{key} moves with the shape')

    def test_no_two_stages_share_a_name(self):
        # What keeps a rebinding visible to the planner: if reconstruct and
        # refine wrote the same name, refine would clobber the artifact and the
        # 'mesh' role's move would be undetectable. Checked over the union of
        # every shape, since names no longer differ between them.
        producers = {}
        for shape in SHAPES.values():
            for key, path in shape.items():
                if key != 'texture.input':
                    producers.setdefault(path, set()).add(key)
        collisions = {p: sorted(k) for p, k in producers.items() if len(k) > 1}
        self.assertEqual({}, collisions)


if __name__ == '__main__':
    unittest.main()
