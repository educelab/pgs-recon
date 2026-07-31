"""``pgs_recon.utils.recon_dir``: locating a run's artifacts from its manifest.

Both ``pgs-retexture`` (which needs the mesh and the frame it lives in) and
``pgs-calibrate`` (which needs neither the mesh nor, given ``--sfm-data``, the
recorded SfM) resolve against the same ``metadata.json``. Each artifact has its
own resolver returning a `Resolved`, so the contract worth pinning is: a resolver
either hands back a path that really exists on disk, or a reason explaining why
it could not -- never a path that cannot be opened, and never silence.

Pure stdlib: this module reads JSON off a temp dir and runs no binary, so unlike
``test_reconstruct`` it needs no skip guard.
"""
import json
import tempfile
import unittest
from pathlib import Path

from pgs_recon.utils.recon_dir import (
    Resolved,
    load_manifest,
    resolve_solved_sfm,
    resolve_textured_mesh,
)

# A completed run: convert records the SfM it consumed, texture the mesh it made.
STAGES_FULL = {'stages': {
    'convert': {'status': 'complete',
                'inputs': {'sfm': 'mvg/recon_dir/sfm_data.bin'}},
    'texture': {'status': 'complete', 'outputs': {'mesh': 'mvs/obj.obj'}}}}
# An SfM-only run (--no-mvs / --to colorize): no convert, so no MVS-frame SfM.
STAGES_SFM_ONLY = {'stages': {'sfm': {'status': 'complete'}}}
STAGES_CONVERT_FAILED = {'stages': {'convert': {'status': 'failed'}}}
STAGES_TEXTURE_FAILED = {'stages': {
    'convert': {'status': 'complete',
                'inputs': {'sfm': 'mvg/recon_dir/sfm_data.bin'}},
    'texture': {'status': 'failed'}}}
# Pre-stage-record manifests: the SfM is recovered from the command log.
LEGACY_FULL = {'commands': {'1': 'openMVG2openMVS -i /old/sfm_data.bin -o s.mvs'},
               'parsed': {'name': 'obj', 'file_type': 'obj'}}
LEGACY_SFM_ONLY = {'commands': {'1': 'openMVG_main_ComputeFeatures -i x'},
                   'parsed': {'name': 'obj', 'file_type': 'obj'}}
LEGACY_NO_NAME = {'commands': {'1': 'openMVG2openMVS -i /old/sfm_data.bin -o s'},
                  'parsed': {}}


class ReconDirCase(unittest.TestCase):
    """Lays out a recon dir: a manifest plus whichever artifacts exist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recon = Path(self.tmp.name) / 'recon'
        self.recon.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def write(self, meta, sfm=True, mesh=True):
        (self.recon / 'metadata.json').write_text(json.dumps(meta))
        if sfm:
            (self.recon / 'mvg' / 'recon_dir').mkdir(parents=True, exist_ok=True)
            (self.recon / 'mvg' / 'recon_dir' / 'sfm_data.bin').write_text('{}')
        if mesh:
            (self.recon / 'mvs').mkdir(parents=True, exist_ok=True)
            (self.recon / 'mvs' / 'obj.obj').write_text('o')
        return self.recon


class TestLoadManifest(ReconDirCase):

    def test_missing_manifest_exits(self):
        """No manifest means recon_dir is not a pgs-recon output at all."""
        with self.assertRaises(SystemExit):
            load_manifest(self.recon)

    def test_returns_path_and_parsed_manifest(self):
        recon = self.write(STAGES_FULL)
        meta_path, meta = load_manifest(recon)
        self.assertEqual(meta_path, recon / 'metadata.json')
        self.assertIn('convert', meta['stages'])


class TestResolved(unittest.TestCase):

    def test_require_returns_the_path(self):
        p = Path('/tmp/x.bin')
        self.assertEqual(Resolved(p, None).require(), p)

    def test_require_exits_with_the_reason(self):
        with self.assertRaises(SystemExit) as ctx:
            Resolved(None, 'because reasons').require()
        self.assertEqual(str(ctx.exception), 'because reasons')


class TestResolveSolvedSfm(ReconDirCase):

    def test_resolves_convert_input(self):
        """The frame is convert's *input*, not any stage's output."""
        for label, meta in (('stage records', STAGES_FULL),
                            ('legacy log', LEGACY_FULL)):
            with self.subTest(label):
                got = resolve_solved_sfm(self.write(meta))
                self.assertIsNone(got.reason)
                self.assertEqual(got.path.name, 'sfm_data.bin')
                # Rebuilt under recon_dir, never trusted as recorded: the legacy
                # log says /old/sfm_data.bin.
                self.assertTrue(got.path.is_relative_to(self.recon))

    def test_unresolvable_reports_reason(self):
        for label, meta in (('convert never ran', STAGES_SFM_ONLY),
                            ('convert failed', STAGES_CONVERT_FAILED),
                            ('legacy, no convert', LEGACY_SFM_ONLY)):
            with self.subTest(label):
                got = resolve_solved_sfm(self.write(meta, sfm=False, mesh=False))
                self.assertIsNone(got.path)
                self.assertTrue(got.reason)

    def test_never_run_and_crashed_are_distinguished(self):
        """The recorded status is what separates 'never ran' from 'crashed'."""
        never = resolve_solved_sfm(self.write(STAGES_SFM_ONLY, sfm=False))
        self.assertIn('no convert stage', never.reason)
        crashed = resolve_solved_sfm(
            self.write(STAGES_CONVERT_FAILED, sfm=False))
        self.assertIn("'failed'", crashed.reason)

    def test_recorded_but_absent_is_not_returned(self):
        got = resolve_solved_sfm(self.write(STAGES_FULL, sfm=False))
        self.assertIsNone(got.path)
        self.assertIn('missing from disk', got.reason)


class TestResolveTexturedMesh(ReconDirCase):

    def test_resolves_texture_output(self):
        for label, meta in (('stage records', STAGES_FULL),
                            ('legacy log', LEGACY_FULL)):
            with self.subTest(label):
                got = resolve_textured_mesh(self.write(meta))
                self.assertIsNone(got.reason)
                self.assertEqual(got.path.name, 'obj.obj')

    def test_unresolvable_reports_reason(self):
        for label, meta, sfm in (('texture never ran', STAGES_SFM_ONLY, False),
                                 ('texture failed', STAGES_TEXTURE_FAILED, True),
                                 ('legacy, no name', LEGACY_NO_NAME, True)):
            with self.subTest(label):
                got = resolve_textured_mesh(self.write(meta, sfm=sfm,
                                                       mesh=False))
                self.assertIsNone(got.path)
                self.assertTrue(got.reason)

    def test_recorded_but_absent_is_not_returned(self):
        got = resolve_textured_mesh(self.write(STAGES_FULL, mesh=False))
        self.assertIsNone(got.path)
        self.assertIn('missing from disk', got.reason)

    def test_mesh_resolves_independently_of_the_sfm(self):
        """pgs-calibrate never asks for the mesh; a missing one must not leak
        into the SfM's resolution (and vice versa)."""
        recon = self.write(STAGES_FULL, mesh=False)
        self.assertIsNotNone(resolve_solved_sfm(recon).path)
        self.assertIsNone(resolve_textured_mesh(recon).path)


class TestInvariant(ReconDirCase):
    """Over every manifest shape and artifact combination: a returned path is
    always openable, and a None path always comes with an explanation."""

    def test_path_xor_reason(self):
        metas = (STAGES_FULL, STAGES_SFM_ONLY, STAGES_CONVERT_FAILED,
                 STAGES_TEXTURE_FAILED, LEGACY_FULL, LEGACY_SFM_ONLY,
                 LEGACY_NO_NAME)
        for i, meta in enumerate(metas):
            for sfm_on in (True, False):
                for mesh_on in (True, False):
                    with self.subTest(meta=i, sfm=sfm_on, mesh=mesh_on):
                        self.check(meta, sfm_on, mesh_on)

    def check(self, meta, sfm_on, mesh_on):
        with tempfile.TemporaryDirectory() as root:
            self.recon = Path(root) / 'recon'
            self.recon.mkdir()
            recon = self.write(meta, sfm=sfm_on, mesh=mesh_on)
            for resolver in (resolve_solved_sfm, resolve_textured_mesh):
                got = resolver(recon)
                if got.path is None:
                    self.assertTrue(got.reason,
                                    f'{resolver.__name__}: None without a reason')
                else:
                    self.assertIsNone(got.reason,
                                      f'{resolver.__name__}: path and reason')
                    self.assertTrue(got.path.is_file(),
                                    f'{resolver.__name__}: unopenable path')


if __name__ == '__main__':
    unittest.main()
