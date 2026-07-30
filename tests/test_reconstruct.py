"""``pgs-recon`` behaviour that needs the real parser.

Unlike ``test_stages``, this imports ``pgs_recon.apps.reconstruct``, which pulls
in ``configargparse``, ``sfm_utils`` and ``exiftool``. It still runs no binary:
every case here stops at ``--dry-run``. The whole module skips when those
packages are absent, so ``python3 -m unittest discover -s tests`` stays useful in
a bare Python.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DEPS = ('configargparse', 'sfm_utils', 'exiftool')
MISSING = [d for d in DEPS if importlib.util.find_spec(d) is None]


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestDryRun(unittest.TestCase):
    """--dry-run must resolve everything and persist nothing (finding #3).

    Anything written here is inherited: a dry run with ``--no-mvs-refine`` used
    to record ``mvs_refine: False`` in ``effective_args``, so the next bare
    ``pgs-recon -o out`` silently dropped refine from the pipeline.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'recon'
        self.images = Path(self.tmp.name) / 'images'
        self.images.mkdir()
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def dry_run(*argv):
        from pgs_recon.apps import reconstruct
        argv = ['pgs-recon', '--log-level', 'ERROR', *argv]
        with mock.patch.object(sys, 'argv', argv):
            reconstruct._main()

    def test_writes_nothing_into_a_fresh_directory(self):
        self.dry_run('-i', str(self.images), '-o', str(self.out),
                     '--name', 'obj', '--dry-run', '--no-mvs-refine')
        self.assertFalse(self.out.exists(),
                         'a dry run created the output directory')

    def test_leaves_an_existing_manifest_untouched(self):
        self.out.mkdir()
        manifest = self.out / 'metadata.json'
        manifest.write_text(json.dumps({
            'effective_args': {'input': str(self.images), 'name': 'obj',
                               'mvs_refine': True, 'mvs_densify': False},
            'stages': {},
            'runs': [],
        }, indent=4))
        before = manifest.read_bytes()

        self.dry_run('-o', str(self.out), '--dry-run', '--no-mvs-refine',
                     '--rerun')

        self.assertEqual(before, manifest.read_bytes(),
                         'a dry run rewrote the manifest')
        after = json.loads(manifest.read_text())
        self.assertTrue(after['effective_args']['mvs_refine'],
                        '--dry-run persisted --no-mvs-refine')
        self.assertEqual([], after['runs'])
        self.assertNotIn('shape', after)
        self.assertEqual([], list(self.out.glob('*_recon_config.txt')))
        self.assertFalse((self.out / 'mvg').exists())
        self.assertFalse((self.out / 'mvs').exists())


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestArgMap(unittest.TestCase):
    def test_every_parser_argument_is_classified(self):
        # The same assertion main() makes at startup: a flag missing from
        # STAGE_ARGS is silently un-overridable.
        from pgs_recon.apps.reconstruct import build_parser
        from pgs_recon.stages import validate_arg_map
        validate_arg_map(build_parser())


if __name__ == '__main__':
    unittest.main()
