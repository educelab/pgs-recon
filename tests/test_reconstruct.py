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

from pgs_recon import layout

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
        manifest = layout.manifest(self.out)
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


#: What each binary actually parses for the arguments we offer a ``choices`` list
#: for, transcribed from the pinned revisions -- OpenMVG ``c92ed1b``, and
#: ``dependencies/utilities/src/global_scaler.cpp`` for the last one. The same
#: move ``test_openmvg.SURFACES`` makes for flags, for values: a name we offer
#: that the binary does not know is a stage that dies partway into a run, and no
#: signature or ``choices`` list can catch it alone.
ACCEPTED = {
    # main_ComputeFeatures.cpp:192-210, stringToEnum:45-58
    'describer_method': ('SIFT', 'SIFT_ANATOMY', 'AKAZE_FLOAT', 'AKAZE_MLDB'),
    'describer_preset': ('NORMAL', 'HIGH', 'ULTRA'),
    # main_ComputeMatches.cpp:232-291
    'matching_method': ('AUTO', 'BRUTEFORCEL2', 'BRUTEFORCEHAMMING', 'HNSWL2',
                        'HNSWL1', 'HNSWHAMMING', 'CASCADEHASHINGL2',
                        'FASTCASCADEHASHINGL2'),
    # main_GeometricFilter.cpp:165-183
    'matching_geometric_model': ('f', 'e', 'h', 'a', 'u', 'o'),
    # main_SfM.cpp:66-72 -- plus 'direct', which is ours: it routes to
    # ComputeStructureFromKnownPoses rather than naming an -s engine.
    'mvg_recon_method': ('incremental', 'incrementalv2', 'global', 'stellar',
                         'direct'),
    # main_SfM.cpp:86-92
    'mvg_initializer': ('EXISTING_POSE', 'MAX_PAIR', 'AUTO_PAIR', 'STELLAR'),
    # global_scaler.cpp:129-143
    'autoscale_method': ('markers', 'sample-square'),
}


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestChoicesMatchThePinnedBinaries(unittest.TestCase):
    """No ``choices`` list may offer a value the binary rejects.

    ``--matching-method ANNL2`` was offered for as long as this parser has
    existed and stopped parsing when upstream replaced ANN with HNSW: choosing it
    got you an ``Invalid Nearest Neighbor method`` and a dead matches stage, after
    features had already run. Offering *fewer* values than the binary accepts is a
    curation and stays allowed -- ``describer_method`` omits ``SIFT_ANATOMY`` --
    so this is a subset assertion, not an equality one.
    """

    @staticmethod
    def choices():
        from pgs_recon.apps.reconstruct import build_parser
        return {a.dest: a.choices for a in build_parser()._actions
                if a.choices is not None}

    def test_every_offered_choice_is_understood(self):
        offered = self.choices()
        for dest, accepted in ACCEPTED.items():
            with self.subTest(argument=dest):
                self.assertIn(dest, offered,
                              f'--{dest.replace("_", "-")} no longer has a '
                              f'choices list; drop it from ACCEPTED too')
                self.assertEqual(
                    set(), set(offered[dest]) - set(accepted),
                    f'--{dest.replace("_", "-")} offers a value the pinned '
                    f'binary does not parse')

    def test_the_hnsw_matchers_are_reachable(self):
        """The other half of the ANNL2 fix: the matchers that replaced it were
        never added, so the only approximate matcher at this pin was
        unreachable."""
        offered = set(self.choices()['matching_method'])
        self.assertLessEqual({'HNSWL2', 'HNSWL1', 'HNSWHAMMING'}, offered)
        self.assertNotIn('ANNL2', offered)


if __name__ == '__main__':
    unittest.main()
