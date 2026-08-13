"""Capture selection in the PGS Scan importer.

Only the filename half of the importer is exercised here: which images a
requested capture selects, and what an index that selects nothing reports. That
is the part with no filesystem, no exiftool and no binaries in it -- everything
after it needs a real scan directory.

Skipped unless the importer's own dependencies are installed, the same way
``test_reconstruct`` handles them; the ``test:python`` and ``test:in-image`` CI
jobs have them.
"""
import unittest
from importlib.util import find_spec
from pathlib import Path

DEPS = ('numpy', 'scipy', 'sfm_utils', 'exiftool')
MISSING = [d for d in DEPS if find_spec(d) is None]

if not MISSING:
    from pgs_recon.pgs_data import (neighbor_lookup_gridscan, parse_scan_name,
                                    select_capture)

PREFIX = 'PGS_'
EXT = 'tif'


def names(*stems):
    return [Path(f'/scan/{s}') for s in stems]


@unittest.skipIf(MISSING, f'missing dependencies: {MISSING}')
class ParseScanName(unittest.TestCase):
    def test_full_name(self):
        self.assertEqual(parse_scan_name('PGS_003_00047_02.tif', PREFIX, EXT),
                         (3, 47, 2))

    def test_capture_field_is_optional(self):
        """A name without the field is capture 0, not an unknown capture.

        Single-capture scans predate the field, and they must keep importing
        under the default.
        """
        self.assertEqual(parse_scan_name('PGS_003_00047.tif', PREFIX, EXT),
                         (3, 47, 0))

    def test_unparseable_name_has_no_capture(self):
        """An unrecognized name belongs to no capture. Reporting it as capture 0
        would make it satisfy the default import and carry an unknown position
        into the view pairs."""
        self.assertEqual(parse_scan_name('IMG_1234.tif', PREFIX, EXT),
                         (None, None, None))

    def test_empty_field_is_unknown_not_zero(self):
        """The numeric fields are ``\\d*``, so a name can match with one empty.
        That field is unknown, and must not be read as index 0."""
        self.assertEqual(parse_scan_name('PGS_003__02.tif', PREFIX, EXT),
                         (3, None, 2))
        self.assertEqual(parse_scan_name('PGS__00047_02.tif', PREFIX, EXT),
                         (None, 47, 2))

    def test_prefix_and_ext_are_literal(self):
        """Both are interpolated into a regex, so a '.' in either must not match
        any character. Asserted with a metacharacter in each, since the real
        PREFIX and EXT have none and would pass unescaped."""
        self.assertEqual(parse_scan_name('PGSx003_00047_00.tif', 'PGS.', EXT),
                         (None, None, None))
        self.assertEqual(parse_scan_name('PGS_003_00047_00.txf', PREFIX, 't.f'),
                         (None, None, None))


@unittest.skipIf(MISSING, f'missing dependencies: {MISSING}')
class SelectCapture(unittest.TestCase):
    IMAGES = names('PGS_000_00000_00.tif', 'PGS_001_00000_00.tif',
                   'PGS_003_00000_01.tif',
                   'PGS_000_00001_00.tif', 'PGS_003_00001_01.tif')

    def test_default_capture_selects_the_first(self):
        selected, found = select_capture(self.IMAGES, PREFIX, EXT, 0)
        self.assertEqual([p.name for p, _, _ in selected],
                         ['PGS_000_00000_00.tif', 'PGS_001_00000_00.tif',
                          'PGS_000_00001_00.tif'])
        self.assertEqual(found, {0, 1})

    def test_selecting_a_later_capture(self):
        """Capture 1 here is one camera at every position -- the shape a
        'Center+IR940' capture actually has."""
        selected, _ = select_capture(self.IMAGES, PREFIX, EXT, 1)
        self.assertEqual([(p.name, cam, pos) for p, cam, pos in selected],
                         [('PGS_003_00000_01.tif', 3, 0),
                          ('PGS_003_00001_01.tif', 3, 1)])

    def test_indices_come_back_parsed(self):
        """The caller must not have to re-parse: selection is the only place the
        name is read."""
        selected, _ = select_capture(names('PGS_004_00123_02.tif'),
                                     PREFIX, EXT, 2)
        self.assertEqual(selected[0][1:], (4, 123))

    def test_empty_selection_reports_what_is_present(self):
        """The captures actually on disk are what a mistyped index is reported
        against, so the error can name them."""
        selected, found = select_capture(self.IMAGES, PREFIX, EXT, 7)
        self.assertEqual(selected, [])
        self.assertEqual(found, {0, 1})

    def test_capture_zero_over_a_legacy_scan(self):
        legacy = names('PGS_000_00000.tif', 'PGS_001_00000.tif')
        selected, found = select_capture(legacy, PREFIX, EXT, 0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(found, {0})

    def test_later_capture_over_a_legacy_scan_selects_nothing(self):
        legacy = names('PGS_000_00000.tif', 'PGS_001_00000.tif')
        selected, _ = select_capture(legacy, PREFIX, EXT, 1)
        self.assertEqual(selected, [])

    def test_unrecognized_names_are_skipped(self):
        """The glob is '{prefix}*.{ext}', so a derived or stray file can match it
        without matching the convention. It belongs to no capture, so it neither
        satisfies the default import nor appears in what an empty selection is
        reported against."""
        mixed = names('PGS_000_00000_00.tif', 'PGS_000_00000_02_rgb.tif')
        with self.assertLogs('pgs_recon.pgs_data', 'WARNING') as logged:
            selected, found = select_capture(mixed, PREFIX, EXT, 0)
        self.assertEqual([p.name for p, _, _ in selected],
                         ['PGS_000_00000_00.tif'])
        self.assertEqual(found, {0})
        self.assertIn('PGS_000_00000_02_rgb.tif', logged.output[0])

    def test_only_unrecognized_names_select_nothing(self):
        stray = names('PGS_000_00000_02_rgb.tif')
        with self.assertLogs('pgs_recon.pgs_data', 'WARNING'):
            selected, found = select_capture(stray, PREFIX, EXT, 0)
        self.assertEqual(selected, [])
        self.assertEqual(found, set())


@unittest.skipIf(MISSING, f'missing dependencies: {MISSING}')
class NeighborLookup(unittest.TestCase):
    """A 3x3 single-layer grid, positions 0-8 in row-major order."""
    SCAN = {'path': 'ROW_RESET', 'dims': [2, 2, 0], 'stepsize': [1, 1, 1],
            'capture_positions': [[0, 0, 0]] * 9}

    def setUp(self):
        self.lookup = neighbor_lookup_gridscan(self.SCAN)

    def test_neighbors_of_an_interior_position(self):
        self.assertEqual(sorted(self.lookup(4, 1).tolist()), list(range(9)))

    def test_neighbors_are_clipped_at_the_edge(self):
        self.assertEqual(sorted(self.lookup(0, 1).tolist()), [0, 1, 3, 4])

    def test_position_past_the_grid_has_no_neighbors(self):
        """The position index comes from a filename, which can name a position
        the grid does not have."""
        self.assertEqual(self.lookup(99, 1).tolist(), [])

    def test_unknown_position_has_no_neighbors(self):
        """``parse_scan_name`` returns None for a field the name omits, and that
        reaches the lookup as a position key."""
        self.assertEqual(self.lookup(None, 1).tolist(), [])


if __name__ == '__main__':
    unittest.main()
