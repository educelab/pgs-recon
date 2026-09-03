"""``--filter-cc-area`` on the two mesh-filtering apps.

``pgs-remove-ground-plane`` and ``pgs-filter-small-components`` both take it,
mutually exclusively with ``--filter-cc``, because the acquisition pipeline
sends it on whichever of the two its ground-removal branch selects. The
interaction worth pinning is that ``--filter-cc``'s ``largest`` default must
not also apply when an area is given -- argparse still fills that default in.

The mesh work itself is ``utils.geometry``'s (see ``test_geometry``), so these
drive ``main()`` with the geometry module and the OBJ reader stubbed out.

Skips when the apps' import chain is incomplete.
"""
import contextlib
import importlib
import importlib.util
import io
import sys
import unittest
from unittest import mock

MISSING = [d for d in ('numpy', 'scipy', 'cv2', 'vtkmodules')
           if importlib.util.find_spec(d) is None]

APPS = ('pgs_recon.apps.filter_small_components',
        'pgs_recon.apps.remove_ground_plane')


def drive(name, argv):
    """Run an app's ``main()`` over stubs; return the geometry calls it made"""
    module = importlib.import_module(name)
    geom = mock.MagicMock()
    # remove_ground_plane prints the fit before it filters
    geom.segment_ground_surface.return_value = (mock.MagicMock(warp=0., rms=0.),
                                                [0])
    args = ['app', '-i', 'in.obj', '-o', 'out.obj'] + argv
    with mock.patch.object(module, 'geom', geom), \
            mock.patch.object(module, 'wobj'), \
            mock.patch.object(sys, 'argv', args), \
            contextlib.redirect_stdout(io.StringIO()):
        module.main()
    return geom


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestFilterCCArea(unittest.TestCase):

    def test_both_apps_filter_by_area(self):
        for name in APPS:
            with self.subTest(name):
                geom = drive(name, ['--filter-cc-area', '0.5'])
                geom.remove_connected_components_by_area.assert_called_once()
                self.assertEqual(
                    geom.remove_connected_components_by_area.call_args.kwargs,
                    {'min_area': 0.5})

    def test_an_area_replaces_the_largest_default(self):
        """The one thing the pipeline cannot defend against from outside."""
        for name in APPS:
            with self.subTest(name):
                geom = drive(name, ['--filter-cc-area', '0.5'])
                geom.keep_largest_connected_component.assert_not_called()
                geom.remove_connected_components_by_size.assert_not_called()

    def test_zero_is_the_no_filtering_spelling(self):
        for name in APPS:
            with self.subTest(name):
                geom = drive(name, ['--filter-cc-area', '0'])
                self.assertEqual(
                    geom.remove_connected_components_by_area.call_args.kwargs,
                    {'min_area': 0.})
                geom.keep_largest_connected_component.assert_not_called()

    def test_the_two_filters_are_mutually_exclusive(self):
        for name in APPS:
            with self.subTest(name):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as e:
                        drive(name, ['--filter-cc', 'largest',
                                     '--filter-cc-area', '0.5'])
                self.assertEqual(e.exception.code, 2)

    def test_rejects_an_area_that_is_not_one(self):
        """``nan`` compares False against every component: it would save an
        empty mesh, and a negative would silently mean 'keep everything'."""
        for name in APPS:
            for arg in ('nope', '-1', 'nan', 'inf'):
                with self.subTest(name, arg=arg):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as e:
                            drive(name, ['--filter-cc-area', arg])
                    self.assertEqual(e.exception.code, 2)


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestFilterCCUnchanged(unittest.TestCase):
    """Callers that say nothing, or say ``--filter-cc``, are unaffected."""

    def test_the_default_is_still_largest(self):
        for name in APPS:
            with self.subTest(name):
                geom = drive(name, [])
                geom.keep_largest_connected_component.assert_called_once_with(
                    mock.ANY, filter_vertices=True)
                geom.remove_connected_components_by_area.assert_not_called()

    def test_a_face_count_still_filters_by_size(self):
        for name in APPS:
            with self.subTest(name):
                geom = drive(name, ['--filter-cc', '100'])
                self.assertEqual(
                    geom.remove_connected_components_by_size.call_args.kwargs,
                    {'num_faces': 100})
                geom.remove_connected_components_by_area.assert_not_called()


if __name__ == '__main__':
    unittest.main()
