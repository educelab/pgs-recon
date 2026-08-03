"""``pgs-calibrate``: no ``choices`` list may offer a value the binary rejects.

The same move ``test_reconstruct.ACCEPTED`` makes for ``pgs-recon``, for the app
that had no equivalent. It matters here for a reason that is specific to this
parser: ``--camera-model`` and ``--resection-method`` build their ``choices``
straight off :class:`~pgs_recon.openmvg.CameraModel` and
:class:`~pgs_recon.openmvg.ResectionMethod`, so a member added to either enum for
``openMVG_main_SfM``'s benefit silently becomes a value ``pgs-calibrate`` offers
for ``openMVG_main_SfM_Localization`` -- a different binary, with its own
validation. ``CameraModel.SPHERICAL`` arrived exactly that way.

The enums' *values* are ``test_openmvg.TestEnumsMatchTheBinary``'s business. What
is asserted here is the other half: that localization accepts them, and that the
help text a user reads to choose one cannot drift from the list they are choosing
from.

Pure parser construction -- nothing here runs a binary, so the only skip guard is
for the imports ``calibrate`` needs.
"""
import importlib.util
import unittest

DEPS = ('configargparse', 'cv2', 'numpy')
MISSING = [d for d in DEPS if importlib.util.find_spec(d) is None]

#: What ``openMVG_main_SfM_Localization`` accepts for each argument we offer a
#: ``choices`` list for, transcribed from the pinned revision (``c92ed1b``).
#:
#: Values rather than names, because these two flags are integers on the command
#: line. Offering *fewer* than the binary accepts is a curation and stays allowed,
#: so this is a subset assertion.
ACCEPTED = {
    # main_SfM_Localization.cpp:135 rejects anything isValid() does not accept,
    # and Camera_Common.hpp defines that as isPinhole() || isSpherical(): the
    # pinhole range 1-5, plus CAMERA_SPHERICAL at 7. The sentinels bracketing it
    # (0, 6) are "Invalid camera type" and a dead localize stage. Spherical is
    # genuinely handled rather than merely accepted -- :377-382 builds an
    # Intrinsic_Spherical for it before resection.
    'camera_model': (1, 2, 3, 4, 5, 7),
    # solver_resection.hpp's SolverType, 0-5. Note that this one is *not*
    # range-checked: main_SfM_Localization static_casts it straight through, so
    # an out-of-range value is undefined behaviour rather than an error message,
    # which is the stronger reason to pin the list.
    'resection_method': (0, 1, 2, 3, 4, 5),
}


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestChoicesMatchThePinnedBinary(unittest.TestCase):

    @staticmethod
    def actions():
        """``dest -> action`` for every argument with a ``choices`` list."""
        from pgs_recon.apps.calibrate import build_parser
        return {a.dest: a for a in build_parser()._actions
                if a.choices is not None}

    def test_every_offered_choice_is_understood(self):
        offered = self.actions()
        for dest, accepted in ACCEPTED.items():
            with self.subTest(argument=dest):
                self.assertIn(dest, offered,
                              f'--{dest.replace("_", "-")} no longer has a '
                              f'choices list; drop it from ACCEPTED too')
                self.assertEqual(
                    set(), set(offered[dest].choices) - set(accepted),
                    f'--{dest.replace("_", "-")} offers a value '
                    f'openMVG_main_SfM_Localization does not accept')

    def test_the_default_is_itself_an_offered_choice(self):
        """An unreachable default would be a run that cannot be reproduced by
        spelling out what it did. ``--resection-method`` defaults to ``None``,
        which means 'let the binary pick' rather than a value."""
        for dest, action in self.actions().items():
            if action.default is None:
                continue
            with self.subTest(argument=dest):
                self.assertIn(action.default, action.choices)

    def test_the_help_text_enumerates_exactly_what_is_offered(self):
        """Both help strings are built by walking the same enum the choices come
        from. If one is ever hand-written, this is what notices: a user picking
        from a stale list gets an argparse error at best and the wrong camera
        model at worst."""
        for dest in ('camera_model', 'resection_method'):
            action = self.actions()[dest]
            with self.subTest(argument=dest):
                for value in action.choices:
                    self.assertIn(f'{value}=', action.help,
                                  f'--{dest.replace("_", "-")} offers {value} '
                                  f'without documenting it')


if __name__ == '__main__':
    unittest.main()
