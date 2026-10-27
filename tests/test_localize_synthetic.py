"""``pgs-localize`` and ``pgs-localize-joint`` against a scene with a known answer.

The only tests here that run the localization binaries end to end. The scene is
a textured plane (:mod:`synthetic_scene`) viewed from known poses, so what is
checked is the *answer* -- the recovered pose against the truth -- and not
merely that the tools exit 0.

What each class guards:

- :class:`TestPgsLocalize` -- render-and-match recovers a pose it was given a
  wrong prior for, and a fixed ``--seed`` makes a run byte-for-byte
  repeatable. The second is what makes an A/B against a refactor meaningful
  (it is how ``localize_core``'s extraction was shown to change nothing).
- :class:`TestLocalizeJoint` -- ADR 0013's per-view path: the overhead position
  map agrees with the plane, and a hand-held view is localized with **no
  prior pose** at all, through the overhead image.

Binaries are found the way the pipeline finds them (``toolchain``), so in the
published image this tests the installed tools. ``pgs-localize-joint`` is only
built with ``PGS_BUILD_VIEWSET=ON``; its tests skip without it.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np

    import synthetic_scene as scene
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = np = scene = None

from pgs_recon import toolchain
from pgs_recon.toolchain import ToolNotFound


def binary(name: str) -> Path:
    try:
        return toolchain.resolve_exe(name)
    except ToolNotFound as e:
        raise unittest.SkipTest(str(e))


def run(argv) -> subprocess.CompletedProcess:
    result = subprocess.run([str(a) for a in argv], capture_output=True,
                            text=True)
    if result.returncode != 0:
        raise AssertionError(f'{Path(argv[0]).name} exited '
                             f'{result.returncode}:\n{result.stdout[-3000:]}'
                             f'\n{result.stderr[-3000:]}')
    return result


def requires_scene(cls):
    if scene is None:
        return unittest.skip('needs numpy and cv2')(cls)
    return cls


@requires_scene
class TestPgsLocalize(unittest.TestCase):
    """Render-and-match from a deliberately wrong prior."""

    @classmethod
    def setUpClass(cls):
        cls.exe = binary('pgs-localize')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        texture = scene.make_texture()
        cls.mesh = scene.write_plane(cls.dir, texture)
        cls.truth = scene.View(
            R=scene.rot_x(6.0) @ scene.rot_z(12.0) @ scene.DOWN,
            C=np.array([3.0, -2.0, 150.0]), f=2000.0, cx=507.3, cy=371.9,
            width=1000, height=750)
        cv2.imwrite(str(cls.dir / 'query.png'), scene.render(cls.truth, texture))
        (cls.dir / 'prior.cam').write_text(
            scene.camera_file(scene.nudged(cls.truth)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def localize(self, tag: str) -> Path:
        out = self.dir / tag
        out.mkdir()
        run([self.exe, '--image', self.dir / 'query.png', '--mesh', self.mesh,
             '--camera', self.dir / 'prior.cam', '--seed', 42,
             '--threads', 2,
             '--output-calibration', out / 'cal.json',
             '--output-camera', out / 'cam.txt',
             '--report', out / 'report.json'])
        return out

    def test_recovers_the_true_pose(self):
        report = json.loads((self.localize('pose') / 'report.json').read_text())
        centre = np.array(report['pose']['center_cm'])
        self.assertLess(np.linalg.norm(centre - self.truth.C), 0.1,
                        f'solved {centre}, truth {self.truth.C}')
        self.assertLess(report['heldout']['rms_px'], 1.0)
        self.assertEqual(report['gates']['worst'], 'pass')

    def test_a_seeded_run_is_repeatable_byte_for_byte(self):
        a, b = self.localize('a'), self.localize('b')
        for name in ('cal.json', 'cam.txt', 'report.json'):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(),
                             f'{name} differs between two seeded runs')


@requires_scene
class TestLocalizeJoint(unittest.TestCase):
    """ADR 0013's per-view path, overhead prior and all."""

    @classmethod
    def setUpClass(cls):
        cls.joint = binary('pgs-localize-joint')
        cls.localize = binary('pgs-localize')
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.dir = Path(cls.tmp.name)
        texture = scene.make_texture()
        cls.mesh = scene.write_plane(d, texture)

        # The overhead: the whole plate, straight down -- localized with
        # pgs-localize, as a real one is (ADR 0011).
        cls.overhead = scene.View(R=scene.DOWN, C=np.array([0.0, 0.0, 150.0]),
                                  f=1700.0, cx=600.0, cy=600.0, width=1200,
                                  height=1200)
        cv2.imwrite(str(d / 'overhead.png'), scene.render(cls.overhead, texture,
                                                          seed=3))
        (d / 'overhead_prior.cam').write_text(
            scene.camera_file(scene.nudged(cls.overhead)))
        run([cls.localize, '--image', d / 'overhead.png', '--mesh', cls.mesh,
             '--camera', d / 'overhead_prior.cam', '--seed', 1, '--threads', 2,
             '--output-calibration', d / 'overhead_calibration.json'])

        # The hand-held view: closer, tilted, rotated, a patch of the plate.
        cls.handheld = scene.View(
            R=scene.rot_x(8.0) @ scene.rot_z(-25.0) @ scene.DOWN,
            C=np.array([8.0, 5.0, 60.0]), f=1600.0, cx=403.0, cy=297.0,
            width=800, height=600)
        cv2.imwrite(str(d / 'handheld.png'), scene.render(cls.handheld, texture,
                                                          gamma=1.2, seed=5))
        # K only: the view gets no prior pose from anywhere but the overhead.
        (d / 'handheld.cam').write_text(
            scene.camera_file(cls.handheld, with_pose=False))

        run([cls.joint, 'position-map',
             '--calibration', d / 'overhead_calibration.json',
             '--mesh', cls.mesh, '--output', d / 'position.npy',
             '--threads', 2])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_position_map_is_the_plane(self):
        pos = np.load(self.dir / 'position.npy')
        self.assertEqual(pos.shape, (1200, 1200, 3))
        self.assertEqual(pos.dtype, np.float32)
        hit = np.isfinite(pos[..., 0])
        self.assertGreater(hit.mean(), 0.8)
        self.assertLess(np.abs(pos[hit][:, 2]).max(), 1e-3, 'off the plane')
        # Against the true overhead: the solved one is within a fraction of a
        # unit of it, so pixel for pixel the two maps agree to about that.
        truth = scene.plane_points(self.overhead)
        both = hit & np.isfinite(truth[..., 0])
        err = np.linalg.norm(pos[both][:, :2] - truth[both][:, :2], axis=1)
        self.assertLess(np.median(err), 0.1)

    def test_view_localizes_with_no_prior_pose(self):
        d = self.dir / 'view'
        d.mkdir()
        run([self.joint, 'view',
             '--image', self.dir / 'handheld.png', '--mesh', self.mesh,
             '--overhead-image', self.dir / 'overhead.png',
             '--overhead-calibration', self.dir / 'overhead_calibration.json',
             '--overhead-position-map', self.dir / 'position.npy',
             '--camera', self.dir / 'handheld.cam',
             '--output-calibration', d / 'cal.json',
             '--output-correspondences', d / 'corr.npy',
             '--output-camera', d / 'cam.txt',
             '--qa-render', d / 'qa.png',
             '--report', d / 'report.json', '--seed', 3, '--threads', 2])
        report = json.loads((d / 'report.json').read_text())
        self.assertEqual(report['status'], 'done')

        centre = np.array(report['pose']['center'])
        self.assertLess(np.linalg.norm(centre - self.handheld.C), 0.2,
                        f'solved {centre}, truth {self.handheld.C}')
        self.assertTrue(report['heldout']['ok'])
        self.assertLess(report['heldout']['rms_px'], 1.5)

        # The correspondences: the inliers, in the full-resolution query's
        # pixels, lifted onto the plane -- so they reproject through the TRUE
        # camera to where they were found.
        corr = np.load(d / 'corr.npy')
        self.assertEqual(corr.shape, (report['solve']['n_inliers'], 5))
        X = corr[:, 2:]
        self.assertLess(np.abs(X[:, 2]).max(), 1e-3)
        cam = (X - self.handheld.C) @ self.handheld.R.T
        uv = self.handheld.f * cam[:, :2] / cam[:, 2:] + [self.handheld.cx,
                                                          self.handheld.cy]
        self.assertLess(np.median(np.linalg.norm(uv - corr[:, :2], axis=1)),
                        1.0)

        for name in ('cal.json', 'cam.txt', 'qa.png', 'qa_diff.png'):
            self.assertTrue((d / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
