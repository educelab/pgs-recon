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
- :class:`TestAdjust` -- the joint adjustment recovers the true K and lens
  distortion from views localized with a wrong one, says the focal is
  observable when the views' tilts make it so, uses tie points from a scene in
  a foreign frame, and leaves a fixed K alone.
- :class:`TestAdjustUntilted` -- the control: views with no tilt do not
  determine the focal, and the adjustment must say so and emit nothing it
  could not measure. Without the check this guards, the solve walks every
  camera hundreds of units away while its residuals *improve*.
- :class:`TestPipeline` -- ``pgs-localize-set`` end to end on that plate: every
  phase, from the position map through the group SfM and the joint adjustment
  to one textured mesh per view, and a second run that redoes nothing.
- :class:`TestSceneConventions` -- the scene's own renderer and projection
  agree, so a failure above is the tool's and not the fixture's.

Binaries are found the way the pipeline finds them (``toolchain``), so in the
published image this tests the installed tools. ``pgs-localize-joint`` is only
built with ``PGS_BUILD_VIEWSET=ON``; its tests skip without it.
"""
import json
import shutil
import subprocess
import sys
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


def make_overhead(d: Path, mesh: Path, texture, localize: Path,
                  joint: Path) -> 'scene.View':
    """The overhead: the whole plate, straight down, localized with
    pgs-localize as a real one is (ADR 0011), and its position map."""
    overhead = scene.View(R=scene.DOWN, C=np.array([0.0, 0.0, 150.0]),
                          f=1700.0, cx=600.0, cy=600.0, width=1200,
                          height=1200)
    cv2.imwrite(str(d / 'overhead.png'), scene.render(overhead, texture,
                                                      seed=3))
    (d / 'overhead_prior.cam').write_text(
        scene.camera_file(scene.nudged(overhead)))
    run([localize, '--image', d / 'overhead.png', '--mesh', mesh,
         '--camera', d / 'overhead_prior.cam', '--seed', 1, '--threads', 2,
         '--output-calibration', d / 'overhead_calibration.json'])
    run([joint, 'position-map',
         '--calibration', d / 'overhead_calibration.json',
         '--mesh', mesh, '--output', d / 'position.npy', '--threads', 2])
    return overhead


def localize_view(joint: Path, d: Path, mesh: Path, image: Path,
                  camera: Path, out: Path, seed: int = 3) -> Path:
    """``pgs-localize-joint view`` with the fixture's overhead."""
    out.mkdir(parents=True, exist_ok=True)
    run([joint, 'view', '--image', image, '--mesh', mesh,
         '--overhead-image', d / 'overhead.png',
         '--overhead-calibration', d / 'overhead_calibration.json',
         '--overhead-position-map', d / 'position.npy',
         '--camera', camera,
         '--output-calibration', out / 'cal.json',
         '--output-correspondences', out / 'corr.npy',
         '--output-camera', out / 'cam.txt', '--qa-render', out / 'qa.png',
         '--report', out / 'report.json', '--seed', seed, '--threads', 2])
    return out


def calibration_center(path: Path) -> 'np.ndarray':
    data = json.loads(Path(path).read_text())
    return np.array(data['extrinsics'][0]['value']['center'])


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
        cls.overhead = make_overhead(d, cls.mesh, texture, cls.localize,
                                     cls.joint)

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
        d = localize_view(self.joint, self.dir, self.mesh,
                          self.dir / 'handheld.png', self.dir / 'handheld.cam',
                          self.dir / 'view')
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



class _AdjustFixture(unittest.TestCase):
    """Four hand-held views of one distorted camera, localized with a wrong K.

    Subclasses choose the views' rotations (:attr:`POSES`). No tests of its
    own, so unittest never sets it up for itself.
    """

    #: The camera the views were taken with.
    TRUE_K = dict(f=1600.0, cx=403.0, cy=297.0, k=(-0.05, 0.0, 0.0))
    #: The K they are localized with: focal 5% long, principal point off,
    #: distortion unknown -- what an EXIF focal and an image centre give.
    START = dict(f=1680.0, cx=411.0, cy=290.0, k=(0.0, 0.0, 0.0))

    @classmethod
    def setUpClass(cls):
        cls.joint = binary('pgs-localize-joint')
        cls.localize = binary('pgs-localize')
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.dir = Path(cls.tmp.name)
        texture = scene.make_texture()
        cls.mesh = scene.write_plane(d, texture)
        make_overhead(d, cls.mesh, texture, cls.localize, cls.joint)

        poses = cls.POSES()
        k = cls.TRUE_K
        cls.views = [scene.View(R=R @ scene.DOWN, C=np.array(C), f=k['f'],
                                cx=k['cx'], cy=k['cy'], width=800, height=600,
                                k=k['k']) for R, C in poses]
        s = cls.START
        start = cls.views[0]._replace(f=s['f'], cx=s['cx'], cy=s['cy'],
                                      k=s['k'])
        cls.start_cam = d / 'start.cam'
        cls.start_cam.write_text(scene.camera_file(start, with_pose=False))

        cls.names = [f'handheld_{i}.png' for i in range(len(cls.views))]
        cls.localized = []
        for i, (view, name) in enumerate(zip(cls.views, cls.names)):
            cv2.imwrite(str(d / name), scene.render(view, texture, gamma=1.2,
                                                    seed=20 + i))
            cls.localized.append(localize_view(
                cls.joint, d, cls.mesh, d / name, cls.start_cam,
                d / f'view_{i}', seed=i + 1))

        # A group solve's tracks: points on the plane where the views overlap.
        g = np.linspace(-5.0, 25.0, 31)
        pts = np.array([(x, y, 0.0) for x in g for y in g])
        (d / 'tie_scene.json').write_text(json.dumps(
            scene.tie_scene(cls.views, cls.names, pts)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def adjust(self, tag: str, *extra) -> dict:
        out = self.dir / tag
        out.mkdir()
        argv = [self.joint, 'adjust']
        for i, v in enumerate(self.localized):
            argv += ['--calibration', v / 'cal.json',
                     '--correspondences', v / 'corr.npy',
                     '--output-calibration', out / f'cal_{i}.json']
        argv += ['--output-scene', out / 'scene.json',
                 '--output-camera', out / 'camera.txt',
                 '--report', out / 'report.json', '--threads', 2, *extra]
        run(argv)
        self.out = out
        return json.loads((out / 'report.json').read_text())



@requires_scene
class TestAdjust(_AdjustFixture):
    """Views tilted about different axes: what makes the focal observable on a
    plane (Zhang's calibration, ADR 0013 Decision 4)."""

    @staticmethod
    def POSES():
        return [
            (scene.rot_x(10) @ scene.rot_z(-25), (8.0, 5.0, 60.0)),
            (scene.rot_y(-12) @ scene.rot_z(15), (14.0, 1.0, 62.0)),
            (scene.rot_x(-14) @ scene.rot_y(6) @ scene.rot_z(40),
             (9.0, 10.0, 58.0)),
            (scene.rot_y(15), (3.0, 3.0, 61.0)),
        ]

    def assert_recovers_k(self, report):
        k = report['intrinsic']['adjusted']
        self.assertAlmostEqual(k['f'] / self.TRUE_K['f'], 1.0, delta=0.005,
                               msg=f'focal {k["f"]}')
        self.assertLess(abs(k['cx'] - self.TRUE_K['cx']), 4.0)
        self.assertLess(abs(k['cy'] - self.TRUE_K['cy']), 4.0)
        self.assertAlmostEqual(k['k'][0], self.TRUE_K['k'][0], delta=0.01,
                               msg=f'k1 {k["k"][0]}')

    def test_recovers_k_and_distortion_from_anchors_alone(self):
        report = self.adjust('anchors')
        self.assert_recovers_k(report)
        self.assertTrue(report['focal_curve']['observable'])
        self.assertEqual(report['divergent'], 0)
        self.assertEqual(report['ties']['used'], 0)
        # Every view ends nearer the truth than its mesh-only solve, which had
        # the wrong K to work with.
        for i, view in enumerate(self.views):
            before = np.linalg.norm(
                calibration_center(self.localized[i] / 'cal.json') - view.C)
            after = np.linalg.norm(
                calibration_center(self.out / f'cal_{i}.json') - view.C)
            self.assertLess(after, before, f'view {i}: {after} >= {before}')
            self.assertLess(after, 0.3, f'view {i}')

    def test_ties_from_a_foreign_frame_are_retriangulated(self):
        report = self.adjust('ties', '--tie-scene', self.dir / 'tie_scene.json')
        self.assertGreater(report['ties']['used'], 100)
        self.assert_recovers_k(report)
        self.assertEqual(report['divergent'], 0)
        # The adjusted ties are on the plane: they were re-triangulated into
        # the mesh's frame, not taken from the scene's scaled, shifted one.
        sfm = json.loads((self.out / 'scene.json').read_text())
        z = np.array([s['value']['X'][2] for s in sfm['structure']])
        self.assertGreater(len(z), 100)
        self.assertLess(np.abs(z).max(), 0.5)

    def test_fixed_intrinsic_is_left_alone(self):
        report = self.adjust('fixed', '--camera', self.start_cam,
                             '--fix-intrinsic')
        k = report['intrinsic']['adjusted']
        self.assertEqual(k['f'], self.START['f'])
        self.assertEqual((k['cx'], k['cy']),
                         (self.START['cx'], self.START['cy']))


@requires_scene
class TestAdjustUntilted(_AdjustFixture):
    """Every view straight down, rotated only about its own axis: a wrong
    focal is absorbed into standoff exactly (ADR 0011), so nothing determines
    it."""

    @staticmethod
    def POSES():
        return [(scene.rot_z(a), (x, y, 60.0)) for a, x, y in
                ((-25, 8.0, 5.0), (15, 14.0, 1.0), (40, 9.0, 10.0),
                 (0, 3.0, 3.0))]

    def test_unobservable_focal_emits_only_what_was_measured(self):
        report = self.adjust('flat')
        self.assertFalse(report['focal_curve']['observable'])
        self.assertEqual(report['intrinsic_emitted'], 'start')
        for i, v in enumerate(self.localized):
            self.assertEqual(report['views'][i]['emitted'], 'mesh-only')
            self.assertEqual((self.out / f'cal_{i}.json').read_bytes(),
                             (v / 'cal.json').read_bytes())
        camera = dict(line.split(None, 1) for line in
                      (self.out / 'camera.txt').read_text().splitlines())
        self.assertEqual(float(camera['fx']), self.START['f'])


@requires_scene
class TestPipeline(_AdjustFixture):
    """``pgs-localize-set`` from the command line, twice."""

    POSES = TestAdjust.POSES

    @classmethod
    def setUpClass(cls):
        try:
            import configargparse  # noqa: F401
        except ImportError:
            raise unittest.SkipTest('needs configargparse')
        super().setUpClass()
        d = cls.dir
        images = d / 'handheld'
        images.mkdir()
        for name in cls.names:
            shutil.copy(d / name, images / name)
        recon = d / 'recon'
        recon.mkdir()
        (recon / 'pgs-recon.json').write_text('{"stages": {}, "commands": {}}')
        cls.set_dir = d / 'set'
        argv = [sys.executable, '-m', 'pgs_recon.viewset.cli',
                '--recon-dir', recon, '--images', images,
                '--output', cls.set_dir, '--mesh', cls.mesh,
                '--ir-reference-mesh', cls.mesh,
                '--overhead-calibration', d / 'overhead_calibration.json',
                '--overhead-image', d / 'overhead.png',
                '--camera', cls.start_cam, '--threads', 2,
                '--path', toolchain.effective_prefix()]
        cls.first = run(argv)
        cls.second = run(argv)
        cls.meta = json.loads((cls.set_dir / 'pgs-localize-set.json')
                              .read_text())

    def test_every_view_is_localized(self):
        views = self.meta['views']
        self.assertEqual(sorted(views), sorted(Path(n).stem
                                               for n in self.names))
        for stem, record in views.items():
            self.assertEqual(record['status'], 'done', stem)

    def test_the_overlapping_views_are_one_group_and_it_solves(self):
        self.assertEqual(len(self.meta['groups']['groups']), 1)
        self.assertEqual(len(self.meta['groups']['groups'][0]), 4)
        self.assertEqual(len(self.meta['groups']['solved']), 1)
        report = json.loads((self.set_dir / 'joint' / 'report.json')
                            .read_text())
        self.assertGreater(report['ties']['used'], 0)

    def test_the_shared_k_is_recovered(self):
        report = json.loads((self.set_dir / 'joint' / 'report.json')
                            .read_text())
        self.assertEqual(report['intrinsic_emitted'], 'adjusted')
        k = report['intrinsic']['adjusted']
        self.assertAlmostEqual(k['f'] / self.TRUE_K['f'], 1.0, delta=0.005)
        self.assertAlmostEqual(k['k'][0], self.TRUE_K['k'][0], delta=0.01)

    def test_each_view_textures_the_mesh(self):
        for name in self.names:
            stem = Path(name).stem
            obj = self.set_dir / 'views' / stem / 'textured' / f'{stem}.obj'
            self.assertTrue(obj.is_file(), obj)
            self.assertIn('vt ', obj.read_text())

    def test_a_second_run_redoes_nothing(self):
        log = self.second.stdout + self.second.stderr
        for name in self.names:
            stem = Path(name).stem
            self.assertIn(f'{stem}: current, skipping', log)
            self.assertIn(f'{stem}: texture current, skipping', log)
        self.assertIn('joint adjustment: current, skipping', log)
        for word in ('localizing', 'adjusting', 'texturing'):
            self.assertNotIn(word, log)

    def test_the_manifest_records_every_phase(self):
        commands = ' '.join(self.meta['commands'].values())
        for marker in ('position-map', ' view ', ' adjust ',
                       'openMVG_main_SfM ', 'pgs_recon.apps.retexture'):
            self.assertIn(marker, commands)
        self.assertEqual(self.meta['joint']['status'], 'done')
        for stem, record in self.meta['views'].items():
            self.assertEqual(record['texture']['status'], 'done', stem)


@requires_scene
class TestSceneConventions(unittest.TestCase):
    """The fixture's renderer and projection are each other's inverse,
    distortion included."""

    def test_projection_inverts_the_render_rays(self):
        view = scene.View(R=scene.rot_x(9) @ scene.DOWN,
                          C=np.array([2.0, 1.0, 60.0]), f=1600.0, cx=403.0,
                          cy=297.0, width=800, height=600,
                          k=(-0.05, 0.01, 0.0))
        pts = scene.plane_points(view)
        rows, cols = np.mgrid[0:600:37, 0:800:41]
        X = pts[rows.ravel(), cols.ravel()]
        uv = scene.project(view, X)
        centres = np.stack([cols.ravel() + 0.5, rows.ravel() + 0.5], 1)
        self.assertLess(np.abs(uv - centres).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
