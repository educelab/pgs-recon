"""The depth test behind ``pgs-retexture``'s projective UV mapping: which faces
one camera actually sees.

Projection alone cannot answer that. A face hidden behind a fold is in view and
facing the camera, so the only thing separating it from the surface in front of
it is depth -- and until this existed, both took the same pixels. The cases here
are the ones that distinguish a real depth test from a plausible one:

  - a face is not occluded by its own neighbours (a smooth surface interpolates
    to the same depth along a shared edge, so nothing there is 'nearer');
  - a face is occluded by an unrelated surface in front of it, whether or not
    that surface is connected to it;
  - a face *straddling* the boundary is neither, and gets a fraction -- the
    whole reason the answer is not a flag;
  - a mesh finer than the pixel grid still occludes. This is the one that fails
    silently: sample only the pixel centres a triangle covers and a sub-pixel
    mesh writes nothing to the depth buffer, so every face reads as visible and
    the test looks like it passed.

Geometry is synthetic and axis-aligned: the camera sits at the origin looking
down +Z with an identity rotation, so a plane at depth d covers pixels at
``f * x / d + cx`` and every expected number is worked out by hand rather than
by the code under test.
"""
import importlib.util
import unittest

MISSING = [d for d in ('numpy',) if importlib.util.find_spec(d) is None]

if not MISSING:
    import numpy as np

    from pgs_recon.utils.visibility import project_points, visible_fraction

    #: Camera at the origin down +Z: unit focal in pixels, centred, 100x100.
    CAM = dict(R=np.eye(3), C=np.zeros(3), f=100.0, cx=49.5, cy=49.5)

W = H = 100


def world(px, z):
    """The world offset that projects to ``px`` pixels off the principal point
    at depth ``z``. Every mesh below is stated in pixels for that reason: a
    patch's size in the image is what the depth test is about, and it is not its
    size in the scene -- the same square is half as wide twice as far away."""
    return px * z / CAM['f']


def quad(z, x0=-40, x1=40, y0=-40, y1=40):
    """Two triangles of a plane at depth ``z``, projecting to the given pixel
    offsets from the principal point."""
    xs, ys = [world(x0, z), world(x1, z)], [world(y0, z), world(y1, z)]
    v = np.array([[xs[0], ys[0], z], [xs[1], ys[0], z],
                  [xs[1], ys[1], z], [xs[0], ys[1], z]], dtype=np.float64)
    return v, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


def triangle(z, points):
    """One triangle at depth ``z`` through pixel-offset ``points``."""
    v = np.array([[world(x, z), world(y, z), z] for x, y in points],
                 dtype=np.float64)
    return v, np.array([[0, 1, 2]], dtype=np.int64)


def grid(z, n, half=40):
    """A plane at depth ``z`` spanning +/- ``half`` pixels, tessellated into
    ``2 * n * n`` triangles."""
    t = np.linspace(world(-half, z), world(half, z), n + 1)
    gx, gy = np.meshgrid(t, t, indexing='xy')
    v = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    i = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b = i[:-1, :-1].ravel(), i[:-1, 1:].ravel()
    c, d = i[1:, 1:].ravel(), i[1:, :-1].ravel()
    return v, np.concatenate([np.column_stack([a, b, c]),
                              np.column_stack([a, c, d])])


def combine(*meshes):
    """Concatenate meshes, returning the vertices, faces, and the face slice
    belonging to each input."""
    verts, faces, spans, base, start = [], [], [], 0, 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + base)
        spans.append(slice(start, start + len(f)))
        base += len(v)
        start += len(f)
    return np.concatenate(verts), np.concatenate(faces), spans


def fractions(V, F, **kwargs):
    """``visible_fraction`` over a mesh, doing the projection the app does."""
    uv, z = project_points(V, **CAM)
    face_uv, face_z = project_points(V[F].mean(axis=1), **CAM)
    return visible_fraction(uv, z, F, W, H, face_uv, face_z, **kwargs)


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class VisibilityCase(unittest.TestCase):
    """The kernel is numpy-only, so the suite runs anywhere numpy does -- and
    skips rather than errors where it does not (CI runs the tests on bare
    Python too)."""


class TestProjection(VisibilityCase):
    """The projection the depth test and the UVs share."""

    def test_the_optical_axis_lands_on_the_principal_point(self):
        uv, z = project_points(np.array([[0.0, 0.0, 5.0]]), **CAM)
        np.testing.assert_allclose(uv[0], [CAM['cx'], CAM['cy']])
        self.assertEqual(z[0], 5.0)

    def test_depth_is_camera_z_not_range(self):
        # 3-4-5: the point is 5 from the camera centre but 4 deep.
        _uv, z = project_points(np.array([[3.0, 0.0, 4.0]]), **CAM)
        self.assertEqual(z[0], 4.0)

    def test_a_point_behind_the_camera_is_marked_by_its_depth(self):
        _uv, z = project_points(np.array([[0.0, 0.0, -5.0]]), **CAM)
        self.assertLess(z[0], 0)

    def test_radial_distortion_moves_the_point_outward(self):
        p = np.array([[0.2, 0.0, 1.0]])
        straight, _ = project_points(p, **CAM)
        bent, _ = project_points(p, disto=[0.1, 0.0, 0.0], **CAM)
        # k1 r^2 = 0.1 * 0.04 = 0.004 of the 20 px offset.
        self.assertAlmostEqual(bent[0, 0] - CAM['cx'],
                               (straight[0, 0] - CAM['cx']) * 1.004)


class TestAnUnobstructedSurfaceIsFullyVisible(VisibilityCase):
    """The null case: with nothing in front, nothing is occluded. What this
    really guards is self-occlusion -- every face shares pixels with its
    neighbours, and one of them is always marginally nearer."""

    def test_a_facing_plane_keeps_every_face(self):
        V, F = quad(5.0)
        np.testing.assert_array_equal(fractions(V, F), [1.0, 1.0])

    def test_a_tessellated_plane_does_not_occlude_itself(self):
        V, F = grid(5.0, 12)
        self.assertEqual(fractions(V, F).min(), 1.0)

    def test_a_tilted_plane_does_not_occlude_itself(self):
        # 45 degrees: depth runs 4 -> 6 across the face, so adjacent faces sit
        # at visibly different depths and still interpolate to the same depth
        # along the edge they share. (A face nothing samples would read 0.0
        # here too, so this also holds that the tilted plane stays in frame.)
        V, F = grid(5.0, 12, half=20)
        V[:, 2] += V[:, 0]
        self.assertEqual(fractions(V, F).min(), 1.0)

    def test_a_tilted_sub_pixel_plane_does_not_occlude_itself(self):
        # The hard version, and the one the slope-scaled tolerance is for:
        # faces smaller than a pixel *and* a depth gradient, so several
        # neighbours land in one pixel at genuinely different depths. A
        # depth-only tolerance speckles this surface with false occlusions.
        V, F = grid(5.0, 90, half=20)
        V[:, 2] += V[:, 0]
        self.assertEqual(fractions(V, F).min(), 1.0)

    def test_a_face_behind_the_camera_is_not_visible(self):
        V, F = quad(-5.0)
        np.testing.assert_array_equal(fractions(V, F), [0.0, 0.0])


class TestSomethingNearerHidesWhatIsBehindIt(VisibilityCase):
    """The case back-face culling cannot see: the hidden face is in view and
    facing the camera, and only depth separates it from the occluder."""

    def setUp(self):
        # A far plane 80 px across in 8x8 cells of 10 px, and a 20x20 px patch
        # at half its depth centred over it -- covering the middle four cells,
        # which is eight of its faces.
        self.V, self.F, self.spans = combine(grid(10.0, 8),
                                             quad(5.0, -10, 10, -10, 10))
        self.far, self.near = self.spans

    def test_the_occluder_itself_stays_visible(self):
        np.testing.assert_array_equal(fractions(self.V, self.F)[self.near],
                                      [1.0, 1.0])

    def test_it_hides_exactly_the_faces_it_covers(self):
        frac = fractions(self.V, self.F)[self.far]
        self.assertEqual(int((frac == 0.0).sum()), 8)
        self.assertEqual(int((frac == 1.0).sum()), len(frac) - 8)

    def test_a_face_fully_behind_the_occluder_is_zero(self):
        V, F, (far, _near) = combine(
            triangle(10.0, [(-5, -5), (5, -5), (0, 5)]),
            quad(5.0, -10, 10, -10, 10))
        self.assertEqual(fractions(V, F)[far][0], 0.0)

    def test_a_face_the_occluder_straddles_is_a_fraction(self):
        # A far triangle running from 30 px left of the image centre to it,
        # so the occluder's edge at -10 px crosses it.
        V, F, (far, _near) = combine(
            triangle(10.0, [(-30, -8), (0, -8), (-15, 8)]),
            quad(5.0, -10, 10, -10, 10))
        frac = fractions(V, F)[far][0]
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0)

    def test_the_occluder_need_not_touch_what_it_hides(self):
        # The two meshes share no vertex, which is the point: connectivity has
        # nothing to do with what is in front of what.
        self.assertEqual(len(np.intersect1d(self.F[self.far],
                                            self.F[self.near])), 0)
        self.assertIn(0.0, fractions(self.V, self.F)[self.far])


class TestASubPixelMeshStillOccludes(VisibilityCase):
    """A camera standing further off than the rig that solved the mesh sees
    faces smaller than a pixel. Sampling only covered pixel centres would leave
    the depth buffer nearly empty and pass everything as visible."""

    def test_a_mesh_finer_than_the_pixel_grid_is_still_hidden(self):
        # 80 px across in 60 cells: every face is well under one pixel.
        V, F, (far, near) = combine(grid(10.0, 60),
                                    quad(5.0, -20, 20, -20, 20))
        frac = fractions(V, F)
        self.assertEqual(frac[near].min(), 1.0)
        # The occluder covers the middle 40x40 px of an 80x80 px plane: a
        # quarter of its area, so a quarter of its faces.
        hidden = float((frac[far] == 0.0).mean())
        self.assertGreater(hidden, 0.2)
        self.assertLess(hidden, 0.3)

    def test_a_sub_pixel_occluder_hides_what_is_behind_it(self):
        # Now the *occluder* is the fine one: it has to reach the depth buffer
        # even where no face of it covers a pixel centre.
        V, F, (far, near) = combine(quad(10.0), grid(5.0, 60, half=20))
        frac = fractions(V, F)
        self.assertEqual(frac[near].min(), 1.0)
        self.assertTrue((frac[far] < 1.0).all())


class TestTheBiasSetsWhatCountsAsNearer(VisibilityCase):
    """The tolerance is relative to depth, so it holds whatever the scene is
    scaled in -- and it is what keeps mesh noise from reading as occlusion."""

    def mesh(self):
        return combine(quad(10.0), quad(9.999, -10, 10, -10, 10))

    def test_a_gap_under_the_bias_does_not_occlude(self):
        V, F, (far, _near) = self.mesh()
        self.assertEqual(fractions(V, F, bias=1e-2)[far].min(), 1.0)

    def test_the_same_gap_occludes_once_the_bias_is_tightened(self):
        V, F, (far, _near) = self.mesh()
        self.assertLess(fractions(V, F, bias=1e-6)[far].min(), 1.0)


class TestDegenerateInputIsAnAnswerNotACrash(VisibilityCase):

    def test_a_face_entirely_off_image_is_not_visible(self):
        V, F = quad(5.0, 500, 600, 500, 600)
        np.testing.assert_array_equal(fractions(V, F), [0.0, 0.0])

    def test_a_zero_area_face_is_answered_by_its_centroid(self):
        V, F = triangle(5.0, [(0, 0), (10, 0), (20, 0)])
        np.testing.assert_array_equal(fractions(V, F), [1.0])

    def test_a_face_crossing_the_camera_plane_is_not_visible(self):
        V = np.array([[0.0, 0.0, 5.0], [0.1, 0.0, -5.0], [0.0, 0.1, 5.0]])
        F = np.array([[0, 1, 2]])
        np.testing.assert_array_equal(fractions(V, F), [0.0])

    def test_a_face_larger_than_the_rasterizer_tile_is_one_face(self):
        # Bounding boxes are split into 32 px tiles internally; a face covering
        # the whole image spans a dozen of them and must still read as one.
        V, F = quad(1.0, -49, 49, -49, 49)
        np.testing.assert_array_equal(fractions(V, F), [1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
