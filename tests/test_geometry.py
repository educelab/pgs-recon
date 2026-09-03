"""``pgs_recon.utils.geometry``'s ground segmentation and component filters.

``pgs-remove-ground-plane`` used to fit a plane and delete its inlier band.
Real scan beds are bowed by far more than the mesh's own noise, so that band
only ever caught the strip where plane and bed happened to coincide, leaving
most of the ground behind. ``segment_ground_surface`` fits a polynomial in the
plane's frame instead; the cases below are that difference, plus the
least-squares refit ``segment_plane`` silently skipped.

The component filters follow: what ``cluster_connected_components`` counts as
one component, and the area threshold ``pgs-remove-ground-plane`` and
``pgs-filter-small-components`` expose as ``--filter-cc-area``.

Skips when numpy or scipy is absent (``geometry`` imports both at module load).
"""
import importlib.util
import unittest

MISSING = [d for d in ('numpy', 'scipy')
           if importlib.util.find_spec(d) is None]

# A bowed ground: a paraboloid sagging 0.5 over a 20x20 bed, plus a small
# block of "object" sitting on top of it. Noise is well under the sag, which
# is what makes a plane fit inadequate.
SAG = 0.5
EXTENT = 10.
NOISE = 0.002


def build_scene(sag=SAG):
    import numpy as np
    rng = np.random.default_rng(4)
    gx, gy = np.meshgrid(np.linspace(-EXTENT, EXTENT, 160),
                         np.linspace(-EXTENT, EXTENT, 160))
    gx, gy = gx.ravel(), gy.ravel()
    gz = sag * (gx ** 2 + gy ** 2) / (2 * EXTENT ** 2)
    gz = gz + rng.normal(0., NOISE, gz.shape)
    ground = np.stack([gx, gy, gz], axis=-1)

    ox, oy = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40))
    ox, oy = ox.ravel(), oy.ravel()
    obj = np.stack([ox, oy, np.full(ox.shape, 3.)], axis=-1)

    return ground, obj


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestSegmentGroundSurface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.ground, cls.obj = build_scene()
        cls.vertices = np.concatenate([cls.ground, cls.obj])

    def mesh(self):
        from pgs_recon.utils import geometry as geom
        mesh = geom.Mesh()
        mesh.vertices = self.vertices.copy()
        return mesh

    def segment(self, **kwargs):
        from pgs_recon.utils import geometry as geom
        kwargs.setdefault('dist_threshold', 5 * NOISE)
        kwargs.setdefault('seed', 0)
        return geom.segment_ground_surface(self.mesh(), **kwargs)

    def test_finds_the_whole_bowed_ground(self):
        _, inliers = self.segment()
        self.assertGreater(len(inliers), 0.99 * len(self.ground))

    def test_leaves_the_object_alone(self):
        import numpy as np
        _, inliers = self.segment()
        # The object's vertices are the tail of the concatenation
        self.assertEqual(np.sum(np.asarray(inliers) >= len(self.ground)), 0)

    def test_a_plane_only_catches_a_strip(self):
        """The regression this replaces: same threshold, a plane's band."""
        from pgs_recon.utils import geometry as geom
        _, inliers = geom.segment_plane(self.mesh(), dist_threshold=5 * NOISE,
                                        seed=0)
        self.assertLess(len(inliers), 0.5 * len(self.ground))

    def test_reports_the_warp(self):
        surface, _ = self.segment()
        self.assertAlmostEqual(surface.warp, SAG, delta=0.05)

    def test_reports_the_fit_rms(self):
        """The number a caller compares against its distance threshold."""
        import numpy as np
        surface, inliers = self.segment()
        dist = surface.signed_distance(self.vertices)[inliers]
        self.assertAlmostEqual(surface.rms, float(np.sqrt((dist ** 2).mean())),
                               places=9)
        # Well inside the band, and of the order of the noise put in
        self.assertLess(surface.rms, 0.4 * 5 * NOISE)
        self.assertAlmostEqual(surface.rms, NOISE, delta=NOISE)

    def test_signed_distance_separates_ground_from_object(self):
        import numpy as np
        surface, _ = self.segment()
        dist = surface.signed_distance(self.vertices)
        self.assertLess(np.abs(dist[:len(self.ground)]).max(), 5 * NOISE)
        self.assertGreater(np.abs(dist[len(self.ground):]).min(), 1.)

    def test_rejects_a_negative_degree(self):
        with self.assertRaises(ValueError):
            self.segment(degree=-1)

    def test_refuses_a_threshold_that_cannot_describe_the_ground(self):
        """A plane over a bowed bed fills its band instead of hugging it."""
        with self.assertRaises(ValueError):
            self.segment(degree=0)


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestSegmentGroundSurfaceFlat(unittest.TestCase):
    """On an unbowed bed there is no warp to find, so degree 0 is enough."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.ground, cls.obj = build_scene(sag=0.)
        cls.vertices = np.concatenate([cls.ground, cls.obj])

    def segment(self, **kwargs):
        from pgs_recon.utils import geometry as geom
        mesh = geom.Mesh()
        mesh.vertices = self.vertices.copy()
        return geom.segment_ground_surface(mesh, dist_threshold=5 * NOISE,
                                           seed=0, **kwargs)

    def test_degree_zero_matches_the_plane(self):
        from pgs_recon.utils import geometry as geom
        mesh = geom.Mesh()
        mesh.vertices = self.vertices.copy()
        _, plane = geom.segment_plane(mesh, dist_threshold=5 * NOISE, seed=0)
        surface, flat = self.segment(degree=0)
        # Not identical: degree 0 still refits the offset over every inlier
        self.assertAlmostEqual(len(flat) / len(plane), 1., delta=0.02)
        self.assertAlmostEqual(surface.warp, 0.)

    def test_degree_two_finds_no_warp(self):
        surface, inliers = self.segment(degree=2)
        self.assertGreater(len(inliers), 0.99 * len(self.ground))
        self.assertLess(surface.warp, 10 * NOISE)


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestSegmentPlaneRefit(unittest.TestCase):
    """The returned model is a fit to every inlier, not to three of them."""

    def test_refits_the_model_over_all_inliers(self):
        import numpy as np
        from pgs_recon.utils import geometry as geom
        rng = np.random.default_rng(7)
        pts = np.zeros((4000, 3))
        pts[:, :2] = rng.uniform(-10., 10., (4000, 2))
        pts[:, 2] = rng.normal(0., 0.01, 4000)
        mesh = geom.Mesh()
        mesh.vertices = pts
        model, inliers = geom.segment_plane(mesh, dist_threshold=0.05, seed=0)
        # A three-point cross product over noise this shallow tilts wildly; a
        # least-squares fit recovers +/-z to well under a degree.
        self.assertGreater(abs(model[2]), 0.9999)


def build_mesh(squares):
    """A mesh of disjoint squares, each ``(x, y, side)``, two faces apiece"""
    import numpy as np
    from pgs_recon.utils import geometry as geom
    vertices, faces = [], []
    for x, y, side in squares:
        v = len(vertices)
        vertices += [[x, y, 0.], [x + side, y, 0.],
                     [x + side, y + side, 0.], [x, y + side, 0.]]
        faces += [[[v, None, None], [v + 1, None, None], [v + 2, None, None]],
                  [[v, None, None], [v + 2, None, None], [v + 3, None, None]]]
    mesh = geom.Mesh()
    mesh.vertices = np.array(vertices, dtype=float)
    mesh.faces = np.array(faces, dtype='O')
    mesh.normals = np.zeros((0, 3))
    mesh.uv_coords = np.zeros((0, 2))
    mesh.mtl_ids = np.zeros((0,), dtype=int)
    return mesh


def areas(inventory_side):
    """Component areas, rounded past the cross products' last bits"""
    return sorted(round(a, 6) for a in inventory_side.tolist())


# Three squares far enough apart to share nothing: 4, 1 and 0.0001 units^2,
# the shape of a real reconstruction -- one artifact, and speckle
SQUARES = [(0., 0., 2.), (10., 0., 1.), (20., 0., 0.01)]
AREAS = [0.0001, 1., 4.]


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestClusterConnectedComponents(unittest.TestCase):

    def test_measures_every_component(self):
        import numpy as np
        from pgs_recon.utils import geometry as geom
        face_cluster, areas, sizes = geom.cluster_connected_components(
            build_mesh(SQUARES))
        self.assertEqual(areas.shape, (3,))
        self.assertEqual(sorted(round(a, 6) for a in areas.tolist()), AREAS)
        # Every face lands in exactly one cluster, and the sizes say so
        self.assertEqual(face_cluster.shape, (6,))
        self.assertEqual(sizes.tolist(), [2, 2, 2])
        self.assertEqual(np.bincount(face_cluster).tolist(), sizes.tolist())

    def test_a_shared_vertex_is_not_a_connection(self):
        """Faces are linked by shared edges, as the BFS this replaces was."""
        import numpy as np
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh([(0., 0., 1.)])
        # Hinge the second face on the first one's far corner: one vertex in
        # common, no edge
        mesh.vertices = np.concatenate([mesh.vertices, [[2., 2., 0.]]])
        mesh.faces[1] = [[2, None, None], [3, None, None], [4, None, None]]
        _, areas, _ = geom.cluster_connected_components(mesh)
        self.assertEqual(areas.shape, (2,))

    def test_handles_a_mesh_with_no_faces(self):
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh(SQUARES)
        geom.keep_triangles_by_mask(mesh, [])
        for part in geom.cluster_connected_components(mesh):
            self.assertEqual(part.shape, (0,))
        inventory = geom.remove_connected_components_by_area(mesh, 0.5)
        self.assertEqual((inventory.kept.size, inventory.dropped.size), (0, 0))

    def test_face_areas_are_true_areas(self):
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh([(0., 0., 2.)])
        # Tilt one edge of the square out of plane: the footprint shrinks, the
        # surface area does not
        mesh.vertices[2][2] = mesh.vertices[3][2] = 2. * 3 ** 0.5
        self.assertAlmostEqual(float(geom.face_areas(mesh).sum()), 8.)


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestFilterByArea(unittest.TestCase):
    """``--filter-cc-area``: keep components of at least so many units^2."""

    def filter(self, min_area, squares=SQUARES):
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh(squares)
        return mesh, geom.remove_connected_components_by_area(mesh, min_area)

    def test_drops_the_components_under_the_threshold(self):
        mesh, inventory = self.filter(0.5)
        self.assertEqual(areas(inventory.kept), [1., 4.])
        self.assertEqual(areas(inventory.dropped), [0.0001])
        self.assertEqual(mesh.faces.shape[0], 4)
        self.assertEqual(mesh.vertices.shape[0], 8)

    def test_keeps_a_component_exactly_at_the_threshold(self):
        _, inventory = self.filter(1.)
        self.assertEqual(areas(inventory.kept), [1., 4.])

    def test_zero_keeps_every_component(self):
        """The pipeline's 'no filtering' spelling, and how you measure."""
        mesh, inventory = self.filter(0.)
        self.assertEqual(areas(inventory.kept), AREAS)
        self.assertEqual(inventory.dropped.size, 0)
        self.assertEqual(mesh.faces.shape[0], 6)
        self.assertEqual(mesh.vertices.shape[0], 12)

    def test_prunes_orphan_vertices_even_when_nothing_is_dropped(self):
        """What the caller removed first (the ground) leaves vertices behind.

        ``pgs-remove-ground-plane`` deletes the ground's faces before it
        filters, and a vertex whose every face went with them survives that.
        The filters have always been where those get cleaned up, so a filter
        that happens to drop no component still has to do it.
        """
        import numpy as np
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh([(0., 0., 2.)])
        # Strand a vertex: drop the faces that reference it, keep the vertex
        geom.keep_triangles_by_mask(mesh, [0])
        self.assertEqual(mesh.vertices.shape[0], 4)
        inventory = geom.remove_connected_components_by_area(mesh, 0.)
        self.assertEqual(inventory.dropped.size, 0)
        self.assertEqual(mesh.vertices.shape[0], 3)
        self.assertEqual(np.unique(mesh.faces[..., 0].astype(int)).tolist(),
                         [0, 1, 2])

    def test_a_threshold_above_everything_empties_the_mesh(self):
        mesh, inventory = self.filter(10.)
        self.assertEqual(inventory.kept.size, 0)
        self.assertEqual(areas(inventory.dropped), AREAS)
        self.assertEqual(mesh.faces.shape[0], 0)
        self.assertEqual(mesh.vertices.shape[0], 0)
        # An empty mesh is a threshold in the wrong units, so say so
        self.assertIn('nothing left', str(inventory))

    def test_keeps_the_surviving_faces_in_order(self):
        mesh, _ = self.filter(0.5)
        self.assertEqual(mesh.faces[..., 0].astype(int).tolist(),
                         [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])

    def test_reports_what_it_did(self):
        _, inventory = self.filter(0.5)
        line = str(inventory)
        self.assertIn('Kept 2 component(s), 5 units^2', line)
        self.assertIn('dropped 1, 0.0001 units^2', line)

    def test_says_nothing_about_units_while_something_survives(self):
        _, inventory = self.filter(0.5)
        self.assertNotIn('nothing left', str(inventory))

    def test_area_says_what_a_face_count_cannot(self):
        """Why the flag exists: the speck and the artifact are both 2 faces."""
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh(SQUARES)
        by_size = geom.remove_connected_components_by_size(mesh, num_faces=2)
        self.assertEqual(areas(by_size.kept), AREAS)
        _, by_area = self.filter(0.5)
        self.assertEqual(areas(by_area.kept), [1., 4.])


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestFilterBySizeAndLargest(unittest.TestCase):
    """The filters that predate the area one, over the same inventory."""

    def test_size_drops_by_face_count(self):
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh(SQUARES)
        inventory = geom.remove_connected_components_by_size(mesh, num_faces=3)
        self.assertEqual(inventory.kept.size, 0)
        self.assertEqual(areas(inventory.dropped), AREAS)
        self.assertEqual(mesh.faces.shape[0], 0)

    def test_largest_keeps_one_component(self):
        from pgs_recon.utils import geometry as geom
        mesh = build_mesh(SQUARES)
        inventory = geom.keep_largest_connected_component(mesh)
        self.assertEqual(areas(inventory.kept), [4.])
        self.assertEqual(areas(inventory.dropped), [0.0001, 1.])
        self.assertEqual(mesh.faces.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
