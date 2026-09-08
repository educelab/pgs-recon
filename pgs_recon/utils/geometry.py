# Much of this file is a port of code originally found in the Open3D library,
# but changed to fit this project's purposes: https://github.com/isl-org/Open3D

import logging
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pgs_recon.utils import wavefront as wobj


class Mesh:
    """Custom mesh class"""
    vertices: np.ndarray = None
    faces: np.ndarray = None
    normals: np.ndarray = None
    uv_coords: np.ndarray = None
    mtl_ids: np.ndarray = None


def wavefront_to_mesh(obj: wobj.WavefrontOBJ) -> Mesh:
    """Convert a Wavefront object to a Mesh"""
    mesh = Mesh()
    mesh.vertices = np.array(obj.vertices)
    mesh.faces = np.array(obj.polygons)
    mesh.normals = np.array(obj.normals)
    mesh.uv_coords = np.array(obj.texcoords)
    mesh.mtl_ids = np.array(obj.mtlid)
    return mesh


def mesh_to_wavefront(mesh: Mesh, obj: wobj.WavefrontOBJ = None):
    """Convert a Mesh to a Wavefront object. If obj is provided, its geometry
    will be modified."""
    if obj is None:
        obj = wobj.WavefrontOBJ()
    obj.vertices = [v.tolist() for v in mesh.vertices]
    obj.polygons = [[v.tolist() for v in f] for f in mesh.faces]
    obj.normals = [n.tolist() for n in mesh.normals]
    obj.texcoords = [uv.tolist() for uv in mesh.uv_coords]
    obj.mtlid = mesh.mtl_ids.tolist()

    return obj


def segment_plane(mesh, dist_threshold=0.1, point_samples=3, iterations=1000,
                  prob=0.99999999, seed=None):
    class RANSACResult:
        # The inlier set stays a boolean mask: materializing an index list per
        # iteration costs more than the distances it is derived from
        mask: np.ndarray
        count: int
        error: float
        fitness: float
        inlier_rmse: float

        def __init__(self):
            self.mask = np.zeros((0,), dtype=bool)
            self.count = 0
            self.error = 0.
            self.fitness = 0.
            self.inlier_rmse = 0.

    def get_plane_from_points(pts: np.ndarray):
        """Fit a plane to a set of points"""
        if len(pts) == 3:
            e0 = pts[1] - pts[0]
            e1 = pts[2] - pts[0]
            abc = np.cross(e0, e1)
            norm = np.linalg.norm(abc)
            if np.isclose(norm, 0.):
                return None
            abc /= norm
            d = -np.dot(abc, pts[0])
        else:
            centroid = np.mean(pts, axis=0)
            cov = np.cov(pts - centroid, rowvar=False)
            val, vec = np.linalg.eigh(cov)
            abc = vec[:, np.argmin(val, axis=0)]
            d = -np.dot(abc, centroid)
        x = np.concatenate([abc, (d,)])
        return x

    def eval_from_distance(pts: np.ndarray, plane: np.ndarray,
                           threshold: float):
        """Evaluate plane fit against a set of points"""
        res = RANSACResult()
        dist = pts @ plane[:3] + plane[3]
        np.abs(dist, out=dist)
        res.mask = dist < threshold
        res.count = int(np.count_nonzero(res.mask))
        res.error = float(np.sum(dist, where=res.mask))

        if res.count > 0:
            res.fitness = res.count / pts.shape[0]
            res.inlier_rmse = res.error / np.sqrt(res.count)

        return res

    # Setup outputs
    best_result = RANSACResult()
    best_model = np.zeros((4,))

    # Iterate up to some max iterations
    rng = np.random.default_rng(seed=seed)
    break_it = iterations
    for i in range(iterations):
        # Break early based on fitness/rmse
        if i > break_it:
            break

        # Fit a plane to N random vertices
        samples = rng.choice(mesh.vertices, point_samples, replace=False)
        model = get_plane_from_points(samples)

        # Skip if the model calculation failed
        if model is None:
            continue

        # Evaluate the model against the entire pointset
        result = eval_from_distance(mesh.vertices, model, dist_threshold)

        # Update our best results if the fitness/rmse are better
        better_fitness = result.fitness > best_result.fitness
        better_rmse = result.fitness == best_result.fitness and result.inlier_rmse < best_result.inlier_rmse
        if better_fitness or better_rmse:
            best_result = result
            best_model = model
            if result.fitness < 1:
                break_it = min(iterations, np.log(1 - prob) / np.log(
                    1 - result.fitness ** point_samples))
            else:
                break

    # get final inlier set
    final_result = eval_from_distance(mesh.vertices, best_model, dist_threshold)
    # update the model using all inliers
    best_model = get_plane_from_points(mesh.vertices[final_result.mask])

    return best_model, final_result.mask.nonzero()[0].tolist()


# Rows per chunk when projecting/fitting over a whole mesh
_POLY_CHUNK = 1 << 20

# Inlier rms, as a fraction of the distance threshold, above which the inlier
# set is a volume rather than a sheet. Uniform over the band gives 0.577
_SHEET_RMS = 0.4


def _poly_terms(x: np.ndarray, y: np.ndarray, degree: int):
    """Monomials of a bivariate polynomial, evaluated at (x, y)"""
    terms = [np.ones_like(x)]
    for d in range(1, degree + 1):
        terms.extend(x ** (d - i) * y ** i for i in range(d + 1))
    return np.stack(terms, axis=-1)


def _fit_poly(x: np.ndarray, y: np.ndarray, z: np.ndarray, degree: int):
    """Least-squares fit of z = p(x, y). Chunked: the design matrix over a
    dense mesh is far larger than the normal equations it accumulates into."""
    n_terms = (degree + 1) * (degree + 2) // 2
    ata = np.zeros((n_terms, n_terms))
    atz = np.zeros((n_terms,))
    for lo in range(0, x.shape[0], _POLY_CHUNK):
        a = _poly_terms(x[lo:lo + _POLY_CHUNK], y[lo:lo + _POLY_CHUNK], degree)
        ata += a.T @ a
        atz += a.T @ z[lo:lo + _POLY_CHUNK]
    return np.linalg.lstsq(ata, atz, rcond=None)[0]


def _eval_poly(x: np.ndarray, y: np.ndarray, coefficients, degree: int):
    out = np.empty(x.shape[0])
    for lo in range(0, x.shape[0], _POLY_CHUNK):
        end = lo + _POLY_CHUNK
        out[lo:end] = _poly_terms(x[lo:end], y[lo:end], degree) @ coefficients
    return out


class GroundSurface:
    """The ground as a polynomial height field over a plane.

    ``warp`` is the peak-to-peak of that height field across the ground, i.e.
    how far the ground departs from being a plane at all. ``rms`` is how
    tightly the ground hugs the fitted surface, and is the number to compare
    against the distance threshold when judging whether that threshold suits a
    capture: a healthy fit lands around a tenth to a quarter of it.

    The frame is oriented: ``normal`` points away from the ground and
    ``signed_distance`` is height above it, positive for anything standing on
    the bed and negative for anything cut into it.
    """

    def __init__(self, origin, basis, scale, coefficients, degree, warp, rms):
        self.origin = origin
        self.basis = basis
        self.scale = scale
        self.coefficients = coefficients
        self.degree = degree
        self.warp = warp
        self.rms = rms

    @property
    def normal(self):
        return self.basis[2]

    def project(self, points: np.ndarray):
        """Points in the surface's frame: (in-plane x, in-plane y, height)"""
        p = points - self.origin
        return (p @ self.basis[0] / self.scale,
                p @ self.basis[1] / self.scale,
                p @ self.basis[2])

    def signed_distance(self, points: np.ndarray):
        """Height above the ground surface, positive away from the ground"""
        x, y, h = self.project(points)
        return h - _eval_poly(x, y, self.coefficients, self.degree)


def segment_ground_surface(mesh, dist_threshold=0.02, degree=2,
                           refinements=12, seed=None, **ransac_kwargs):
    """Segment the ground and return (GroundSurface, inlier vertex indices).

    RANSAC finds the ground's orientation reliably but not its shape: a scan
    bed is bowed by far more than the mesh's noise, so a plane's inlier band
    only ever catches the strip where the two happen to coincide. Fit a
    degree-``degree`` polynomial in the plane's own frame instead, and refit it
    until the inlier set settles -- the band stays at ``dist_threshold``
    throughout, so each pass reaches a little further along the bed without
    ever opening wide enough to let the object bend the fit.

    ``degree=0`` is the plane itself, i.e. ``segment_plane``'s inlier set.
    """
    logger = logging.getLogger(__name__)
    if degree < 0:
        raise ValueError(f'ground surface degree must be >= 0: {degree}')
    n_terms = (degree + 1) * (degree + 2) // 2

    # Orientation from RANSAC
    _, inliers = segment_plane(mesh, dist_threshold=dist_threshold, seed=seed,
                               **ransac_kwargs)
    if len(inliers) < n_terms:
        raise ValueError('too few plane inliers to fit a ground surface')

    # The plane's frame: in-plane axes first, normal last
    ground = mesh.vertices[inliers]
    origin = ground.mean(axis=0)
    basis = np.linalg.svd(ground - origin, full_matrices=False)[2]

    # Normalize the in-plane extent so the higher-order terms stay conditioned
    x, y, h = ((mesh.vertices - origin) @ basis.T).T
    scale = max(np.abs(x).max(), np.abs(y).max())
    x /= scale
    y /= scale

    mask = index_to_boolean_mask(inliers, mesh.vertices.shape[0:1])
    for i in range(refinements):
        coefficients = _fit_poly(x[mask], y[mask], h[mask], degree)
        residual = h - _eval_poly(x, y, coefficients, degree)
        refit = np.abs(residual) < dist_threshold
        count = int(np.count_nonzero(refit))
        if count < n_terms:
            raise ValueError('ground surface fit collapsed; '
                             'distance threshold may be too small')
        moved = int(np.count_nonzero(refit != mask))
        mask = refit
        logger.debug(f'ground fit {i}: {count} inliers, {moved} moved, '
                     f'rms {np.sqrt(np.mean(residual[mask] ** 2)):.5f}')
        if moved <= count // 10000:
            break

    # A sheet's residuals hug the surface. Residuals that instead fill the band
    # (uniform would be 1/sqrt(3) of it) mean the surface has bent into the
    # object, which is what a threshold wider than the object's own relief does
    rms = float(np.sqrt(np.mean(residual[mask] ** 2)))
    if rms > _SHEET_RMS * dist_threshold:
        # A fraction well above the ground's own share of the mesh means the
        # band reached the object; near it, the band is down in the noise
        raise ValueError(
            f'ground surface fills its distance threshold (rms {rms:.5f} of '
            f'{dist_threshold}, covering {count / mask.shape[0]:.1%} of the '
            f'mesh) -- the threshold is below the mesh noise, or wide enough '
            f'to reach the object')

    # The SVD picks its normal's sign off the vertex data, so the same bed
    # reads +z on one capture and -z on the next -- a caller asking which side
    # of the bed something is on cannot use that. Point the normal away from
    # the ground instead, which makes signed_distance a height above it.
    #
    # Which side is up is a question of how far the fit's outliers reach, not
    # how many of them there are. An artifact standing on the bed clears it by
    # centimetres, while nothing reconstructs far beneath an opaque bed: the
    # recesses cut into it bottom out a few millimetres down. Counting instead
    # inverts on a small fragment ringed by the bed's fiducial recesses, which
    # outnumber its vertices while reaching a thirtieth as far -- and an
    # inverted frame is worse than an unoriented one, because
    # remove_connected_components_below_surface would then deliver the
    # recesses and drop the fragment. Compare a high quantile of each side
    # rather than its extreme, so that one stray vertex under the bed cannot
    # decide it either.
    #
    # Only basis[2] and coefficients reach signed_distance, so negating both
    # is exact -- nothing needs refitting. The in-plane axes stay as they were
    # fitted, which can leave the frame left-handed; nothing reads it as a
    # rotation.
    def _reach(side):
        return float(np.percentile(side, 99)) if side.size else 0.
    up = _reach(residual[residual >= dist_threshold])
    down = _reach(-residual[residual <= -dist_threshold])
    if down > up:
        logger.debug(f'flipping the ground normal: the fit came out inverted '
                     f'(outliers reach {up:.4f} one way, {down:.4f} the other)')
        basis[2] = -basis[2]
        coefficients = -coefficients

    # Warp is what a plane could not have absorbed: the height field over the
    # ground, less its own best-fit plane
    gx, gy = x[mask], y[mask]
    height = _eval_poly(gx, gy, coefficients, degree)
    height -= _eval_poly(gx, gy, _fit_poly(gx, gy, height, 1), 1)
    surface = GroundSurface(origin, basis, scale, coefficients, degree,
                            float(height.max() - height.min()), rms)
    return surface, mask.nonzero()[0].tolist()


def index_to_boolean_mask(mask, shape):
    """Convert an index mask to a boolean mask"""
    bool_mask = np.zeros(shape, dtype=bool)
    bool_mask[mask] = True
    return bool_mask


def keep_vertices_by_mask(mesh: Mesh, mask):
    # get the vertices
    len_v = mesh.vertices.shape[0]
    mesh.vertices = mesh.vertices[mask, ...]

    # keep faces which don't reference the removed vertices
    tri_mask = mask[mesh.faces[..., 0].astype(int)]
    tri_mask = np.all(tri_mask, axis=-1)
    keep_triangles_by_mask(mesh, tri_mask)

    # LUT for (vid + None,) -> new_vid
    # Unlike other mesh properties, vid's should never be None.
    # Int, because take's out= below is the mesh's own index column: gathering
    # floats into it casts across dtype kinds, which numpy 2.5 deprecates
    lut = np.full((len_v,), -1, dtype=int)
    new_idx = np.arange(mask.nonzero()[0].shape[0], dtype=int)
    lut[mask] = new_idx
    v_map = mesh.faces[..., 0].astype(int)
    lut.take(v_map, out=v_map)
    v_map = v_map.astype('O')
    mesh.faces[..., 0] = v_map


def remove_vertices_by_index(mesh: Mesh, index_mask):
    # convert the mask to a boolean mask
    mask = index_to_boolean_mask(index_mask, shape=mesh.vertices.shape[0:1])
    # keep vertices not in the mask
    keep_vertices_by_mask(mesh, np.invert(mask))


def remove_unreferenced_vertices(mesh: Mesh):
    # get a list of unique vertices attached to faces
    index_mask = list(set(mesh.faces[..., 0].flatten().tolist()))
    # convert to a boolean mask
    mask = index_to_boolean_mask(index_mask, mesh.vertices.shape[0:1])
    # keep these
    keep_vertices_by_mask(mesh, mask)


def keep_triangles_by_mask(mesh: Mesh, mask):
    # Remove triangles
    faces = mesh.faces[mask, ...]

    # Remove mtl ids
    if mesh.mtl_ids.shape[0] > 0:
        mesh.mtl_ids = mesh.mtl_ids[mask, ...]

    # Filter out unreferenced uvs/normals and update refs
    for a in (1, 2):
        # get the remaining set of unique uv/normal indices
        a_map = faces[..., a]
        a_mask = a_map[a_map != None]
        a_mask = list(set(a_mask.tolist()))
        # keep only the referenced indices
        if a == 1:
            len_a = mesh.uv_coords.shape[0]
            mesh.uv_coords = mesh.uv_coords[a_mask, ...]
        else:
            len_a = mesh.normals.shape[0]
            mesh.normals = mesh.normals[a_mask, ...]
        # Construct a LUT for (id + None,) -> new_id
        lut = np.full((len_a + 1,), -1, dtype=int)
        new_idx = np.arange(len(a_mask), dtype=int)
        lut[a_mask] = new_idx
        # Replace None with -1. Will map to -1 at end of LUT
        a_map[a_map == None] = -1
        # Apply the LUT
        a_map = a_map.astype(int)
        lut.take(a_map, out=a_map)
        # Convert back to None
        a_map = a_map.astype('O')
        a_map[a_map == -1] = None
        # Assign to mesh
        faces[..., a] = a_map

    # Update the mesh
    mesh.faces = faces


def face_areas(mesh: Mesh) -> np.ndarray:
    """Every face's area, in the mesh's units squared"""
    tris = mesh.vertices[mesh.faces[..., 0].astype(np.int64)]
    cross = np.cross(tris[:, 0] - tris[:, 1], tris[:, 0] - tris[:, 2])
    return 0.5 * np.linalg.norm(cross, axis=-1)


def cluster_connected_components(mesh: Mesh):
    """Group faces into connected components and measure each one.

    Returns each face's cluster id, and per cluster its area (the sum of its
    face areas, in the mesh's units squared) and its face count, all three as
    arrays: on these meshes a Python list per cluster of the faces in it was
    most of what this used to spend. Faces are the nodes and shared edges the
    links, so two faces meeting at a single vertex belong to different
    components.

    Vectorized because these meshes are not small: the per-face BFS this
    replaces took 17 minutes and peaked at 29 GiB on a 16.7M-face
    reconstruction that this measures in 17 seconds.
    """
    logger = logging.getLogger(__name__)
    logger.info('Computing connected components...')
    n_faces = mesh.faces.shape[0]
    empty = np.zeros((0,), dtype=int)
    if n_faces == 0:
        return empty, empty.astype(float), empty
    tris = mesh.faces[..., 0].astype(np.int64)

    # One integer per face edge, low vertex first, so the faces sharing an
    # edge write the same key
    a = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    b = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    stride = int(tris.max()) + 1
    key = np.minimum(a, b) * stride + np.maximum(a, b)
    face_of_edge = np.tile(np.arange(n_faces), 3)

    # Sorting by key gathers the faces around each edge; linking each to the
    # one before it in its group connects the whole group
    order = np.argsort(key, kind='stable')
    key, face_of_edge = key[order], face_of_edge[order]
    shared = key[1:] == key[:-1]
    edge = (face_of_edge[:-1][shared], face_of_edge[1:][shared])
    graph = coo_matrix((np.ones(edge[0].shape, dtype=np.uint8), edge),
                       shape=(n_faces, n_faces))
    n_clusters, face_cluster = connected_components(graph, directed=False)

    logger.debug(f'Measuring {n_clusters} clusters...')
    sizes = np.bincount(face_cluster, minlength=n_clusters)
    areas = np.bincount(face_cluster, weights=face_areas(mesh),
                        minlength=n_clusters)
    return face_cluster, areas, sizes


class ComponentInventory(NamedTuple):
    """What a component filter kept and dropped: one area per component.

    Nothing else records how many components a delivered mesh has, so the
    filters report this and the apps print it.
    """
    kept: np.ndarray
    dropped: np.ndarray

    def __str__(self):
        line = (f'Kept {self.kept.size} component(s), '
                f'{self.kept.sum():.6g} units^2; '
                f'dropped {self.dropped.size}')
        if self.dropped.size:
            line += (f', {self.dropped.sum():.6g} units^2 '
                     f'(largest {self.dropped.max():.6g})')
        if not self.kept.size and self.dropped.size:
            # Every component gone is a threshold in the wrong units, not a
            # mesh worth delivering
            line += ' -- nothing left, is the threshold in the mesh\'s units?'
        return line


def filter_connected_components(mesh: Mesh, face_cluster, keep, areas,
                                filter_vertices=True) -> ComponentInventory:
    """Keep the clusters ``keep`` marks, and report the inventory either way.

    A filter that drops nothing skips the retriangulation, so asking for the
    inventory alone (a threshold of 0) costs little beyond the clustering. The
    vertex cleanup still runs: what the caller removed before filtering (the
    ground, say) leaves vertices behind whichever way this goes.
    """
    keep = np.asarray(keep, dtype=bool)
    if not areas.size:
        return ComponentInventory(areas, areas)

    inventory = ComponentInventory(areas[keep], areas[~keep])
    if inventory.dropped.size:
        keep_triangles_by_mask(mesh, keep[face_cluster])
    if filter_vertices:
        remove_unreferenced_vertices(mesh)
    return inventory


def keep_largest_connected_component(mesh: Mesh, filter_vertices=True):
    face_cluster, areas, _ = cluster_connected_components(mesh)
    keep = np.zeros(areas.shape, dtype=bool)
    if areas.size:
        keep[areas.argmax()] = True
    return filter_connected_components(mesh, face_cluster, keep, areas,
                                       filter_vertices)


def remove_connected_components_by_size(mesh: Mesh, num_faces: int,
                                        filter_vertices=True):
    face_cluster, areas, sizes = cluster_connected_components(mesh)
    return filter_connected_components(mesh, face_cluster, sizes >= num_faces,
                                       areas, filter_vertices)


def remove_connected_components_by_area(mesh: Mesh, min_area: float,
                                        filter_vertices=True):
    """Keep only the components whose surface area is at least ``min_area``.

    The measure is true surface area in the mesh's units squared (cm^2 on an
    autoscaled reconstruction), not a footprint. Face count -- what
    ``remove_connected_components_by_size`` thresholds -- is not a physical
    property: the same fragment carries wildly different counts depending on
    densification, refinement and image count, so a threshold tuned on one
    scan means something else on the next.
    """
    face_cluster, areas, _ = cluster_connected_components(mesh)
    return filter_connected_components(mesh, face_cluster, areas >= min_area,
                                       areas, filter_vertices)


def remove_connected_components_below_surface(mesh: Mesh,
                                             surface: GroundSurface,
                                             filter_vertices=True):
    """Keep only the components that reach above ``surface``.

    The scan bed's fiducial squares reconstruct as shallow recesses *below*
    the bed, which ground removal cannot take: they are not inside its band,
    they are a few tenths under it. Three measured captures each delivered one
    real component and 27-29 of these, and no area threshold gets them: 20-22
    per scan measured 0.51-2.84 cm^2, overlapping the real fragments the
    0.5 cm^2 floor exists to keep, and only the remainder was speckle an area
    filter would have caught anyway. Which side of the bed they are on
    separates them completely: every island topped out at -0.02 while the
    artifact reached +2.5 to +3.0, with not one of its vertices below +0.02.

    The test is a component's *highest* vertex, so geometry that hangs below
    the bed survives as long as some of it stands above, and an island's rim
    draping down into the cut is not what condemns it. The test needs no
    tolerance of its own: ground removal already took everything within the
    distance threshold of the surface, so a component that survived that and
    is still below sits a full threshold down.
    """
    face_cluster, areas, _ = cluster_connected_components(mesh)
    height = surface.signed_distance(mesh.vertices)
    # Per component, the highest vertex any of its faces reaches. Faces rather
    # than vertices because a vertex two components share belongs to both
    top = np.full(areas.shape, -np.inf)
    np.maximum.at(top, face_cluster,
                  height[mesh.faces[..., 0].astype(np.int64)].max(axis=-1))
    return filter_connected_components(mesh, face_cluster, top > 0., areas,
                                       filter_vertices)


def remove_degenerate_faces(mesh: Mesh):
    # get list of good faces
    mask = []
    start = mesh.faces.shape[0]
    for idx, f in enumerate(mesh.faces):
        if f[0] != f[1] != f[2] != f[0]:
            mask.append(idx)
    keep_triangles_by_mask(mesh, mask)
    end = mesh.faces.shape[0]
    print(f'Removed {start - end} degenerate faces.')
