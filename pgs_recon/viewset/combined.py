"""One mesh textured from every view: the scene OpenMVS textures it from.

Each view's deliverable is a one-view calibration (ADR 0013 Decision 7).
``TextureMesh`` needs them as one openMVG scene -- every view, its pose and its
intrinsic -- so :func:`merge_calibrations` joins them. Nothing else about the
combined texture needs code of its own: ``openMVG2openMVS`` undistorts the
images with each view's K, and ``TextureMesh`` picks one view per face and
levels the seams between them.

Unlike the per-view meshes, whose UVs point at the original image, the combined
texture is a resampled atlas (ADR 0002's objection). On CR04 its texels came
out finer than the photos' pixels, so what that costs is one resampling.

Stdlib-only.
"""
import json
from pathlib import Path
from typing import Dict, Sequence

#: cereal's flag on an id that is being declared rather than referred to.
_NEW = 0x80000000


def merge_calibrations(calibrations: Sequence[Path], images: Sequence[str],
                       root: Path) -> Dict:
    """One openMVG scene holding every calibration's view, pose and intrinsic.

    ``images[i]`` is the filename, in ``root``, that view ``i`` reads; it
    replaces whatever the calibration named. View, pose and intrinsic ids are
    the view's position in ``calibrations``. Every intrinsic must be the same
    camera model, because cereal registers the polymorphic type once, on its
    first use, and every later one refers back to it.
    """
    if len(calibrations) != len(images):
        raise ValueError(f'{len(calibrations)} calibrations and {len(images)} '
                         f'images: one each per view')
    if len(calibrations) < 2:
        raise ValueError('a combined texture needs at least two views')

    # In openMVG's own key order: cereal reads a JSON object in order.
    scene = {'sfm_data_version': None, 'root_path': str(root), 'views': [],
             'intrinsics': [], 'extrinsics': [], 'structure': [],
             'control_points': []}
    model = None
    n = len(calibrations)
    for key, (path, image) in enumerate(zip(calibrations, images)):
        data = json.loads(Path(path).read_text())
        if any(len(data.get(k, [])) != 1
               for k in ('views', 'intrinsics', 'extrinsics')):
            raise ValueError(f'{path} is not a one-view calibration')
        scene['sfm_data_version'] = (scene['sfm_data_version']
                                     or data.get('sfm_data_version'))

        view = data['views'][0]
        view['key'] = key
        # Every View is the base type, 0x40000000, never registered by name.
        # Pointer ids are declared in file order: the views, then intrinsics.
        view['value']['ptr_wrapper']['id'] = _NEW | (1 + key)
        vd = view['value']['ptr_wrapper']['data']
        vd.update(filename=image, local_path='', id_view=key, id_pose=key,
                  id_intrinsic=key)
        scene['views'].append(view)

        intrinsic = data['intrinsics'][0]
        intrinsic['key'] = key
        value = intrinsic['value']
        value['ptr_wrapper']['id'] = _NEW | (1 + n + key)
        if model is None:
            model = value.get('polymorphic_name')
        elif value.get('polymorphic_name', model) != model:
            raise ValueError(f'{path} is a {value.get("polymorphic_name")}, '
                             f'the first view a {model}: one camera model only')
        else:
            value['polymorphic_id'] &= ~_NEW
            value.pop('polymorphic_name', None)
        scene['intrinsics'].append(intrinsic)

        pose = data['extrinsics'][0]
        pose['key'] = key
        scene['extrinsics'].append(pose)
    return scene
