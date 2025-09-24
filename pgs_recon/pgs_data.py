import argparse
import itertools
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict

import exiftool
import numpy as np
import sfm_utils as sfm
from scipy.spatial.transform import Rotation as Rot
from sfm_utils.openmvg import __OPENMVG_CAMDB_DEFAULT_PATH

from pgs_recon.utility import current_timestamp


def get_tag_option(tags, opts):
    """Return the value of the first key in `opts` that is present in `tags`"""
    for o in opts:
        if o in tags.keys():
            return tags[o]
    raise KeyError(f'key option not found: {opts}')


def load_cam_calib(calib_path: Path) -> dict:
    with calib_path.open() as f:
        calib = json.loads(f.read())
    return calib


def neighbor_lookup_gridscan(scan_meta):
    """
    Create a neighbor lookup function for a PGS Grid Scan
    :param scan_meta: The 'scan' section of a PGS Scan metadata file
    :return:
    """
    logger = logging.getLogger(__name__)
    # Validate that it's a grid scan
    path = scan_meta['path']
    if not path.startswith('ROW_'):
        raise ValueError(f'Scan path is not a recognized grid scan: {path}')

    # Calculate the grid geometry
    dim_x, dim_y, dim_z = scan_meta['dims']
    step_x, step_y, step_z = scan_meta['stepsize']
    depth = math.ceil(dim_z / step_z) + 1
    cols = math.ceil(dim_x / step_x) + 1
    rows = math.ceil(dim_y / step_y) + 1

    # Validate against the number of captured positions
    num_positions = len(scan_meta['capture_positions'])
    num_expected = cols * rows * depth
    if num_expected != num_positions:
        logger.warning(f'Number of logged positions does not match expected: '
                       f'{num_positions} != {num_expected}')

    # Lookup table
    lut = np.zeros((depth, rows, cols), dtype=np.int64)
    pos = 0
    for d in range(depth):
        for r in range(rows):
            c_range = range(cols)
            if path == 'ROW_CONTINUOUS' and r & 1:
                c_range = reversed(c_range)
            for c in c_range:
                lut[d, r, c] = pos
                pos += 1

    # Neighbor lookup function
    def get_neighbors(pos_idx, radius):
        # Only search in XY
        _, y, x = [a.item() for a in np.where(lut == pos_idx)]
        ly, hy = max(0, y - radius), min(rows, y + radius + 1)
        lx, hx = max(0, x - radius), min(cols, x + radius + 1)
        n = lut[:, ly:hy, lx:hx].flatten()
        return n

    return get_neighbors


def import_pgs_scan(scan_dir: Path, cam_db: dict,
                    cam_calib: dict = None,
                    pairs_file_radius: int = 2) -> tuple[sfm.Scene, list]:
    logger = logging.getLogger(__name__)
    # Load scan metadata
    meta_path = scan_dir / 'metadata.json'
    with meta_path.open() as f:
        scan_meta = json.loads(f.read())

    # Insert calibration data into camera metadata
    if cam_calib is not None:
        for cam in scan_meta['scanner']['cameras']:
            if 's/n' in cam.keys():
                serial_no = cam['s/n']
                if serial_no in cam_calib['calibs'].keys():
                    cam['k3'] = cam_calib['calibs'][serial_no]['k3']
                    # cam['t2'] = cam_calib['calibs'][serial_no]['t2']

    # Get list of images
    prefix = scan_meta['scan']['file_prefix']
    ext = scan_meta['scan']['format'].lower()
    images = list(scan_dir.glob(f'{prefix}*.{ext}'))
    images.sort()

    # Get image metadata
    files = [str(i) for i in images]
    if len(files) == 0:
        logger.error(
            'Provided scan metadata specifies file pattern, but no files match.')
        raise RuntimeError()

    with exiftool.ExifToolHelper() as et:
        img_metadata = et.get_metadata(files)

    # Setup sfm
    scene = sfm.Scene()
    scene.root_dir = scan_dir

    # view IDs per pose
    position_views = dict()

    # Fill out sfm with data
    intrinsics = {}
    for img in images:
        # Lookup this images tags
        tags = next((i for i in img_metadata if i['File:FileName'] == img.name),
                    None)
        if tags is None:
            logger.error(f'No tags loaded for image: {str(img)}')
            continue

        # Setup view
        view = sfm.View()
        view.path = img
        view.width = get_tag_option(tags, ['File:ImageWidth', 'EXIF:ImageWidth'])
        view.height = get_tag_option(tags, ['File:ImageHeight', 'EXIF:ImageHeight'])
        view.make = tags['EXIF:Make']
        view.model = tags['EXIF:Model']

        # Get the camera idx and the position idx
        cam_idx = None
        pos_idx = None
        cap_idx = 0
        match = re.fullmatch(
            rf'{re.escape(prefix)}(?P<camera>\d*)_(?P<position>\d*)(_(?P<capture>\d*))?\.{ext}',
            img.name)
        if match:
            cam_idx = int(match.group('camera'))
            pos_idx = int(match.group('position'))
            if match.group('capture'):
                cap_idx = int(match.group('capture'))

        # Skip anything but the primary capture
        # TODO: Handle other captures
        if cap_idx != 0:
            logger.warning(f'Skipping {img.name} from capture group {cap_idx}')
            continue

        # Setup intrinsic
        intrinsic = sfm.IntrinsicRadialK3()
        if cam_idx is not None:
            cam = scan_meta['scanner']['cameras'][cam_idx]
            if 'k3' in cam.keys() and 't2' in cam.keys():
                intrinsic = sfm.IntrinsicBrownT2()
                intrinsic.dist_params = cam['k3'] + cam['t2']
            elif 'k3' in cam.keys():
                intrinsic.dist_params = cam['k3']
        intrinsic.width = view.width
        intrinsic.height = view.height
        intrinsic.focal_length = tags['EXIF:FocalLength']
        if f'{view.make} {view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.make} {view.model}']
        elif f'{view.model}' in cam_db.keys():
            intrinsic.sensor_width = cam_db[f'{view.model}']
        else:
            logger.warning(
                f'Camera not in database: {view.make} {view.model}. Ignoring file: {img.name}')
            continue

        # Init extrinsics
        pose = sfm.Pose()
        if cam_idx is not None:
            # Get Camera
            cam = scan_meta['scanner']['cameras'][cam_idx]

            # Assign position
            if cam['is_absolute_pos'] is True or pos_idx is None:
                if pos_idx is None:
                    logger.warning(
                        f'Couldn\'t parse position index. Interpreting file\'s pose as absolute: {img.name}')
                pose.center = cam['position']
            else:
                center = np.array(
                    scan_meta['scan']['capture_positions'][pos_idx])
                offset = cam['position']
                position = np.add(center, offset)
                pose.center = position.round(15).tolist()

            # Calculate the rotation matrix
            # Our rotation matrix is right-handed and row-major
            # We compose rotations as ZYX, so reverse the angle list
            euler_angles = cam['rotation'][::-1]
            rotation = Rot.from_euler('zyx', euler_angles, degrees=True)
            pose.rotation = rotation.as_matrix().round(15)

        # Only add everything to the SfM at the end
        view = scene.add_view(view)
        # One intrinsic per camera, not per body and lens combo
        if cam_idx in intrinsics.keys():
            view.intrinsic = intrinsics[cam_idx]
        else:
            view.intrinsic = scene.add_intrinsic(intrinsic, group_models=False)
            intrinsics[cam_idx] = view.intrinsic
        view.pose = scene.add_pose(pose)

        # Add this view to the list of views for this position
        ps = position_views.get(pos_idx, list())
        ps.append(view.id)
        position_views[pos_idx] = ps

    # Get a neighbor lookup function
    neighbor_lookup = neighbor_lookup_gridscan(scan_meta['scan'])

    # Compile the list of view pairs
    # For each position for which we have at least one view...
    view_pairs = set()
    for pos_idx, view_list in position_views.items():
        # Get the neighbor positions
        neighbors = neighbor_lookup(pos_idx, pairs_file_radius)

        # Calculate the view pairs this position to all neighbor positions
        for n in neighbors:
            neighbor_list = position_views.get(n, [])
            pairs = []
            for a, b in itertools.product(view_list, neighbor_list):
                # never compare a view with itself
                if a == b:
                    continue
                # smallest view ID first for set union to work
                pairs.append((b, a) if b < a else (a, b))
            view_pairs = view_pairs.union(pairs)

    # Sort the view pairs
    view_pairs = list(view_pairs)
    view_pairs.sort()

    # Return the filled out sfm
    return scene, view_pairs


def export_view_pairs(path: Path, view_pairs: list):
    with path.open('w') as f:
        for a, b in view_pairs:
            f.write(f'{a} {b}\n')


def init_sfm_pgs(paths: Dict[str, Path], pairs_file_radius: int = 2,
                 metadata: Dict = None):
    """Init an SfM from a PGS Scan"""
    logger = logging.getLogger(__name__)
    # Load the camera db
    cam_db = sfm.openmvg_load_camdb(paths['CAM_DB'])

    # Load the calib if provided
    calib = None
    if 'input_calib' in paths.keys():
        logger.info('Loading camera calibrations')
        if metadata is not None:
            metadata['commands'][
                current_timestamp()] = f'load_cam_calib {str(paths["input_calib"])}'
        calib = load_cam_calib(paths['input_calib'])

    # Load the pgs file
    scene, view_pairs = import_pgs_scan(paths['input'].resolve(), cam_db=cam_db,
                                        cam_calib=calib,
                                        pairs_file_radius=pairs_file_radius)
    if metadata is not None:
        cmd = (f'import_pgs_scan(scan_dir={str(paths["input"])}, '
               f'cam_db={str(paths["CAM_DB"])}, '
               f'cam_calib={str(paths.get("input_calib", None))}, '
               f'pairs_file_radius={pairs_file_radius})')
        if calib:
            cmd += f' {str(paths["input_calib"])}'
        metadata['commands'][current_timestamp()] = cmd

    # Write the SFM
    sfm.export_scene(path=paths['sfm'], scene=scene)

    # Write the view pairs
    if view_pairs is not None:
        paths['view_pairs'] = paths['matches_dir'] / 'pgs_view_pairs.txt'
        export_view_pairs(paths['view_pairs'], view_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pgs-dir', '-p', required=True,
                        help='PGS Scan directory')
    parser.add_argument('--cam-db', '-d', default=__OPENMVG_CAMDB_DEFAULT_PATH,
                        help='Camera database path')
    parser.add_argument('--cam-calib', '-c', help="Camera calibrations file")
    parser.add_argument('--output-sfm', '-o', type=Path,
                        default='sfm_data.json', help='Output SFM file')
    parser.add_argument('--view-pairs', '-v', type=Path,
                        default='pgs_view_pairs.txt', help='Output view pairs file')
    args = parser.parse_args()

    # Logger
    logger = logging.getLogger("pgs-import")

    # Load the camera db
    cam_db_path = Path(args.cam_db)
    cam_db = sfm.openmvg_load_camdb(cam_db_path)

    # Load the camera calibrations (if present)
    calib = None
    if args.cam_calib:
        logger.info('Loading camera calibrations')
        calib_path = Path(args.cam_calib)
        calib = load_cam_calib(calib_path)

    # Load the pgs file
    logger.info('Loading PGS Scan')
    pgs_dir_path = Path(args.pgs_dir)
    scene, view_pairs = import_pgs_scan(pgs_dir_path, cam_db, calib)

    # Write the SFM
    logger.info('Exporting SfM scene')
    sfm.export_scene(path=args.output_sfm, scene=scene)

    # Write the view pairs file
    if view_pairs is not None:
        logger.info('Exporting view pairs')
        export_view_pairs(args.view_pairs, view_pairs)

    logger.info('Done.')


if __name__ == '__main__':
    main()
