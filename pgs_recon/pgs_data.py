import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict

import exiftool
import numpy as np
import sfm_utils as sfm
from scipy.spatial.transform import Rotation as Rot
from sfm_utils.openmvg import __OPENMVG_CAMDB_DEFAULT_PATH

from pgs_recon.utility import current_timestamp


def load_cam_calib(calib_path: Path) -> dict:
    with calib_path.open() as f:
        calib = json.loads(f.read())
    return calib


def import_pgs_scan(scan_dir: Path, cam_db: dict, cam_calib: dict = None) -> sfm.Scene:
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
        logger.error('Provided scan metadata specifies file pattern, but no files match.')
        raise RuntimeError()

    with exiftool.ExifToolHelper() as et:
        img_metadata = et.get_metadata(files)

    # Setup sfm
    scene = sfm.Scene()
    scene.root_dir = scan_dir

    # Fill out sfm with data
    for img in images:
        # Lookup this images tags
        tags = next((i for i in img_metadata if i['File:FileName'] == img.name), None)
        if tags is None:
            logger.error(f'No tags loaded for image: {str(img)}')
            continue

        # Setup view
        view = sfm.View()
        view.path = img
        view.width = tags['File:ImageWidth']
        view.height = tags['File:ImageHeight']
        view.make = tags['EXIF:Make']
        view.model = tags['EXIF:Model']

        # Get the camera idx and the position idx
        cam_idx = None
        pos_idx = None
        if re.fullmatch(rf'{re.escape(prefix)}\d*_\d*\.{re.escape(ext)}', img.name):
            cam_idx, pos_idx = img.name.replace(prefix, '').replace(f'.{ext}', '').split('_')
            cam_idx = int(cam_idx)
            pos_idx = int(pos_idx)

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
            logger.warning(f'Camera not in database: {view.make} {view.model}. Ignoring file: {img.name}')
            continue

        # Init extrinsics
        pose = sfm.Pose()
        if cam_idx is not None:
            # Get Camera
            cam = scan_meta['scanner']['cameras'][cam_idx]

            # Assign position
            if cam['is_absolute_pos'] is True or pos_idx is None:
                if pos_idx is None:
                    logger.warning(f'Couldn\'t parse position index. Interpreting file\'s pose as absolute: {img.name}')
                pose.center = cam['position']
            else:
                center = np.array(scan_meta['scan']['capture_positions'][pos_idx])
                offset = cam['position']
                position = np.add(center, offset)
                pose.center = position.round(15).tolist()

            # Calculate the rotation matrix
            # Our rotation matrix is right handed and row-major
            # We compose rotations as ZYX, so reverse the angle list
            euler_angles = cam['rotation'][::-1]
            rotation = Rot.from_euler('zyx', euler_angles, degrees=True)
            pose.rotation = rotation.as_matrix().round(15)

        # Only add everything to the SfM at the end
        scene.add_view(view)
        view.intrinsic = scene.add_intrinsic(intrinsic)
        view.pose = scene.add_pose(pose)

    # Return the filled out sfm
    return scene


def init_sfm_pgs(paths: Dict[str, Path], metadata: Dict = None):
    """Init an SfM from a PGS Scan"""
    logger = logging.getLogger(__name__)
    # Load the camera db
    cam_db = sfm.openmvg_load_camdb(paths['CAM_DB'])

    # Load the calib if provided
    calib = None
    if 'input_calib' in paths.keys():
        logger.info('Loading camera calibrations')
        if metadata is not None:
            metadata['commands'][current_timestamp()] = 'load_cam_calib ' + str(paths['input_calib'])
        calib = load_cam_calib(paths['input_calib'])

    # Load the pgs file
    scene = import_pgs_scan(paths['input'].resolve(), cam_db=cam_db, cam_calib=calib)
    if metadata is not None:
        cmd = ' '.join(['import_pgs_scan', str(paths["input"]), str(paths['CAM_DB'])])
        if calib:
            cmd += ' ' + str(paths['input_calib'])
        metadata['commands'][current_timestamp()] = cmd

    # Write the SFM
    sfm.export_scene(path=paths['sfm'], scene=scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pgs-dir', '-p', required=True, help='PGS Scan directory')
    parser.add_argument('--cam-db', '-d', default=__OPENMVG_CAMDB_DEFAULT_PATH, help='Camera database path')
    parser.add_argument('--cam-calib', '-c', help="Camera calibrations file")
    parser.add_argument('--output-sfm', '-o', default='sfm_data.json', help='Output SFM file')
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
    scene = import_pgs_scan(pgs_dir_path, cam_db, calib)

    # Write the SFM
    logger.info('Exporting SfM scene')
    sfm.export_scene(path=args.output_sfm, scene=scene)

    logger.info('Done.')


if __name__ == '__main__':
    main()
