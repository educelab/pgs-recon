from pathlib import Path
from typing import Dict

from pgs_recon.utility import current_timestamp, run_command


def mvs_densify(paths: Dict[str, Path], mvs_key: str, resolution_lvl: int = None, metadata: Dict = None) -> str:
    """Densify a point cloud"""
    out_key = mvs_key + '_dense'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_dense.mvs')
    command = [
        str(paths['MVS_BIN'] / 'DensifyPointCloud'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs']),
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvs_reconstruct(paths: Dict[str, Path], mvs_key: str, free_space=False, metadata: Dict = None) -> str:
    """Reconstruct an MVS scene"""
    out_key = mvs_key + '_mesh'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_mesh.mvs')
    command = [
        str(paths['MVS_BIN'] / 'ReconstructMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs'])
    ]
    if free_space:
        command.extend(['--free-space-support', '1'])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvs_refine(paths: Dict[str, Path], mvs_key: str, decimation_factor: float = None, resolution_lvl: int = None,
               metadata: Dict = None) -> str:
    """Refine a reconstructed mesh"""
    out_key = mvs_key + '_refine'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + '_refine.mvs')
    command = [
        str(paths['MVS_BIN'] / 'RefineMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '-w', str(paths['mvs'])
    ]
    if decimation_factor is not None:
        command.extend(['--decimate', str(decimation_factor)])
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key


def mvs_texture(paths: Dict[str, Path], mvs_key: str, file_format: str = 'ply', resolution_lvl: int = None,
                metadata: Dict = None) -> str:
    """Texture a mesh"""
    out_key = mvs_key + '_texture'
    in_path = paths[mvs_key]
    paths[out_key] = in_path.parent / (in_path.stem + f'_texture.{file_format.lower()}')
    command = [
        str(paths['MVS_BIN'] / 'TextureMesh'),
        '-i', str(paths[mvs_key].name),
        '-o', str(paths[out_key].name),
        '--export-type', file_format.lower(),
        '-w', str(paths['mvs'])
    ]
    if resolution_lvl is not None:
        command.extend(['--resolution-level', str(resolution_lvl)])
    if metadata is not None:
        metadata['commands'][current_timestamp()] = (str(' ').join(command))
    run_command(command)
    return out_key
