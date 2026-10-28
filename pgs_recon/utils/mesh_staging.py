"""Staging a mesh where OpenMVS can read it.

Shared by the two things that texture an existing mesh through OpenMVS:
``pgs-retexture`` and ``pgs-localize-set``'s combined texture. Apps may not
import each other, so it lives here rather than in either.
"""
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def ensure_ply_mesh(mesh_path: Path, work_dir: Path,
                    out_stem: str = None) -> Path:
    """Stage a mesh into ``work_dir`` in a form OpenMVS can reliably load.

    OpenMVS' OBJ reader is strict (and mis-resolves a relative ``mtllib`` when
    a working folder is set), so OBJ inputs are converted to a geometry-only
    binary PLY (vertices + triangles). Texture coordinates/materials are
    irrelevant here because TextureMesh regenerates its own UVs. PLY inputs are
    copied through unchanged. Either way the result lives in ``work_dir`` so
    TextureMesh can reference it by basename. ``out_stem``, if given, names the
    staged file ``<out_stem>_input.ply`` so it cannot clash with the recon's
    own files in a shared working dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_name = f'{out_stem}_input.ply' if out_stem else None
    if mesh_path.suffix.lower() == '.ply':
        out = work_dir / (out_name or mesh_path.name)
        if mesh_path.resolve() != out.resolve():
            shutil.copy(mesh_path, out)
        return out

    verts = []
    faces = []
    with mesh_path.open() as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):  # triangulate fan
                    faces.append((idx[0], idx[i], idx[i + 1]))

    v = np.asarray(verts, dtype='<f4')
    fcs = np.asarray(faces, dtype='<i4')
    out = work_dir / (out_name or (mesh_path.stem + '.ply'))
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {len(v)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        f'element face {len(fcs)}\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    )
    face_rec = np.empty(len(fcs), dtype=[('n', 'u1'), ('i', '<i4', 3)])
    face_rec['n'] = 3
    face_rec['i'] = fcs
    with out.open('wb') as fh:
        fh.write(header.encode('ascii'))
        fh.write(v.tobytes())
        fh.write(face_rec.tobytes())
    logger.info(f'Converted {mesh_path.name} -> {out.name} '
                f'({len(v)} verts, {len(fcs)} tris)')
    return out
