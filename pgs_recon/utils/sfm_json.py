"""Edits to OpenMVG ``SfM_Data`` JSON that both retexturing and calibration make.

OpenMVG serializes scenes with cereal, and its JSON has two traps that any tool
rewriting a scene by hand has to handle: polymorphic type registration lives on
whichever instance happens to be *first* (`fix_polymorphic_registration`), and
extrinsics are stored as (rotation, center) rather than the (R, t) most
conventions expect (`transform_extrinsic`). Both bite when a scene is filtered
down to a subset of views, which is exactly what ``pgs-calibrate`` does to
extract one localized view and ``pgs-retexture`` does to keep one rig camera.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_POLY_FLAG = 0x80000000


def camera_from_calibration(calibration_json: Path):
    """Extract (R, C, f, cx, cy, W, H, disto) from a one-view calibration."""
    cal = json.loads(calibration_json.read_text())
    did = cal['intrinsics'][0]['value']['ptr_wrapper']['data']
    f = did['focal_length']
    cx, cy = did['principal_point']
    W, H = did['width'], did['height']
    disto = None
    if 'disto_k3' in did:               # [k1, k2, k3]
        disto = list(did['disto_k3'])
    elif 'disto_k1' in did:             # [k1]
        disto = list(did['disto_k1']) + [0.0, 0.0]
    e = cal['extrinsics'][0]['value']
    R = np.asarray(e['rotation'], dtype=np.float64)   # X_cam = R (X - C)
    C = np.asarray(e['center'], dtype=np.float64)
    return R, C, f, cx, cy, W, H, disto


def fix_polymorphic_registration(all_items, kept_items):
    """Repair cereal polymorphic-pointer registration after filtering.

    In openMVG's cereal JSON the first instance of each polymorphic type sets
    the high bit on ``polymorphic_id`` and carries a ``polymorphic_name``;
    later instances reference the type by its bare numeric id. If filtering
    drops the registering instance, surviving instances reference an
    unregistered type id and the scene fails to load. This promotes the first
    kept instance of each type back to the registration form.
    """
    # Map type-number -> registered name from the full (original) list.
    registry = {}
    for it in all_items:
        pid = it['value'].get('polymorphic_id', 0)
        if pid & _POLY_FLAG:
            registry[pid & ~_POLY_FLAG] = it['value'].get('polymorphic_name')

    seen = set()
    for it in kept_items:
        val = it['value']
        pid = val.get('polymorphic_id', 0)
        typenum = (pid & ~_POLY_FLAG) if (pid & _POLY_FLAG) else pid
        if typenum not in seen:
            val['polymorphic_id'] = _POLY_FLAG | typenum
            if registry.get(typenum) is not None:
                val['polymorphic_name'] = registry[typenum]
            seen.add(typenum)
        else:
            val.pop('polymorphic_name', None)
            val['polymorphic_id'] = typenum
    return kept_items


def bare_polymorphic_id(value: dict) -> int:
    """The cereal type id in ``value``, with the registration bit cleared.

    Use when appending an instance of a type some *other* surviving instance
    already registers: it must reference the type by bare numeric id and carry no
    ``polymorphic_name``, or cereal sees the type registered twice and the scene
    fails to load. The mirror image of `fix_polymorphic_registration`, which
    promotes an instance *to* the registering form.
    """
    return value.get('polymorphic_id', 0) & ~_POLY_FLAG


def transform_extrinsic(R, C, sfm_transform):
    """Transform an OpenMVG extrinsic (R, C) from the original SfM frame to
    the frame defined by sfm_transform (4x4 similarity matrix from pgs-center
    --save-transform). The scale is stripped from the upper-left 3x3 block so
    R_new remains a proper rotation matrix. C_new uses the full (scaled) block
    since centers are points, not directions."""
    s = np.linalg.norm(sfm_transform[:3, 0])
    R_T = sfm_transform[:3, :3] / s
    R_new = R @ R_T.T
    C_new = sfm_transform[:3, :3] @ C + sfm_transform[:3, 3]
    return R_new, C_new


def transform_sfm_extrinsics(sfm_json: Path, sfm_transform) -> None:
    """Rewrite all extrinsics in an SfM JSON file to the frame defined by
    sfm_transform. Edits the file in-place."""
    data = json.loads(sfm_json.read_text())
    for e in data.get('extrinsics', []):
        val = e['value']
        R = np.asarray(val['rotation'], dtype=np.float64)
        C = np.asarray(val['center'], dtype=np.float64)
        R_new, C_new = transform_extrinsic(R, C, sfm_transform)
        val['rotation'] = R_new.tolist()
        val['center'] = C_new.tolist()
    sfm_json.write_text(json.dumps(data, indent=2))
    logger.info(f'Transformed {len(data.get("extrinsics", []))} '
                f'extrinsics to centered frame')
