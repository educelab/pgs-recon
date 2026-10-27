"""Provenance and per-view resume for a view-set run (ADR 0013 Decisions 7, 8).

Deliberately **not** :class:`pgs_recon.stages.StageTracker`. That planner is
bound to ``pgs-recon``'s ``STAGES`` and ``pipeline_shape``; generalizing it would
change the core this component must not touch. What a view set needs is
narrower: a view's results are current when the inputs they were made from are
the inputs this run would use, and stale otherwise.

"Inputs" are fingerprinted, not just named. Re-registering the overhead view or
re-reconstructing the mesh in place keeps every path and invalidates every
calibration made through them, and only a fingerprint sees that.

The decision functions (:func:`stale_reason`, :func:`provenance_drift`) work on
plain dicts, like the stage planner, so they are testable without a disk.
Stdlib-only.
"""
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

#: Bumped when the record's shape changes incompatibly.
FORMAT = 1


def fingerprint(path: Path) -> Dict:
    """What identifies one input file's *content*, cheaply: size and mtime.

    Not a hash. The inputs are meshes and textures in the gigabytes, read on
    every resume; size+mtime is what ``make`` trusts, and the failure it admits
    (an edit preserving both) is not one a pipeline produces.
    """
    st = Path(path).stat()
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def stale_reason(record: Optional[Mapping],
                 inputs: Mapping[str, Mapping]) -> Optional[str]:
    """Why a view's recorded results cannot be reused, or ``None`` if they can.

    ``record`` is the view's entry in the manifest (``None`` if it has none);
    ``inputs`` maps each input's role to its current :func:`fingerprint`.
    """
    if record is None:
        return 'never run'
    if record.get('status') != 'done':
        return f'last run ended {record.get("status", "without a status")}'
    recorded = record.get('inputs', {})
    for role in sorted(set(inputs) | set(recorded)):
        if role not in recorded:
            return f'{role} is a new input'
        if role not in inputs:
            return f'{role} is no longer an input'
        if dict(recorded[role]) != dict(inputs[role]):
            return f'{role} changed ({recorded[role]["path"]})'
    return None


def provenance_drift(recorded: Mapping, current: Mapping) -> Dict[str, str]:
    """Run-level inputs whose fingerprint changed since the manifest was written.

    Separate from :func:`stale_reason` because the answer is different: a
    changed reference or overhead calibration invalidates *every* view, and the
    run says so once rather than once per view.
    """
    drift = {}
    for role in sorted(set(recorded) | set(current)):
        before, after = recorded.get(role), current.get(role)
        if before != after:
            drift[role] = ('added' if before is None else
                           'removed' if after is None else 'changed')
    return drift


def load(path: Path) -> Dict:
    """The manifest at ``path``, or a fresh one if there is none yet."""
    if not Path(path).exists():
        return {'format': FORMAT, 'provenance': {}, 'views': {}}
    meta = json.loads(Path(path).read_text())
    if meta.get('format') != FORMAT:
        raise ValueError(f'{path} is format {meta.get("format")}; this version '
                         f'reads {FORMAT}')
    return meta


def write(path: Path, meta: Mapping) -> None:
    """Write atomically: a crash mid-write leaves the previous manifest whole."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)
