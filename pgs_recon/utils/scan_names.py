"""Parsers for the PGS scan filename convention.

``{prefix}{camera}_{position}[_{capture}].{ext}`` is the only correspondence
between a scan image and a camera pose — there is no EXIF or ordering fallback —
so both the importer and the retexture app parse it. They differ only in what
they know: the importer reads a scan directory and knows its extension, while
retexture reads filenames stored in a solve that may have been imported from
converted copies.
"""
import re


def parse_scan_name(name: str, prefix: str, ext: str) -> tuple:
    """Parse ``(camera, position, capture)`` out of a PGS scan image name.

    A field the name does not supply comes back as ``None``, except the capture:
    the field is optional in the convention, and a name without one is capture 0.
    A name that does not parse at all comes back all ``None`` -- a missing field
    and an unrecognized name are different things, and only the first belongs to
    a capture.
    """
    match = re.fullmatch(
        rf'{re.escape(prefix)}(?P<camera>\d*)_(?P<position>\d*)'
        rf'(_(?P<capture>\d*))?\.{re.escape(ext)}', name)
    if match is None:
        return None, None, None
    cam = int(match.group('camera')) if match.group('camera') else None
    pos = int(match.group('position')) if match.group('position') else None
    cap = int(match.group('capture')) if match.group('capture') else 0
    return cam, pos, cap


def parse_view_name(name: str, prefix: str):
    """Parse an SfM view's stored filename into (camera, position) ints.

    Anchored on the scan's ``file_prefix`` but not on its extension: a solve may
    have been imported from converted copies of the scan images, so the suffix
    carries no information here. Anchoring on the prefix is what makes the
    optional capture field safe to omit — with an unanchored prefix,
    ``PGS_003_00047_02`` would also parse as camera 47 of position 2.

    Returns None if the name does not match.
    """
    m = re.fullmatch(rf'{re.escape(prefix)}(?P<cam>\d+)_(?P<pos>\d+)'
                     rf'(_(?P<cap>\d+))?\.\w+', name)
    if m is None:
        return None
    return int(m.group('cam')), int(m.group('pos'))
