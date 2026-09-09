"""Reading a PLY file's header, and nothing else.

One function, for one job: the ``coarsen`` stage's face target is a fraction
of its input mesh's face count, so the pipeline has to know that count before
it can build the argv. A PLY declares it in the header -- ``element face N``,
inside the first few hundred bytes -- so this reads the header and stops,
rather than pulling twenty-six million faces through Python to count them.

Deliberately not a mesh reader. Nothing here loads geometry; anything that
needs the mesh itself hands the path to a binary.
"""
from pathlib import Path
from typing import Union

#: How far into a file to look for ``end_header`` before giving up. A PLY
#: header is a few hundred bytes even with a long comment block, so anything
#: past this is a file that is not going to declare an element count -- and the
#: cap is what keeps a mis-bound artifact from being read into memory in full.
MAX_HEADER_BYTES = 64 * 1024


class NotAPlyHeader(ValueError):
    """A file did not begin with a PLY header this could read."""


def face_count(path: Union[str, Path]) -> int:
    """The number of faces a PLY header declares.

    Works for ascii and both binary byte orders, because the count is in
    the header either way. Element names other than ``face`` are skipped, so
    the vertex block's own count is never mistaken for this one.

    Raises :class:`NotAPlyHeader` when the file is not a PLY, when the header
    does not end inside :data:`MAX_HEADER_BYTES`, or when it declares no
    ``face`` element -- all of which are the same thing to a caller: this file
    cannot answer the question.
    """
    path = Path(path)
    faces = None
    with path.open('rb') as f:
        first = f.readline(80).strip()
        if first not in (b'ply', b'PLY'):
            raise NotAPlyHeader(f'{path} does not begin with a PLY magic line; '
                                f'got {first[:20]!r}.')
        read = len(first)
        while True:
            line = f.readline(1024)
            if not line:
                raise NotAPlyHeader(f'{path} ended before its PLY header did.')
            read += len(line)
            tokens = line.split()
            if tokens[:1] == [b'end_header']:
                break
            if read > MAX_HEADER_BYTES:
                raise NotAPlyHeader(
                    f'{path} has no end_header in its first '
                    f'{MAX_HEADER_BYTES} bytes, so it is not a PLY this can '
                    f'read.')
            # `element <name> <count>`; the first `face` block wins, as
            # a PLY declaring two would be malformed anyway.
            if (len(tokens) >= 3 and tokens[0] == b'element'
                    and tokens[1] == b'face' and faces is None):
                try:
                    faces = int(tokens[2])
                except ValueError:
                    raise NotAPlyHeader(
                        f'{path} declares a face element with an unreadable '
                        f'count: {tokens[2]!r}.') from None
    if faces is None:
        raise NotAPlyHeader(f'{path} declares no face element, so it carries no '
                            f'surface to count.')
    return faces
