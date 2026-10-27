"""Split the hand-held match graph into connected groups (ADR 0013 Decision 3).

Incremental SfM keeps one reconstruction and silently drops every view outside
it. Overlap in a hand-held set is variable, so the set may be several islands;
splitting the graph *before* SfM is what lets each island be reconstructed and
none be lost. A view with no surviving pair is a **singleton**: it is localized
on its own, through the reference and the overhead prior alone.

The graph is the set of pairs that survived ``openMVG_main_GeometricFilter``
(its ``-s``, :func:`pgs_recon.openmvg.geometric_filter`'s ``output_pairs``).
Edges are unweighted: the filter's own inlier threshold already decided which
pairs are real, and a second threshold here would be a second, disagreeing
answer to that question.

Stdlib-only.
"""
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, NamedTuple, Sequence, Tuple


class Groups(NamedTuple):
    #: Connected components of two or more views, largest first; ties broken by
    #: their smallest member, so the numbering is stable across runs.
    groups: Tuple[Tuple[Hashable, ...], ...]
    #: Views no surviving pair touches, in input order.
    singletons: Tuple[Hashable, ...]


def split(views: Sequence[Hashable],
          pairs: Iterable[Tuple[Hashable, Hashable]]) -> Groups:
    """Connected components of ``views`` under ``pairs``.

    A pair naming a view not in ``views`` is an error rather than a new view:
    it means the pairs file and the scene disagree about what the set is.
    """
    index = {v: i for i, v in enumerate(views)}
    if len(index) != len(views):
        raise ValueError('views must be unique')
    parent = list(range(len(views)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    touched = set()
    for a, b in pairs:
        for v in (a, b):
            if v not in index:
                raise ValueError(f'pair ({a}, {b}) names {v!r}, which is not '
                                 f'in the view set')
        if a == b:
            continue
        ia, ib = index[a], index[b]
        touched.update((ia, ib))
        ra, rb = find(ia), find(ib)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    components: Dict[int, List[int]] = {}
    for i in sorted(touched):
        components.setdefault(find(i), []).append(i)
    ordered = sorted(components.values(), key=lambda c: (-len(c), c[0]))
    return Groups(
        groups=tuple(tuple(views[i] for i in c) for c in ordered),
        singletons=tuple(v for i, v in enumerate(views) if i not in touched),
    )


def read_pairs(path: Path) -> List[Tuple[int, int]]:
    """Read an OpenMVG view-pairs file into ``(a, b)`` view-id pairs.

    OpenMVG's format pairs the first id on a line with every other id on it;
    :func:`pgs_recon.pgs_data.export_view_pairs` writes two per line, which is
    the common case of the same thing. Blank lines are skipped.
    """
    pairs = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            ids = [int(f) for f in fields]
        except ValueError:
            raise ValueError(f'{path}:{n}: not a list of view ids: {line!r}')
        if len(ids) < 2:
            raise ValueError(f'{path}:{n}: a pair needs two view ids')
        pairs.extend((ids[0], other) for other in ids[1:])
    return pairs
