"""Localize a hand-held view set against the rig's IR940 mesh (ADR 0013).

An optional component: nothing in ``pgs-recon``'s pipeline imports it, and it
only reads a finished reconstruction. The dependency runs one way -- this
package imports from the core (``toolchain``, ``openmvg``, ``utils``), the core
never imports this package -- which is what lets it be left out of an install,
a build or an image without anything else noticing.

Module map:

- :mod:`.layout` -- what every artifact of a view-set run is called.
- :mod:`.inputs` -- what a plate supplies, and which path (ADR 0013 Decision 2)
  that makes available.
- :mod:`.groups` -- the hand-held match graph split into connected groups and
  singletons (Decision 3).
- :mod:`.manifest` -- provenance and per-view resume (Decision 7, 8).
- :mod:`.tools` -- argv builders for ``pgs-localize-joint``, the C++ half.
- :mod:`.pipeline` -- the phases, in order.
- :mod:`.cli` -- ``pgs-localize-set``.

``layout``, ``inputs``, ``groups`` and ``manifest`` are stdlib-only, like the
planner core, so their tests run on a bare Python.
"""
