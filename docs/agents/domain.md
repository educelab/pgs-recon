# Domain Docs

How the engineering skills should consume this repo's domain documentation when
exploring the codebase. **Layout: single-context.**

## Before exploring, read these

- **`CONTEXT.md`** at the repo root: the glossary of project-specific terms whose
  ambiguity causes mistakes (capture, capture position, camera, stage, role, …).
  It is deliberately *not* a description of the code.
- **`docs/adr/`**: read the ADRs that touch the area you're about to work in.
  There is no `CONTEXT-MAP.md` and there are no context-scoped ADR directories —
  every decision lives in the one `docs/adr/`.
- **`CLAUDE.md`** at the repo root carries the build, run and architecture
  orientation (pipeline flow, module layout, conventions).

If any of these files don't exist, **proceed silently**. Don't flag their absence;
don't suggest creating them upfront. The `/domain-modeling` skill (reached via
`/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when
terms or decisions actually get resolved.

## File structure

```
/
├── CLAUDE.md
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-retexture-with-alternate-modality.md
│   ├── …
│   └── 0007-name-the-manifest-for-the-tool.md
├── pgs_recon/          ← the Python orchestration layer
├── dependencies/       ← CMake superbuild for the C++ toolchain
└── tests/
```

This is a single-context repo: one `CONTEXT.md`, one `docs/adr/`. If it ever
splits into genuinely separate contexts, a root `CONTEXT-MAP.md` pointing at
per-context `CONTEXT.md` files is the convention to adopt, and skills should read
each context relevant to the topic.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a
hypothesis, a test name), use the term as defined in `CONTEXT.md` — including its
_Avoid_ list, which exists because those synonyms have caused real confusion here.

If the concept you need isn't in the glossary yet, that's a signal: either you're
inventing language the project doesn't use (reconsider) or there's a real gap
(note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than
silently overriding:

> _Contradicts ADR-0006 (stage-named artifacts), but worth reopening because…_
