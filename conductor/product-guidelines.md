# Product Guidelines

## Voice and Tone

Concise and direct. Short, need-to-know statements; minimal prose. Applies to
docs, `--help` text, log lines and error messages.

- Error messages say what failed and what to check (e.g. a prefix failure names
  the path it looked for and which tier chose it).
- In-code comments: minimal, 2-3 line blocks at most. Docstrings descriptive, not
  exhaustive, written for humans.

## Design Principles

1. **Thin, honest wrappers.** Python orchestrates; it does not reimplement.
   Wrappers are pure argv builders ending in `toolchain.run`. `None` omits a flag
   so the binary's default wins. Every upstream flag is reachable (ADR 0005).
2. **Measure, don't guess.** Defaults are derived constants backed by
   measurements and recorded in ADRs (e.g. `COARSEN_RATIO`, `DECIMATE_RATIO`).
   Behaviour changes are deliberate and documented.
3. **Fail loudly.** Refuse ambiguous state rather than guess
   (`ArtifactsNotColocated`, `ToolFailed` with exit status, prefix resolution
   errors that name their source).

## Decision Records

Non-obvious decisions get an ADR in `docs/adr/`. Domain terms go in `CONTEXT.md`.
