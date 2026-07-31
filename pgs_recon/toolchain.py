"""Finding the toolchain's binaries, and running them.

Two jobs, both of which every wrapper needs and none of which is about
photogrammetry: turn a tool *name* into a path to an executable, and run an argv
while recording it.

**Configuration is process-wide, not threaded through call signatures.** An
app calls :func:`configure` once in ``main()``; :func:`using` scopes an override.
A wrapper takes no ``prefix`` and no ``metadata`` argument, which is what lets it
be a straight translation of the binary's own interface (`ADR 0005
<../docs/adr/0005-wrappers-mirror-the-binary.md>`_) -- and means no wrapper can
forget to record what it ran, because :func:`run` is the only way to run
anything.

**Resolution happens at call time**, tier by tier: an explicit ``prefix``
argument, then :func:`configure`, then ``$PGS_RECON_PREFIX``, then
:data:`DEFAULT_PREFIX`. Late binding is load-bearing twice over: a default bound
at definition time (``def f(bin_dir=_discover())``) would freeze before
``main()`` could configure anything, and it would put filesystem discovery on the
import path -- where it must not be, since CI runs this test suite *before* it
builds ``dependencies/``, so ``import pgs_recon.toolchain`` has to work on a
machine with no OpenMVG at all.

A caller-supplied prefix is made absolute as it is accepted (:func:`_absolute`),
because two stages run with ``cwd`` set and a relative argv[0] means something
different to them than it did to the check that approved it.

Because resolution checks the local filesystem, it is inherently local: where a
binary lives is a property of the execution environment, not of the submitting
process. Running stages inside a container or through a batch scheduler would
mean putting the tool *name* in the argv and resolving inside the runner. ADR
0005 records why that seam is not built yet.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import MutableMapping, Optional, Sequence, Tuple, Union

from pgs_recon.utility import ToolFailed, current_timestamp, run_command

#: Prefix assumed when nothing else says otherwise: the container/install
#: layout, not the CMake superbuild's ``dependencies/installed``.
DEFAULT_PREFIX = Path('/usr/local')

#: Environment variable naming the install prefix, for callers that cannot pass
#: one (a wrapper script, a batch job template).
PREFIX_ENV = 'PGS_RECON_PREFIX'

#: Prefix-relative directories the tool families live in. OpenMVG's binaries and
#: our own ``pgs-*`` utilities share ``bin/``; OpenMVS keeps its own.
MVG_BIN = 'bin'
MVS_BIN = 'bin/OpenMVS'

#: Prefix-relative path of the OpenMVG camera sensor database.
CAM_DB = 'lib/openMVG/sensor_width_camera_database.txt'

PathLike = Union[str, os.PathLike]


class _Unset:
    """The type of the "argument omitted" marker.

    A distinct type, not just a sentinel object, so :func:`configure`'s
    signature can say what its arguments mean: ``None`` clears that tier,
    omitting it leaves the tier alone.
    """


_UNSET = _Unset()

#: What :func:`configure` and :func:`using` accept for each tier: a value,
#: ``None`` to clear it, or nothing at all to leave it as it is.
PrefixSetting = Union[PathLike, None, _Unset]
RecorderSetting = Union['Recorder', None, _Unset]

_prefix: Optional[Path] = None
_recorder: Optional['Recorder'] = None


class ToolNotFound(ToolFailed):
    """A tool could not be located under the effective prefix.

    A ``ToolFailed`` on purpose: every ``main()`` already handles those, and
    ``exit_code`` is 127 -- what a shell reports for a command it could not
    find. The message names the tier the prefix came from, because "not found at
    /usr/local/bin/X" is only actionable once you know whether /usr/local was
    asked for or merely assumed.
    """

    def __init__(self, name: str, expected: Path, prefix: Path, tier: str,
                 problem: str = 'not found'):
        #: Tool name as the caller asked for it.
        self.name = name
        #: Where it was expected to be.
        self.expected = Path(expected)
        #: The effective prefix, and where that came from.
        self.prefix = Path(prefix)
        self.tier = tier
        super().__init__([str(expected)], None,
                         detail=f'{problem}; prefix {prefix} came from {tier}')


def configure(*, prefix: PrefixSetting = _UNSET,
              recorder: RecorderSetting = _UNSET) -> None:
    """Set the process-wide toolchain configuration.

    Call once, early in ``main()``. Only the arguments given are touched;
    passing ``None`` explicitly clears that tier (so resolution falls through to
    the environment or the default, and running stops being recorded).
    """
    global _prefix, _recorder
    if prefix is not _UNSET:
        # Absolutised once, here, rather than on every resolution: the prefix
        # then cannot shift under a stage that changes directory (see
        # :func:`_absolute`).
        _prefix = None if prefix is None else _absolute(prefix)
    if recorder is not _UNSET:
        _recorder = recorder


@contextmanager
def using(*, prefix: PrefixSetting = _UNSET,
          recorder: RecorderSetting = _UNSET):
    """Scope a configuration override, restoring the previous one on exit.

    For a caller that has to reach two toolchains in one process, and for tests
    that must not leak configuration into the next one. Restores on the way out
    however the block ends.
    """
    global _prefix, _recorder
    saved = _prefix, _recorder
    try:
        configure(prefix=prefix, recorder=recorder)
        yield
    finally:
        _prefix, _recorder = saved


def effective_prefix() -> Path:
    """The install prefix in force, resolved now."""
    return _prefix_and_tier(None)[0]


def _absolute(path: PathLike) -> Path:
    """A prefix as an absolute path, with ``~`` expanded.

    Every caller-supplied tier goes through this (:data:`DEFAULT_PREFIX` is
    already absolute), because a *relative* prefix is a trap rather than a
    convenience: :func:`resolve_exe` would check it against the current
    directory and pass, and then ``subprocess`` -- which resolves a relative
    argv[0] against ``cwd``, not against the directory it was launched from --
    would fail to exec for the two stages that run with ``cwd`` set to ``mvs/``.
    A tool that was just confirmed to exist would come back "not found".

    ``~`` matters for the same reason it does not for a parsed argument: a shell
    expands it before ``--path`` ever sees it, but ``$PGS_RECON_PREFIX`` can
    reach us with a literal tilde in it.
    """
    return Path(path).expanduser().resolve()


def _prefix_and_tier(explicit: Optional[PathLike]) -> Tuple[Path, str]:
    """The effective prefix and a phrase naming the tier it came from."""
    if explicit is not None:
        return _absolute(explicit), 'the prefix argument'
    if _prefix is not None:
        return _prefix, 'configure(prefix=...)'
    from_env = os.environ.get(PREFIX_ENV)
    if from_env:
        return _absolute(from_env), f'${PREFIX_ENV}'
    # Not through _absolute(): already absolute, and leaving the constant
    # untouched keeps the default's error messages predictable on a host where
    # /usr/local is itself a symlink.
    return DEFAULT_PREFIX, 'the built-in default'


def resolve_exe(name: str, subdir: str = MVG_BIN,
                prefix: Optional[PathLike] = None) -> Path:
    """Locate the executable ``name``, or raise :class:`ToolNotFound`.

    ``subdir`` selects the tool family (:data:`MVG_BIN`, :data:`MVS_BIN`).
    Validating here rather than letting ``exec`` fail is what makes the error
    say which prefix was searched and why that prefix was chosen.

    The result is always absolute, which :func:`run` relies on: a stage that
    passes ``cwd`` would otherwise have argv[0] re-interpreted against the new
    directory.
    """
    root, tier = _prefix_and_tier(prefix)
    exe = root / subdir / name
    if not exe.is_file():
        raise ToolNotFound(name, exe, root, tier)
    if not os.access(exe, os.X_OK):
        raise ToolNotFound(name, exe, root, tier, problem='is not executable')
    return exe


def cam_db(path: Optional[PathLike] = None,
           prefix: Optional[PathLike] = None) -> Path:
    """The OpenMVG camera sensor database: ``path`` if given, else prefix-relative.

    Deliberately not checked for existence, unlike :func:`resolve_exe`. Only one
    of the three importers passes this to a binary at all, and the two Python
    ones open it themselves -- so whoever reads it reports a missing file with
    more context than a check here could.

    Absolute either way, for the same reason an executable is: the one importer
    that passes it to a binary hands it over as an argument, and a relative path
    would mean something different to a stage running with ``cwd`` set.
    """
    if path is not None:
        return _absolute(path)
    return _prefix_and_tier(prefix)[0] / CAM_DB


class Recorder:
    """The record of what a run did, as it happens.

    Wraps the manifest's ``commands``: timestamp -> one command per entry. That
    mapping is a compatibility surface, not an internal detail --
    ``recon_dir._sfm_from_commands`` greps its *values* for ``openMVG2openMVS``
    to recover the solved SfM from manifests predating stage records -- so the
    shape stays timestamp -> joined string.

    It records Python steps too, via :meth:`step`. The two importers are not
    binary invocations, but a run that cannot say how its scene was imported is
    no more reproducible for the distinction.

    Not thread-safe: :meth:`_write`'s collision check is a read followed by a
    write. Nothing runs stages concurrently today (``convert.py``'s pool is pure
    image I/O and reaches no binary), so this is a note for whoever changes
    that, not a live defect.
    """

    def __init__(self, commands: MutableMapping[str, str]):
        self._commands = commands

    def command(self, command: Sequence) -> None:
        """Record an argv as a single readable line.

        Space-joined rather than shell-quoted: the line is for a human, and
        ``recon_dir._sfm_from_commands`` splits it on whitespace and reads the
        token after ``-i``, which quoting would break. An argument containing a
        space is therefore recorded faithfully but is not copy-pasteable.
        """
        self._write(' '.join(str(c) for c in command))

    def step(self, name: str, *args, **kwargs) -> None:
        """Record a Python step as a call: ``name(arg, key=value)``."""
        rendered = [str(a) for a in args]
        rendered += [f'{k}={v}' for k, v in kwargs.items()]
        self._write(f'{name}({", ".join(rendered)})')

    def note(self, text: str) -> None:
        """Record a bare line -- a milestone rather than an invocation."""
        self._write(text)

    def _write(self, entry: str) -> None:
        # Two entries can land in the same microsecond; a plain assignment would
        # drop the earlier one. Keys are only ever read by a human (see the
        # class docstring: consumers walk the values), so disambiguating one is
        # free.
        stamp = key = current_timestamp()
        n = 1
        while key in self._commands:
            n += 1
            key = f'{stamp} ({n})'
        self._commands[key] = entry


def run(command: Sequence, cwd: Optional[PathLike] = None,
        recorder: Optional[Recorder] = None) -> None:
    """Record ``command``, then run it to completion.

    The single chokepoint every wrapper goes through, and so the single place
    the manifest's command log is written -- recording before executing, so a
    binary that dies still leaves the invocation that killed it. Raises
    ``ToolFailed`` carrying the child's exit status.

    ``cwd`` is for the two binaries that write relative to their working
    directory rather than to an output path. Their argv[0] comes from
    :func:`resolve_exe` and is absolute, which is what keeps ``cwd`` from
    changing which file gets executed.
    """
    argv = [str(c) for c in command]
    rec = _recorder if recorder is None else recorder
    if rec is not None:
        rec.command(argv)
    # Called through the module global on purpose: patching
    # ``pgs_recon.toolchain.run_command`` is the test seam for argv assertions.
    run_command(argv, cwd=cwd)
