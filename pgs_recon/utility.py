import signal
import subprocess
from datetime import datetime as dt, timezone as tz
from typing import List, Optional, Sequence


def current_timestamp() -> str:
    return dt.now(tz.utc).strftime("%m/%d/%Y, %H:%M:%S.%f %Z")


class ToolFailed(Exception):
    """An external binary failed. Carries the child's exit status.

    Raised instead of ``sys.exit`` so the child's status survives to the caller:
    ``sys.exit(str)`` always exits 1, which made an OOM-killed ``RefineMesh``
    (the failure ADR 0004 exists for) indistinguishable from a bad argument.
    ``main()`` catches this and exits ``exit_code``. Same move ``StageError``
    already made for the planner.
    """

    def __init__(self, command: Sequence[str], returncode: Optional[int],
                 detail: Optional[str] = None):
        self.command = [str(c) for c in command]
        #: The child's ``subprocess`` returncode: negative for a signal death,
        #: ``None`` if the binary never started.
        self.returncode = returncode
        self.detail = detail
        super().__init__(f'{self._reason()}: {" ".join(self.command)}')

    def _reason(self) -> str:
        if self.detail is not None:
            return f'Command {self.detail}'
        if self.returncode is not None and self.returncode < 0:
            num = -self.returncode
            try:
                name = signal.Signals(num).name
            except ValueError:
                name = 'unknown'
            return f'Command killed by signal {num} ({name})'
        return f'Command failed with exit code {self.returncode}'

    @property
    def exit_code(self) -> int:
        """This failure as a POSIX exit status, for ``sys.exit``.

        A signal death becomes ``128 + signum`` (so an OOM kill reads as 137 in
        a Slurm log) and a binary that never started becomes 127, matching what
        a shell would report. Anything else passes the child's code through.
        """
        if self.returncode is None:
            return 127
        if self.returncode < 0:
            return 128 - self.returncode
        return self.returncode


def run_command(cmd: List[str], cwd=None):
    """Run an external binary to completion, raising ``ToolFailed`` on failure.

    Deliberately no bare ``except``: ``KeyboardInterrupt`` propagates so Ctrl-C
    still aborts a run promptly, rather than being reported as a tool failure.
    """
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except OSError as e:
        raise ToolFailed(cmd, None, detail=f'failed to start ({e})') from e
    except subprocess.CalledProcessError as e:
        raise ToolFailed(cmd, e.returncode) from e
    except subprocess.SubprocessError as e:
        raise ToolFailed(cmd, None,
                         detail=f'failed ({type(e).__name__}: {e})') from e
