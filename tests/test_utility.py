"""``run_command``: it raises, and the child's exit status survives.

Real subprocesses, but only short ``sys.executable -c`` calls -- no toolchain
binary is involved. The point of these tests is what ``subprocess`` reports back
and what ``ToolFailed`` makes of it, which a mock could only restate.

Every ``assertRaises(ToolFailed)`` here is also an assertion that nothing exits:
``sys.exit`` raises ``SystemExit``, which is not an ``Exception``, so it would
escape ``assertRaises`` and fail the test outright rather than being caught.
"""
import signal
import subprocess
import sys
import unittest
from unittest import mock

from pgs_recon.utility import ToolFailed, run_command

from test_stages import make_args
from test_tracker import TrackerCase


def python(code: str) -> list:
    return [sys.executable, '-c', code]


class TestRunCommand(unittest.TestCase):

    def test_success_does_not_raise(self):
        run_command(python('pass'))

    def test_nonzero_exit_is_preserved(self):
        with self.assertRaises(ToolFailed) as ctx:
            run_command(python('raise SystemExit(3)'))
        self.assertEqual(3, ctx.exception.returncode)
        self.assertEqual(3, ctx.exception.exit_code)

    def test_signal_death_reads_as_128_plus_signum(self):
        # The ADR 0004 case: an OOM-killed stage must not look like a bad
        # argument. 137 is what a shell (and a Slurm log) reports for SIGKILL.
        with self.assertRaises(ToolFailed) as ctx:
            run_command(python('import os, signal; '
                               'os.kill(os.getpid(), signal.SIGKILL)'))
        self.assertEqual(-signal.SIGKILL, ctx.exception.returncode)
        self.assertEqual(128 + int(signal.SIGKILL), ctx.exception.exit_code)
        self.assertIn('SIGKILL', str(ctx.exception))

    def test_missing_binary_raises(self):
        with self.assertRaises(ToolFailed) as ctx:
            run_command(['pgs-no-such-binary', '--flag'])
        self.assertIsNone(ctx.exception.returncode)
        self.assertEqual(127, ctx.exception.exit_code)

    def test_message_carries_the_command(self):
        with self.assertRaises(ToolFailed) as ctx:
            run_command(python('raise SystemExit(1)'))
        self.assertIn(' '.join(python('raise SystemExit(1)')),
                      str(ctx.exception))
        self.assertEqual(python('raise SystemExit(1)'),
                         ctx.exception.command)

    def test_keyboard_interrupt_propagates(self):
        # The bare ``except`` this replaced swallowed Ctrl-C into an exit,
        # leaving a long run's interrupt indistinguishable from a tool failure.
        with mock.patch('subprocess.run', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_command(python('pass'))

    def test_other_subprocess_errors_raise(self):
        err = subprocess.TimeoutExpired(python('pass'), 1.0)
        with mock.patch('subprocess.run', side_effect=err):
            with self.assertRaises(ToolFailed) as ctx:
                run_command(python('pass'))
        self.assertIsNone(ctx.exception.returncode)
        self.assertIn('TimeoutExpired', str(ctx.exception))


class TestTrackerRecordsFailure(TrackerCase):
    """A raising ``run_command`` still leaves a ``failed`` stage record.

    ``main()`` no longer exits inside the stage, so the ``except BaseException``
    that calls ``abort()`` is what keeps the manifest legible. Mirrors that
    block: abort, then re-raise.
    """

    def test_failed_stage_is_recorded_and_the_error_propagates(self):
        tracker = self.tracker(make_args())
        with self.assertRaises(ToolFailed):
            try:
                self.assertTrue(tracker.begin('import'))
                run_command(python('raise SystemExit(9)'))
            except BaseException:
                tracker.abort()
                raise

        record = self.records()['import']
        self.assertEqual('failed', record['status'])
        self.assertIn('elapsed_s', record)


if __name__ == '__main__':
    unittest.main()
