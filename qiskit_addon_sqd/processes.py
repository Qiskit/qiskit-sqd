# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Support for collective multi-process execution.

This module exists to support coordinated parallel execution across multiple
isolated processes, with explicit global synchronization and communication
(e.g., MPI-style SPMD collectives).

The current implementation relies on MPI, but the public interface is designed
to support other collective-execution backends in the future.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast


@dataclass(frozen=True)
class _ProcessGroup:
    mpi_enabled: bool = False
    mpi_comm: int | None = None
    process_index: int = 0
    process_count: int = 1


def _detect_mpi() -> _ProcessGroup:
    """Automatically detect if running under MPI and return state.

    This function is called at module import time to detect MPI environment.
    It checks for common MPI environment variables and attempts to import mpi4py
    if they are detected.
    """
    # Check for common MPI environment variables
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # Intel MPI, MPICH
        "SLURM_NTASKS",  # SLURM
        "MPI_LOCALNRANKS",  # IBM Spectrum MPI
    ]

    mpi_detected = any(var in os.environ for var in mpi_env_vars)

    if not mpi_detected:
        # single-process default
        return _ProcessGroup()

    try:
        from mpi4py import MPI
    except ImportError:
        # mpi4py not available, return single-process default
        return _ProcessGroup()

    comm = MPI.COMM_WORLD

    return _ProcessGroup(
        mpi_enabled=True,
        mpi_comm=comm,
        process_index=comm.Get_rank(),
        process_count=comm.Get_size(),
    )


def is_mpi_execution() -> bool:
    """Check if running in MPI mode.

    Returns:
       ``True`` if MPI is detected.
    """
    return _process_group.mpi_enabled


def _get_comm():
    """Get the MPI communicator.

    Returns:
        MPI communicator if distributed mode is enabled, ``None`` otherwise.
    """
    return _process_group.mpi_comm


def process_index() -> int:
    """Get the index of the current process.

    Returns:
        Process index (0 for the control process).
    """
    return _process_group.process_index


def process_count() -> int:
    """Get the total number of collective processes.

    Returns:
        Number of collective processes (1 if not in distributed mode).
    """
    return _process_group.process_count


def is_control_process() -> bool:
    """Return ``True`` if the current process is the control process.

    Returns:
        ``True`` if this is rank 0 or if not in distributed mode.
    """
    return process_index() == 0


def broadcast(obj: Any, root: int = 0) -> Any:
    """Broadcast a picklable object to all processes.

    Args:
        obj: Object to broadcast (only used on the root process).
        root: Root process index (default: 0 = broadcast from the control process).

    Returns:
        The broadcasted object on all ranks.
    """
    if not is_mpi_execution():
        return obj

    comm = _get_comm()
    return comm.bcast(obj, root=root)


_T = TypeVar("_T")


def run_on_control_process(func: Callable[[], _T]) -> _T:
    """Run a computation on the control process and give every process its outcome.

    Calls ``func`` on the control process only and broadcasts its return value to
    every process.  If ``func`` raises, the exception is re-raised on every process
    rather than on the control process alone.

    Use this for any step that has no distributed implementation.  Calling ``func``
    under :func:`is_control_process` and broadcasting the result directly leaves the
    other processes waiting in a broadcast that a failing control process never
    reaches, and MPI matches broadcasts by program order rather than by call site, so
    whichever broadcast the control process issues next is the one they receive.

    An exception that cannot be sent to the other processes aborts the job instead.
    See :func:`_prepare_failure` for why, and for what that costs.

    Every process must call this, since it contains a collective operation.  It is the
    calling process's rank, not any argument, that decides whether ``func`` runs here.

    Args:
        func: Callable taking no arguments.  Run on the control process only, so it may
            close over values that only that process has.

    Returns:
        Whatever ``func`` returned, on every process.

    Raises:
        BaseException: Whatever ``func`` raised, on every process.
    """
    if not is_mpi_execution():
        return func()

    # One broadcast, reached by every process on every path. Writing the root and
    # non-root sides as separate calls in separate branches is what made the bug this
    # function prevents possible, so the outcome is assembled first and sent once.
    outcome: tuple[bool, Any] = (True, None)
    original: BaseException | None = None
    if is_control_process():
        try:
            outcome = (True, func())
        except BaseException as exc:
            # Includes KeyboardInterrupt and SystemExit: under MPI, a Ctrl-C on the
            # control process must not leave the others waiting either.
            original = exc
            outcome = (False, _prepare_failure(exc))

    ok, payload = broadcast(outcome, root=0)
    if not ok:
        # Every process raises the same copy, so a caller's `except` runs on all of
        # them or none. On the control process the original is chained onto it, which
        # keeps that traceback reachable; pickle does not carry __cause__, so the other
        # processes raise the copy alone and report the control process's text instead.
        raise payload from original
    return cast("_T", payload)


def _prepare_failure(exc: BaseException) -> BaseException:
    """Return the copy of ``exc`` that every process will raise, or abort the job.

    Called on the control process only, and only under MPI: it reports the failure as
    that process's and may call ``MPI_Abort``, neither of which means anything
    elsewhere.  :func:`run_on_control_process` is the only caller, and broadcasts the
    returned copy to the other processes.

    The exception object itself is what gets sent, so that every process raises the
    same type and a caller's ``except`` runs on all of them or none.  An exception that
    cannot make the round trip aborts the job rather than being approximated: processes
    raising different types would run the caller's handler on only some of them, which
    is the failure this exists to prevent.

    The returned copy is what the control process raises too, rather than its own
    original.  A ``__reduce__`` may return a different type than it was given, so
    raising the original there would be the same divergence by another route.  The
    caller chains the original onto it, which keeps that traceback reachable on the
    control process, while the other processes raise the copy on its own.

    The round trip happens here, before the broadcast, because ``comm.bcast`` pickles
    on the root before it issues any MPI call -- so letting it discover the failure
    would strand the other processes in the very hang this avoids -- and unpickles on
    the receivers afterwards, so a payload that pickles but will not unpickle would
    raise a pickle error there instead of this exception.  Both directions are
    therefore checked locally, which costs nothing on an error path.

    Args:
        exc: The exception the control process's computation raised.

    Returns:
        A copy of ``exc`` that is known to survive a broadcast.  Carries no
        ``__cause__``, ``__context__`` or ``__traceback__``, since ``pickle`` does not
        preserve them.
    """
    try:
        payload: BaseException = pickle.loads(pickle.dumps(exc))
    except Exception:
        # No traceback survives MPI_Abort, so print this one while it still exists.
        print(
            f"qiskit-addon-sqd: process {process_index()} cannot send the following "
            "exception to the other processes, so the job is being aborted.",
            file=sys.stderr,
        )
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
        _get_comm().Abort(1)
        raise AssertionError(  # pragma: no cover
            "unreachable: MPI_Abort does not return"
        ) from exc

    return payload


def barrier() -> None:
    """Synchronize all processes.

    This is a no-op if not in distributed mode.
    """
    if is_mpi_execution():
        comm = _get_comm()
        comm.Barrier()


# Automatically detect MPI at module import time
_process_group = _detect_mpi()

# Made with Bob
