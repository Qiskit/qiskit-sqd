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

"""Tests of collective behavior, run under an MPI launcher by ``tox -e mpi``.

Everything here is marked ``mpi`` and is selected by ``tox -e mpi``, so these tests
only run under ``mpirun``.  They assert on behavior that cannot differ between
processes: that a failing call fails on every process rather than on the control process
alone, and that it fails the same way everywhere.

A test that asserts "every process raises" has to be written so that a process which
*wrongly* does not raise fails the test rather than hanging the job.  A bare
``pytest.raises`` cannot do that: on a process where the call returns instead of raising,
execution runs on into the next collective and waits there forever, and a hung job gives
no failure report.  So each test records its own outcome, then compares outcomes across
processes through a collective that every process reaches on every path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from qiskit_addon_sqd.fermion import (
    diagonalize_fermionic_hamiltonian,
)
from qiskit_addon_sqd.processes import is_control_process, process_count

# `tox -e mpi` selects this file with `-m mpi`; nothing else runs it. The marker alone
# would not keep it out of `tox -e py`, where an unknown marker is not a skip and every
# test here would be collected and fail, so gate on the imports that only the `mpi`
# environment installs.
pytest.importorskip("pytest_mpi", reason="MPI tests run under `tox -e mpi`")
pytest.importorskip("mpi4py", reason="MPI tests run under `tox -e mpi`")

pytestmark = pytest.mark.mpi


def _gather_outcomes(outcome: object) -> list:
    """Collect one process's outcome onto every process.

    Uses ``allgather`` directly rather than anything in
    :mod:`qiskit_addon_sqd.processes`, so that a bug in the code under test cannot
    affect how the test reports.  Every process must reach this call on every path.
    """
    from mpi4py import MPI

    return MPI.COMM_WORLD.allgather(outcome)


def _record(call) -> tuple[str, str]:
    """Run ``call`` and describe what it did, without raising either way."""
    try:
        call()
    except BaseException as exc:
        return (type(exc).__name__, str(exc))
    return ("<returned>", "")


def _tiny_problem(*, valid: bool) -> dict:
    """A 4-orbital problem whose samples are either all valid or all invalid.

    With ``valid=False`` no sample has the right Hamming weights, so postselection
    leaves nothing and the call must raise ``ValueError``.
    """
    norb, nelec = 4, (2, 2)
    if valid:
        bitstrings = np.tile([0, 0, 1, 1, 0, 0, 1, 1], (10, 1)).astype(bool)
    else:
        bitstrings = np.zeros((10, 2 * norb), dtype=bool)
    return dict(
        one_body_tensor=np.eye(norb),
        two_body_tensor=np.zeros((norb,) * 4),
        bit_array=bitstrings,
        samples_per_batch=4,
        norb=norb,
        nelec=nelec,
        max_iterations=1,
    )


def test_the_launcher_actually_started_more_than_one_process():
    """Guard the rest of the file: at one process nothing here tests anything.

    ``tox -e mpi`` runs under ``mpirun -n 2`` by default.  Were the launcher missing or
    the detection in :mod:`qiskit_addon_sqd.processes` broken, every other test in this
    file would pass trivially while testing nothing collective.
    """
    assert process_count() > 1, (
        f"expected more than one process, got {process_count()}. "
        "These tests must run under an MPI launcher; use `tox -e mpi`."
    )


def test_invalid_bitstrings_raise_on_every_process():
    """The "no valid bitstrings" ValueError must reach every process.

    This is the failure in #451: the check runs on the control process alone, so the
    other processes used to wait in a broadcast that never came.
    """
    outcome = _record(lambda: diagonalize_fermionic_hamiltonian(**_tiny_problem(valid=False)))
    outcomes = _gather_outcomes(outcome)

    returned = [i for i, (name, _) in enumerate(outcomes) if name == "<returned>"]
    assert not returned, (
        f"processes {returned} returned a result instead of raising; outcomes={outcomes}"
    )
    names = {name for name, _ in outcomes}
    assert names == {"ValueError"}, f"processes disagreed on the exception: {outcomes}"
    messages = {message for _, message in outcomes}
    assert len(messages) == 1, f"processes disagreed on the message: {outcomes}"
    assert "did not contain any valid bitstrings" in messages.pop()


def test_a_failed_call_does_not_desync_the_next_one():
    """After a call that raises, a later call must still agree across processes.

    Without the fix, the processes stranded by the first call are released by the second
    call's broadcasts and return *its* result as the first call's, so a call that should
    have raised returns a result computed from different inputs.
    """
    first = _record(lambda: diagonalize_fermionic_hamiltonian(**_tiny_problem(valid=False)))
    assert _gather_outcomes(first[0]) == [first[0]] * process_count(), (
        "the processes did not agree on whether the first call raised"
    )

    problem = _tiny_problem(valid=True)
    problem["one_body_tensor"] = 2 * np.eye(problem["norb"])
    result = diagonalize_fermionic_hamiltonian(**problem)

    energies = _gather_outcomes(result.energy)
    assert len(set(energies)) == 1, f"processes disagreed on the energy: {energies}"


def test_an_exception_that_cannot_be_sent_aborts_the_job(tmp_path):
    """An exception the other processes cannot be given must abort, not strand them.

    ``SubspacePolicy`` is user code and runs on the control process alone, so the same
    asymmetry applies to anything it raises.  The exception class here is one ``pickle``
    cannot reconstruct -- its ``__init__`` takes two arguments while ``args`` holds one
    -- which is the case a fix that merely broadcast the exception would get wrong,
    raising the original on the control process and a pickle error on the others.

    Runs in a nested launcher rather than in this job, because the behavior under test
    is ``MPI_Abort``: asserting on it here would kill the process running the assertion
    and take the rest of the suite with it.  Only the control process launches the
    child, so the ranks of this job stay in step; the others wait at the ``allgather``
    below, which doubles as the barrier that keeps them from racing ahead.
    """
    script = tmp_path / "aborts.py"
    script.write_text(
        textwrap.dedent(
            """
            import numpy as np
            from qiskit_addon_sqd.fermion import (
                StandardPolicy,
                diagonalize_fermionic_hamiltonian,
            )

            class UnreconstructableError(Exception):
                def __init__(self, stage, detail):
                    super().__init__(f"{stage}: {detail}")
                    self.stage, self.detail = stage, detail

            class FailingPolicy(StandardPolicy):
                def refine(self, results, round_index):
                    raise UnreconstructableError("refine", "deliberate failure")

            norb, nelec = 4, (2, 2)
            diagonalize_fermionic_hamiltonian(
                np.eye(norb),
                np.zeros((norb,) * 4),
                np.tile([0, 0, 1, 1, 0, 0, 1, 1], (10, 1)).astype(bool),
                samples_per_batch=4,
                norb=norb,
                nelec=nelec,
                max_iterations=1,
                policy=FailingPolicy(),
            )
            """
        )
    )

    outcome = None
    if is_control_process():
        launcher = shutil.which("mpirun") or shutil.which("mpiexec")
        if launcher is None:
            outcome = "skip: no mpirun or mpiexec on PATH"
        else:
            # Strip this job's launcher variables, or the nested mpirun inherits them
            # and refuses to start its own.
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_LOCAL"))
            }
            completed = subprocess.run(
                [launcher, "-n", "2", sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
                check=False,
            )
            if completed.returncode == 0:
                outcome = f"the job exited 0 instead of aborting; stderr={completed.stderr[-2000:]}"
            elif "refine: deliberate failure" not in completed.stderr:
                outcome = (
                    "aborted without reporting the original exception; "
                    f"stderr={completed.stderr[-2000:]}"
                )

    # Every process reaches this, so the ranks stay in step whatever the control
    # process found.
    outcome = _gather_outcomes(outcome)[0]
    if isinstance(outcome, str) and outcome.startswith("skip:"):
        pytest.skip(outcome[len("skip:") :].strip())
    assert outcome is None, outcome
