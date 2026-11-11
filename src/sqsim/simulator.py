"""Public entry point: a drop-in for Qiskit's simulators.

    counts = sqsim.Simulator().run(qc, shots=1024).result().get_counts()
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from sqsim.convert import from_qiskit
from sqsim.engines import dense
from sqsim.engines.dense import SeedLike


class Result:
    """Outcome of one run. Mirrors qiskit.result.Result.get_counts()."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class Job:
    """Finished job. Simulation runs synchronously, so result() returns immediately."""

    def __init__(self, result: Result) -> None:
        self._result = result

    def result(self) -> Result:
        return self._result


class Simulator:
    """Samples measurement counts of a QuantumCircuit (currently with the dense engine)."""

    def run(self, circuit: QuantumCircuit, shots: int = 1024, seed: SeedLike = None) -> Job:
        flat = dense.sample_counts(from_qiskit(circuit), shots, seed=seed)
        counts = {_split_registers(circuit, key): n for key, n in flat.items()}
        return Job(Result(counts))


def _split_registers(qc: QuantumCircuit, key: str) -> str:
    """Insert spaces between classical registers like Qiskit, e.g. "011" -> "01 1".

    The last-added register comes first; within a register, bit 0 is rightmost.
    """
    if len(qc.cregs) <= 1:
        return key
    width = len(key)
    parts = []
    for creg in reversed(qc.cregs):
        indices = [qc.find_bit(bit).index for bit in creg]
        parts.append("".join(key[width - 1 - i] for i in reversed(indices)))
    return " ".join(parts)
