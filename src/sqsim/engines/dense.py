"""Dense state-vector engine: the correctness baseline.

The full 2**n state is stored, and every gate is applied as a small k-qubit
tensor contracted into the state, so no 2**n x 2**n operator is ever built.
This engine deliberately ignores ``Gate.kind``. It is the plain reference that
the structured and MPS engines get compared against.

Memory is ``16 * 2**n`` bytes for the state (n=25 -> 512 MiB, n=30 -> 16 GiB),
plus one temporary copy per gate.

Layout
------
The flat state follows Qiskit's little-endian convention: amplitude index
``i`` has qubit ``q`` in bit ``q`` of ``i``. Reshaped to the tensor
``(2,) * n`` (C order), **axis ``n - 1 - q`` belongs to qubit ``q``**.
"""

from __future__ import annotations

import numpy as np

from sqsim.ir import Circuit, Gate

SeedLike = int | np.random.Generator | None


def statevector(circuit: Circuit) -> np.ndarray:
    """Final state of ``circuit`` started from |0...0>, as a flat little-endian vector.

    Includes the global phase, so it is directly comparable to
    ``qiskit.quantum_info.Statevector(qc).data``. Measurements are ignored.
    """
    n = circuit.num_qubits
    psi = np.zeros((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0
    for gate in circuit.gates:
        psi = _apply(psi, gate)
    return np.exp(1j * circuit.global_phase) * psi.reshape(-1)


def sample_counts(circuit: Circuit, shots: int, seed: SeedLike = None) -> dict[str, int]:
    """Simulate ``circuit`` and sample its terminal measurements ``shots`` times.

    Returns a dict that maps a bitstring of length ``num_clbits`` (clbit 0
    rightmost, as in Qiskit) to its count. Outcomes that never occur are left
    out, and clbits that are never measured read as 0. Keys have no spaces
    between classical registers. The IR doesn't know about registers, so
    inserting those spaces is the public API's job.
    """
    if circuit.num_clbits == 0:
        raise ValueError("circuit has no classical bits, so there is nothing to count")
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")

    probs = marginal_probabilities(circuit)
    probs = probs / probs.sum()  # remove float drift before sampling
    rng = np.random.default_rng(seed)
    outcome_counts = rng.multinomial(shots, probs)

    m = len(circuit.measurements)
    counts: dict[str, int] = {}
    for outcome in np.flatnonzero(outcome_counts):
        value = 0
        for j, meas in enumerate(circuit.measurements):
            bit = (int(outcome) >> (m - 1 - j)) & 1
            value |= bit << meas.clbit
        counts[f"{value:0{circuit.num_clbits}b}"] = int(outcome_counts[outcome])
    return counts


def marginal_probabilities(circuit: Circuit) -> np.ndarray:
    """Probabilities of the measured qubits' joint outcomes.

    Entry ``i`` is the probability of outcome ``i``, read big-endian over
    ``circuit.measurements``: measurement 0 is the most significant bit. With
    no measurements, the result is ``[1.0]``.
    """
    n = circuit.num_qubits
    probs = np.abs(statevector(circuit)) ** 2
    measured_axes = [n - 1 - meas.qubit for meas in circuit.measurements]
    probs = np.moveaxis(probs.reshape((2,) * n), measured_axes, range(len(measured_axes)))
    return probs.reshape(2 ** len(measured_axes), -1).sum(axis=1)


def _apply(psi: np.ndarray, gate: Gate) -> np.ndarray:
    """Contract ``gate`` into the state tensor ``psi`` (shape ``(2,) * n``)."""
    n = psi.ndim
    k = gate.num_qubits
    # Gate matrix rows/cols are little-endian over gate.qubits, so after
    # reshaping to (2,)*2k the axes are [out q_{k-1}..q_0, in q_{k-1}..q_0].
    u = gate.matrix.reshape((2,) * (2 * k))
    state_axes = [n - 1 - q for q in reversed(gate.qubits)]
    # tensordot puts the gate's k output axes first and the untouched axes after;
    # moveaxis puts the output axes back where those qubits live.
    out = np.tensordot(u, psi, axes=(list(range(k, 2 * k)), state_axes))
    return np.moveaxis(out, list(range(k)), state_axes)
