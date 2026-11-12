"""Turning a simulated state into Qiskit-style counts. Shared by every engine.

Counts keys are bitstrings of length ``num_clbits`` with clbit 0 rightmost, as
in Qiskit. Outcomes that never occur are left out, clbits that are never
measured read as 0, and there are no spaces between classical registers (the
IR doesn't know about registers, so inserting those is the public API's job).

Keys are built as strings rather than packed into integers, so circuits with
more than 63 measured qubits work (the MPS engine reaches those sizes).
"""

from __future__ import annotations

import numpy as np

from sqsim.ir import Circuit

SeedLike = int | np.random.Generator | None

_ZERO = ord("0")


def check_sampling_args(circuit: Circuit, shots: int) -> None:
    """Raise ``ValueError`` if ``circuit`` can't be sampled ``shots`` times."""
    if circuit.num_clbits == 0:
        raise ValueError("circuit has no classical bits, so there is nothing to count")
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}")


def marginals_from_statevector(psi: np.ndarray, circuit: Circuit) -> np.ndarray:
    """Probabilities of the measured qubits' joint outcomes, from a full statevector.

    ``psi`` is a flat little-endian vector (qubit ``q`` is bit ``q`` of the index).
    Entry ``i`` of the result is the probability of outcome ``i``, read
    big-endian over ``circuit.measurements``: measurement 0 is the most
    significant bit. With no measurements, the result is ``[1.0]``.
    """
    n = circuit.num_qubits
    probs = np.abs(psi) ** 2
    measured_axes = [n - 1 - meas.qubit for meas in circuit.measurements]
    probs = np.moveaxis(probs.reshape((2,) * n), measured_axes, range(len(measured_axes)))
    return probs.reshape(2 ** len(measured_axes), -1).sum(axis=1)


def counts_from_probabilities(
    probs: np.ndarray, circuit: Circuit, shots: int, seed: SeedLike = None
) -> dict[str, int]:
    """Sample ``shots`` outcomes from ``probs`` (as returned by :func:`marginals_from_statevector`)."""
    probs = probs / probs.sum()  # remove float drift before sampling
    rng = np.random.default_rng(seed)
    hits = rng.multinomial(shots, probs)
    outcomes = np.flatnonzero(hits)
    m = len(circuit.measurements)
    bits = (outcomes[:, None] >> np.arange(m - 1, -1, -1)) & 1
    return counts_from_samples(bits, circuit, weights=hits[outcomes])


def counts_from_samples(
    bits: np.ndarray, circuit: Circuit, weights: np.ndarray | None = None
) -> dict[str, int]:
    """Build a counts dict from per-sample measurement outcomes.

    ``bits`` has shape ``(samples, len(circuit.measurements))``; column ``j``
    is the outcome of ``circuit.measurements[j]``. ``weights`` (default all
    ones) says how many shots each row stands for. Rows that format to the same
    key are merged.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1, len(circuit.measurements))
    if weights is None:
        weights = np.ones(len(bits), dtype=np.int64)

    chars = np.full((len(bits), circuit.num_clbits), _ZERO, dtype=np.uint8)
    columns = [circuit.num_clbits - 1 - meas.clbit for meas in circuit.measurements]
    chars[:, columns] = _ZERO + bits

    rows, inverse = np.unique(chars, axis=0, return_inverse=True)
    totals = np.bincount(inverse.reshape(-1), weights=weights, minlength=len(rows))
    return {
        row.tobytes().decode("ascii"): int(total)
        for row, total in zip(rows, totals)
        if total > 0
    }
