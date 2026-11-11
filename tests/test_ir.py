"""IR construction, validation and gate classification."""

import numpy as np
import pytest
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CXGate,
    CYGate,
    HGate,
    RXGate,
    RZGate,
    SwapGate,
    TGate,
    UGate,
    YGate,
)

from sqsim.ir import Circuit, Gate, GateKind, Measure, classify

X = [[0, 1], [1, 0]]


@pytest.mark.parametrize(
    "qiskit_gate, kind",
    [
        (TGate(), GateKind.DIAGONAL),
        (RZGate(0.4), GateKind.DIAGONAL),
        (CPhaseGate(0.3), GateKind.DIAGONAL),
        (CXGate(), GateKind.PERMUTATION),
        (SwapGate(), GateKind.PERMUTATION),
        (CCXGate(), GateKind.PERMUTATION),
        (YGate(), GateKind.MONOMIAL),
        (CYGate(), GateKind.MONOMIAL),
        (HGate(), GateKind.DENSE),
        (RXGate(0.2), GateKind.DENSE),
        (UGate(0.1, 0.2, 0.3), GateKind.DENSE),
    ],
    ids=lambda v: getattr(v, "name", None),
)
def test_classify_standard_gates(qiskit_gate, kind):
    assert classify(qiskit_gate.to_matrix()) is kind


def test_classify_tolerates_float_noise():
    noisy_identity = np.eye(2) + 1e-15
    assert classify(noisy_identity) is GateKind.DIAGONAL


def test_gate_normalizes_fields():
    g = Gate("rz", [np.int64(2)], RZGate(0.5).to_matrix(), params=[0.5])
    assert g.qubits == (2,) and isinstance(g.qubits[0], int)
    assert g.params == (0.5,)
    assert g.matrix.dtype == np.complex128
    assert g.kind is GateKind.DIAGONAL
    assert np.allclose(g.diagonal(), np.diag(RZGate(0.5).to_matrix()))


def test_gate_matrix_is_read_only():
    g = Gate("x", (0,), X)
    with pytest.raises(ValueError):
        g.matrix[0, 0] = 5


@pytest.mark.parametrize(
    "qubits, matrix, match",
    [
        ((), X, "no qubits"),
        ((0, 0), np.eye(4), "repeated"),
        ((0, 1), X, "needs a"),
    ],
)
def test_gate_validation(qubits, matrix, match):
    with pytest.raises(ValueError, match=match):
        Gate("g", qubits, matrix)


def test_circuit_validates_indices():
    c = Circuit(num_qubits=2, num_clbits=1)
    with pytest.raises(ValueError, match="out of range"):
        c.add_gate(Gate("x", (2,), X))
    with pytest.raises(ValueError, match="qubit"):
        c.add_measure(Measure(qubit=5, clbit=0))
    with pytest.raises(ValueError, match="clbit"):
        c.add_measure(Measure(qubit=0, clbit=1))
    with pytest.raises(ValueError):
        Circuit(num_qubits=1, gates=[Gate("x", (1,), X)])
