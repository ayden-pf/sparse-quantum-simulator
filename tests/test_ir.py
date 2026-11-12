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


# --- control detection -------------------------------------------------------

from qiskit.circuit.library import CHGate, CRYGate, CZGate, MCXGate, RYGate, RZZGate  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

from sqsim.ir import split_controls  # noqa: E402


@pytest.mark.parametrize(
    "qiskit_gate, controls, target",
    [
        (CXGate(), (0,), X),
        (CHGate(), (0,), HGate().to_matrix()),
        (CRYGate(0.4), (0,), RYGate(0.4).to_matrix()),
        (CCXGate(), (0, 1), X),
        (Operator(MCXGate(3)), (0, 1, 2), X),
        (CZGate(), (0, 1), [[-1]]),  # symmetric: every bit is a control
        (CPhaseGate(0.3), (0, 1), [[np.exp(0.3j)]]),
        (HGate(), (), HGate().to_matrix()),
        (SwapGate(), (), SwapGate().to_matrix()),
        (RZZGate(0.5), (), RZZGate(0.5).to_matrix()),  # diagonal, but no control
    ],
    ids=lambda v: getattr(v, "name", None),
)
def test_split_controls(qiskit_gate, controls, target):
    matrix = qiskit_gate.data if isinstance(qiskit_gate, Operator) else qiskit_gate.to_matrix()
    found, sub = split_controls(matrix)
    assert found == controls
    np.testing.assert_allclose(sub, np.asarray(target, dtype=complex), atol=1e-12)


def test_split_controls_finds_control_in_any_position():
    # CX with the control on the gate's second qubit (bit 1).
    cx_reversed = CXGate().to_matrix()[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]
    controls, sub = split_controls(cx_reversed)
    assert controls == (1,)
    np.testing.assert_allclose(sub, X)
