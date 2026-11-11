"""Qiskit -> IR converter.

These tests don't depend on any sqsim engine. The oracle is Qiskit itself: the
IR is rebuilt as a QuantumCircuit made of UnitaryGates, and its full Operator
has to equal the original circuit's (global phase included).
"""

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, random_unitary

from sqsim.convert import UnsupportedCircuitError, from_qiskit
from sqsim.ir import Circuit, GateKind, Measure


def rebuild(c: Circuit) -> QuantumCircuit:
    qc = QuantumCircuit(c.num_qubits, global_phase=c.global_phase)
    for g in c.gates:
        qc.append(UnitaryGate(g.matrix, check_input=False), list(g.qubits))
    return qc


def assert_same_unitary(qc: QuantumCircuit) -> None:
    expected = Operator(qc.remove_final_measurements(inplace=False)).data
    actual = Operator(rebuild(from_qiskit(qc))).data
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("seed", range(20))
def test_random_circuits(seed):
    assert_same_unitary(random_circuit(5, 8, max_operands=3, seed=seed))


def test_registers_composites_and_qubit_order():
    a, b = QuantumRegister(2, "a"), QuantumRegister(2, "b")
    qc = QuantumCircuit(a, b, global_phase=0.7)
    qc.h(b[1])
    qc.append(QFTGate(3), [3, 0, 2])  # composite gate on out-of-order qubits
    qc.append(UnitaryGate(random_unitary(4, seed=1)), [2, 0])
    sub = QuantumCircuit(2)
    sub.ry(0.3, 0)
    sub.cx(0, 1)
    qc.append(sub.to_instruction(), [1, 3])  # plain Instruction, not a Gate
    assert_same_unitary(qc)

    c = from_qiskit(qc)
    assert [g.qubits for g in c.gates] == [(3,), (3, 0, 2), (2, 0), (1, 3)]
    assert c.global_phase == pytest.approx(0.7)


def test_metadata_and_measurements():
    qr, cr = QuantumRegister(3), ClassicalRegister(3)
    qc = QuantumCircuit(qr, cr)
    qc.rz(0.5, 0)
    qc.cx(0, 1)
    qc.append(UnitaryGate(np.eye(2)), [2])
    qc.barrier()
    qc.delay(100, 1)
    qc.measure([0, 1, 2], [2, 0, 1])

    c = from_qiskit(qc)
    assert (c.num_qubits, c.num_clbits) == (3, 3)
    assert [g.name for g in c.gates] == ["rz", "cx", "unitary"]  # barrier/delay dropped
    assert [g.kind for g in c.gates] == [GateKind.DIAGONAL, GateKind.PERMUTATION, GateKind.DIAGONAL]
    assert c.gates[0].params == (0.5,)
    assert c.gates[2].params == ()  # UnitaryGate's matrix param is dropped
    assert c.measurements == [Measure(0, 2), Measure(1, 0), Measure(2, 1)]


def test_bound_parameters_are_accepted():
    t = Parameter("t")
    qc = QuantumCircuit(1, global_phase=t)
    qc.rx(t, 0)
    c = from_qiskit(qc.assign_parameters({t: 0.25}))
    assert c.gates[0].params == (0.25,)
    assert c.global_phase == pytest.approx(0.25)


def test_gate_on_other_qubit_after_measure_is_allowed():
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.x(1)
    assert len(from_qiskit(qc).gates) == 1


def _gate_after_measure(qc):
    qc.measure(0, 0)
    qc.h(0)


def _qubit_measured_twice(qc):
    qc.measure(0, 0)
    qc.measure(0, 1)


def _clbit_written_twice(qc):
    qc.measure(0, 0)
    qc.measure(1, 0)


def _control_flow(qc):
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)


@pytest.mark.parametrize(
    "build, match",
    [
        (_gate_after_measure, "already-measured"),
        (_qubit_measured_twice, "measured twice"),
        (_clbit_written_twice, "written twice"),
        (_control_flow, "control flow"),
        (lambda qc: qc.reset(0), "not a unitary"),
        (lambda qc: qc.rx(Parameter("theta"), 0), "unbound parameters"),
    ],
    ids=["gate-after-measure", "double-measure", "double-clbit", "if-test", "reset", "unbound"],
)
def test_unsupported(build, match):
    qc = QuantumCircuit(2, 2)
    build(qc)
    with pytest.raises(UnsupportedCircuitError, match=match):
        from_qiskit(qc)


def test_unsupported_is_a_value_error():
    assert issubclass(UnsupportedCircuitError, ValueError)


@pytest.mark.parametrize("name, size", [("GHZ", 6), ("MerminBell", 4), ("HamiltonianSimulation", 4)])
def test_supermarq_circuits(name, size):
    supermarq = pytest.importorskip("supermarq")
    qc = getattr(supermarq.benchmarks, name)(size).qiskit_circuit()
    assert_same_unitary(qc)
    assert len(from_qiskit(qc).measurements) == qc.count_ops().get("measure", 0)
