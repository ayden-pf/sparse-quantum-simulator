"""Public Simulator API, checked against Aer."""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

import sqsim


def aer_counts(qc, shots):
    return AerSimulator(seed_simulator=7).run(qc, shots=shots).result().get_counts()


def test_run_bell_matches_aer():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    counts = sqsim.Simulator().run(qc, shots=10_000, seed=1).result().get_counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 10_000
    assert hellinger_fidelity(counts, aer_counts(qc, 10_000)) > 0.99


def test_register_spacing_matches_aer():
    q, a, b = QuantumRegister(3), ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
    qc = QuantumCircuit(q, a, b)
    qc.x(1)
    qc.measure([0, 1, 2], [0, 1, 2])  # a[0]=q0=0, b[0]=q1=1, b[1]=q2=0
    ours = sqsim.Simulator().run(qc, shots=100, seed=0).result().get_counts()
    assert ours == aer_counts(qc, 100) == {"01 0": 100}


def test_default_shots():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    assert sqsim.Simulator().run(qc).result().get_counts() == {"0": 1024}
