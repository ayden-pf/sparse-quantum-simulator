"""Dense state-vector engine, checked against Qiskit's exact Statevector and Aer."""

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit_aer import AerSimulator

from sqsim.convert import from_qiskit
from sqsim.engines import dense


def aer_counts(qc, shots):
    aer = AerSimulator(seed_simulator=7)
    # Aer rejects some library gates (e.g. c3sx) unless transpiled to its basis first.
    raw = aer.run(transpile(qc, aer), shots=shots).result().get_counts()
    return {k.replace(" ", ""): v for k, v in raw.items()}  # drop register separators


# --- statevector -----------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_statevector_random(seed):
    qc = random_circuit(6, 10, max_operands=3, seed=seed)
    np.testing.assert_allclose(
        dense.statevector(from_qiskit(qc)), Statevector(qc).data, atol=1e-10
    )


def test_statevector_includes_global_phase():
    qc = QuantumCircuit(1, global_phase=np.pi / 3)
    qc.h(0)
    np.testing.assert_allclose(dense.statevector(from_qiskit(qc)), Statevector(qc).data)


def test_endianness_single_x():
    qc = QuantumCircuit(3)
    qc.x(0)
    psi = dense.statevector(from_qiskit(qc))
    assert np.flatnonzero(psi).tolist() == [1]  # qubit 0 is bit 0 of the index


def test_empty_circuit_is_all_zeros():
    psi = dense.statevector(from_qiskit(QuantumCircuit(4)))
    assert psi[0] == 1 and np.count_nonzero(psi) == 1


# --- sampling ----------------------------------------------------------------


def test_counts_bit_order_matches_qiskit():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure([0, 1, 2], [0, 1, 2])
    assert dense.sample_counts(from_qiskit(qc), 100, seed=0) == {"001": 100}


def test_clbit_mapping_and_unmeasured_clbits():
    qc = QuantumCircuit(2, 4)
    qc.x(1)
    qc.measure(1, 3)  # qubit 1 (=1) -> clbit 3
    qc.measure(0, 1)  # qubit 0 (=0) -> clbit 1; clbits 0 and 2 never written
    assert dense.sample_counts(from_qiskit(qc), 50, seed=0) == {"1000": 50}
    assert aer_counts(qc, 50) == {"1000": 50}


def test_partial_measurement_marginalizes():
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.measure(2, 0)
    assert dense.sample_counts(from_qiskit(qc), 200, seed=0) == {"1": 200}


def test_marginal_probabilities_order():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0, 0)  # measurement 0 is the most significant bit of the outcome index
    qc.measure(1, 1)
    np.testing.assert_allclose(dense.marginal_probabilities(from_qiskit(qc)), [0, 0, 1, 0])


def test_seed_reproducible_and_shots_sum():
    qc = random_circuit(5, 6, measure=True, seed=3)
    c = from_qiskit(qc)
    a = dense.sample_counts(c, 1000, seed=42)
    assert a == dense.sample_counts(c, 1000, seed=42)
    assert sum(a.values()) == 1000
    assert all(len(k) == qc.num_clbits for k in a)


def test_sample_counts_input_validation():
    with pytest.raises(ValueError, match="no classical bits"):
        dense.sample_counts(from_qiskit(QuantumCircuit(1)), 10)
    qc = QuantumCircuit(1, 1)
    with pytest.raises(ValueError, match="shots"):
        dense.sample_counts(from_qiskit(qc), 0)


# --- agreement with Aer --------------------------------------------------------

SHOTS = 20_000


def _bell_multi_register():
    q, c1, c2 = QuantumRegister(3), ClassicalRegister(1), ClassicalRegister(2)
    qc = QuantumCircuit(q, c1, c2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.7, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


@pytest.mark.parametrize(
    "qc",
    [_bell_multi_register()] + [random_circuit(5, 8, measure=True, seed=s) for s in range(5)],
    ids=["multi-register"] + [f"random-{s}" for s in range(5)],
)
def test_counts_match_aer(qc):
    ours = dense.sample_counts(from_qiskit(qc), SHOTS, seed=1)
    assert hellinger_fidelity(ours, aer_counts(qc, SHOTS)) > 0.99


@pytest.mark.parametrize("name, size", [("GHZ", 6), ("MerminBell", 4), ("HamiltonianSimulation", 4)])
def test_supermarq_counts_match_aer(name, size):
    supermarq = pytest.importorskip("supermarq")
    qc = getattr(supermarq.benchmarks, name)(size).qiskit_circuit()
    ours = dense.sample_counts(from_qiskit(qc), SHOTS, seed=1)
    assert hellinger_fidelity(ours, aer_counts(qc, SHOTS)) > 0.99
