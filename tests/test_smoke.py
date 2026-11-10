"""Environment sanity checks: the package imports and the reference toolchain works."""

import sqsim


def test_package_imports():
    assert sqsim.__version__


def test_reference_simulator_bell_state():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = AerSimulator(seed_simulator=1).run(qc, shots=1000).result().get_counts()
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 1000
