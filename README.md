# sparse-quantum-simulator

A gate-based quantum circuit simulator for Python that improves compute and memory efficiency by exploiting circuit structure. It is designed as a drop-in replacement for Qiskit's built-in simulators.

This is a course project for CSC 569 (Quantum Computing), a graduate class at North Carolina State University.

> **Status:** Early development. This README describes the project's planned design and scope; the implementation is not available yet.

## Development setup

Requires Python 3.11 or newer (tested on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bench]"
pytest
```

The `dev` extra installs `pytest` and `qiskit-aer`, which serves as the reference simulator. The `bench` extra installs `supermarq` for generating benchmark circuits. The package is imported as `sqsim`.

## Motivation

Access to quantum hardware is limited, and today's devices produce unreliable results because of decoherence and hardware fragility. Classical simulators give programmers a way to run quantum circuits without hardware, and their noiseless execution lets researchers separate algorithm bugs from hardware errors.

The cost is scale: a full state vector doubles in size with every added qubit, so memory use grows exponentially. Dense matrix simulation also stores and processes every matrix entry, even when most entries are zero or follow a predictable pattern. This project reduces that unnecessary storage and computation by using the algebraic structure of circuit operations.

## Approach

### Sparse representations

Most gates in realistic circuits are highly structured: phase gates are diagonal, CNOT and Toffoli are permutations, and controlled gates leave half the state untouched. The simulator will:

- Classify gates by their structure and store each in a correspondingly compact form
- Apply gates directly to the state rather than building large operators
- Combine sequences of structured gates into a single operation where possible

### Tensor networks

Treating a circuit as a network of connected tensors, rather than a fixed sequence of matrix multiplications, makes the contraction order a choice. The project will explore:

- Search strategies for finding good contraction paths
- When memory limits make it worthwhile to trade extra computation for a smaller footprint
- Matrix product states (MPS) for circuits with chain-like connectivity
- Qubit reordering, since keeping the declared qubit order can place interacting qubits far apart

### Specialized operations

- Carry gate structure into the tensor network, so diagonal and permutation gates cost far less than general gates
- Identify portions of a circuit that belong to classes with known efficient simulation methods and route them accordingly
- Parallelize within individual operations and across independent parts of a contraction, targeting the ARC cluster

## Planned implementation

The library will accept Qiskit `QuantumCircuit` objects. The simulator class will provide a `run` function that returns a result object containing a counts dictionary of measurement outcomes.

Internally, the simulator will:

1. Convert the `QuantumCircuit` into the engine's representation, accounting for Qiskit's qubit endianness
2. Simulate the circuit with an MPS-based engine, with possible extension to tree tensor networks (TTNs) if time permits
3. Run tensor calculations with GPU parallelism, likely using PyTorch

## Evaluation

Test circuits of varying size and type, such as GHZ and Mermin-Bell, will be generated with the [SupermarQ](https://arxiv.org/abs/2202.11045) benchmark library. Each circuit will run on both Qiskit's simulators and this one, and the results will be compared on:

| Metric | Method |
| --- | --- |
| Correctness | Hellinger fidelity |
| Time | Wrapping the simulator calls |
| Memory usage | To be determined; comparing CPU and GPU implementations needs further investigation |

## References

- R. Orús, *A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States*, [arXiv:1306.2164](https://arxiv.org/abs/1306.2164)
- U. Schollwöck, *The density-matrix renormalization group in the age of matrix product states*, [arXiv:1008.3477](https://arxiv.org/abs/1008.3477)
- G. Vidal, *Efficient classical simulation of slightly entangled quantum computations*, [arXiv:quant-ph/0301063](https://arxiv.org/abs/quant-ph/0301063)
- T. Tomesh et al., *SupermarQ: A Scalable Quantum Benchmark Suite*, [arXiv:2202.11045](https://arxiv.org/abs/2202.11045)
