"""Intermediate representation (IR) consumed by every sqsim engine.

The front end converts a Qiskit QuantumCircuit object into a Circuit class.
Engines (dense, sparse, MPS, ...) read only this IR and never touch Qiskit.

Conventions
-----------
* Qubits and clbits are plain integer indices. 
* Gate matrices use **Qiskit's little-endian convention relative to the gate's
  own qubit tuple**: qubits[0] is the least significant bit of the matrix
  row/column index. This is exactly what Gate.to_matrix() returns, so the
  converter can copy matrices through unchanged. Engines are responsible for
  mapping this onto their own state layout.
  States are seen as | q_n, q_n-1 ,..., q_0 >
* Measurements are terminal only (for now): the engine evolves the state through
  all of Circuit.gates and then samples Circuit.measurements.
  Mid-circuit measurement is not yet supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

ATOL = 1e-12
"""Absolute tolerance used when deciding whether a matrix entry is zero / one."""


class GateKind(Enum):
    """Structural class of a gate's matrix, from most to least exploitable.
    """

    DIAGONAL = "diagonal"
    """Only diagonal entries are nonzero (Z, S, T, RZ, P, CZ, CP, ...).
    Applying it is an elementwise multiply on the state."""

    PERMUTATION = "permutation"
    """A 0/1 permutation matrix (X, CX, SWAP, CCX, ...).
    Applying it is a reindexing of the state with no arithmetic."""

    MONOMIAL = "monomial"
    """Exactly one nonzero per row and column, i.e. permutation x diagonal (Y, CY, ...).
    Applying it is a reindex plus an elementwise multiply."""

    DENSE = "dense"
    """No exploitable structure (H, RX, RY, U, ...)."""


def classify(matrix: np.ndarray, atol: float = ATOL) -> GateKind:
    """Return the most specific GateKind describing matrix.""" 
    nonzero = np.abs(matrix) > atol
    if not nonzero.any():       # all values non-zero, dense
        return GateKind.DENSE  

    off_diag = nonzero & ~np.eye(matrix.shape[0], dtype=bool)
    if not off_diag.any():
        return GateKind.DIAGONAL

    one_per_row = (nonzero.sum(axis=1) == 1).all()
    one_per_col = (nonzero.sum(axis=0) == 1).all()
    if one_per_row and one_per_col:
        if np.allclose(matrix[nonzero], 1.0, atol=atol):
            return GateKind.PERMUTATION
        return GateKind.MONOMIAL

    return GateKind.DENSE


@dataclass(frozen=True, eq=False)
class Gate:
    """A unitary acting on a small set of qubits.

    eq=False because np.ndarray does not support a boolean ==;
    gates compare by identity.
    """

    name: str
    """Qiskit operation name ("h", "cx", "unitary", ...). Informational."""

    qubits: tuple[int, ...]
    """Target qubits; qubits[0] is the least significant bit of matrix."""

    matrix: np.ndarray
    """matrix repr of quantum gate with k size."""

    params: tuple[float, ...] = ()
    """Numeric gate parameters (already bound). Informational; matrix is authoritative."""

    kind: GateKind = field(init=False)
    """Computed from matrix at construction time."""

    def __post_init__(self) -> None:
        k = len(self.qubits)
        if k == 0:
            raise ValueError(f"gate {self.name!r} acts on no qubits")
        if len(set(self.qubits)) != k:
            raise ValueError(f"gate {self.name!r} has repeated qubits {self.qubits}")

        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.shape != (2**k, 2**k):
            raise ValueError(
                f"gate {self.name!r} on {k} qubit(s) needs a {(2**k, 2**k)} matrix, "
                f"got {matrix.shape}"
            )
        matrix.setflags(write=False)  # keep the frozen dataclass actually immutable

        # frozen=True blocks normal assignment, so set fields via object.__setattr__.
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))
        object.__setattr__(self, "kind", classify(matrix))

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def diagonal(self) -> np.ndarray:
        """Diagonal entries; meaningful only when kind is GateKind.DIAGONAL."""
        return np.diagonal(self.matrix)


@dataclass(frozen=True)
class Measure:
    """Terminal measurement of qubit (computational basis) into clbit."""

    qubit: int
    clbit: int


@dataclass
class Circuit:
    """A full circuit in engine-ready form."""

    num_qubits: int
    num_clbits: int = 0
    gates: list[Gate] = field(default_factory=list)
    """Unitary operations in program order."""
    measurements: list[Measure] = field(default_factory=list)
    """Terminal measurements, applied after all gates."""
    global_phase: float = 0.0
    """Overall phase e^{i*global_phase}; irrelevant to counts, relevant to statevectors."""

    def __post_init__(self) -> None:
        for gate in self.gates:
            self._check_gate(gate)
        for m in self.measurements:
            self._check_measure(m)

    def add_gate(self, gate: Gate) -> None:
        self._check_gate(gate)
        self.gates.append(gate)

    def add_measure(self, measure: Measure) -> None:
        self._check_measure(measure)
        self.measurements.append(measure)

    def _check_gate(self, gate: Gate) -> None:
        if any(not 0 <= q < self.num_qubits for q in gate.qubits):
            raise ValueError(
                f"gate {gate.name!r} on qubits {gate.qubits} is out of range "
                f"for a {self.num_qubits}-qubit circuit"
            )

    def _check_measure(self, m: Measure) -> None:
        if not 0 <= m.qubit < self.num_qubits:
            raise ValueError(f"measure qubit {m.qubit} out of range")
        if not 0 <= m.clbit < self.num_clbits:
            raise ValueError(f"measure clbit {m.clbit} out of range")
