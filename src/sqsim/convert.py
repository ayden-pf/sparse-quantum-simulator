"""Front end: convert a Qiskit QuantumCircuit into the sqsim IR.

This is the only module (besides the public simulator wrapper) that knows about
Qiskit's data model. Engines consume Circuit objects exclusively.

Supported input
---------------
* Any unitary operation: standard gates, UnitaryGate, composite gates such
  as QFTGate, and gate-only sub-circuits appended via to_instruction().
  The matrix comes from Qiskit itself, so no basis-gate transpilation is needed.
* measure, as long as each measured qubit is not acted on afterwards
  (terminal measurement) and each clbit is written at most once.
* barrier and delay are ignored (no effect on an ideal simulation).

Everything else (reset, control flow such as if_test, mid-circuit
measurement, unbound parameters) raises UnsupportedCircuitError.
"""

from __future__ import annotations

import numbers

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, Gate as QiskitGate, Instruction, ParameterExpression
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator

from sqsim.ir import Circuit, Gate, Measure

_IGNORED = frozenset({"barrier", "delay"})


class UnsupportedCircuitError(ValueError):
    """The circuit uses a feature the sqsim front end cannot represent."""


def from_qiskit(qc: QuantumCircuit) -> Circuit:
    """Convert qc to an IR :class:~sqsim.ir.Circuit.

    Qubit/clbit indices are Qiskit's flat indices (qc.find_bit(bit).index),
    and gate matrices keep Qiskit's little-endian ordering, so no bit
    reordering happens here.
    """
    if qc.parameters:
        names = ", ".join(sorted(p.name for p in qc.parameters))
        raise UnsupportedCircuitError(
            f"circuit has unbound parameters ({names}); "
            "bind them first with qc.assign_parameters({...})"
        )

    circuit = Circuit(
        num_qubits=qc.num_qubits,
        num_clbits=qc.num_clbits,
        global_phase=float(qc.global_phase),
    )
    measured_qubits: set[int] = set()
    written_clbits: set[int] = set()

    for position, inst in enumerate(qc.data):
        op = inst.operation
        name = op.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        # Used for debugging errors
        where = f"instruction {position} ({name!r} on qubits {qubits})"

        if name in _IGNORED:
            continue

        if name == "measure":
            (qubit,) = qubits
            (clbit,) = (qc.find_bit(c).index for c in inst.clbits)
            if qubit in measured_qubits:
                raise UnsupportedCircuitError(f"{where}: qubit {qubit} is measured twice")
            if clbit in written_clbits:
                raise UnsupportedCircuitError(f"{where}: clbit {clbit} is written twice")
            measured_qubits.add(qubit)
            written_clbits.add(clbit)
            circuit.add_measure(Measure(qubit=qubit, clbit=clbit))
            continue

        if isinstance(op, ControlFlowOp):
            raise UnsupportedCircuitError(f"{where}: control flow is not supported")

        touched = measured_qubits.intersection(qubits)
        if touched:
            raise UnsupportedCircuitError(
                f"{where}: acts on already-measured qubit(s) {sorted(touched)}; "
                "only terminal measurements are supported"
            )

        circuit.add_gate(
            Gate(
                name=name,
                qubits=qubits,
                matrix=_unitary(op, where),
                params=_numeric_params(op),
            )
        )

    return circuit


def _unitary(op: Instruction, where: str) -> np.ndarray:
    """Return the unitary matrix of op in Qiskit's little-endian ordering."""
    if op.num_clbits:
        raise UnsupportedCircuitError(f"{where}: operation uses classical bits")
    if isinstance(op, QiskitGate):
        try:
            return op.to_matrix()
        except (AttributeError, QiskitError, TypeError):
            pass  # no direct matrix; fall back to synthesizing from its definition
    try:
        # Operator recurses through op.definition, which covers composite gates
        # and gate-only sub-circuits appended with to_instruction().
        return Operator(op).data
    except QiskitError as exc:
        raise UnsupportedCircuitError(f"{where}: not a unitary operation") from exc


def _numeric_params(op: Instruction) -> tuple[float, ...]:
    """Real scalar params of op. Non-scalar params (e.g. UnitaryGate's matrix) are dropped."""
    out = []
    for p in op.params:
        if isinstance(p, ParameterExpression):
            out.append(float(p))  # parameters are bound (checked in from_qiskit)
        elif isinstance(p, numbers.Real):
            out.append(float(p))
    return tuple(out)
