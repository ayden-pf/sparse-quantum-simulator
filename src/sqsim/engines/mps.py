from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from sqsim.engines.sampling import SeedLike, check_sampling_args, counts_from_samples
from sqsim.ir import Circuit, Gate, GateKind

DEFAULT_CUTOFF = 1e-24  # Default relative discarded weight per SVD. Drops only numerical noise.


# What a simulation cost. All fields are cumulative over the run.
@dataclass
class MPSStats:
    max_bond_dimension: int = 1  # Largest bond dimension reached at any point.
    num_swaps: int = 0  # SWAPs inserted to make gate qubits adjacent.
    num_svds: int = 0
    discarded_weight: float = 0.0  # Sum of relative discarded weights over all truncations.
    # Product of 1 - discarded over all truncations (1.0 means exact).
    fidelity_estimate: float = 1.0
    peak_state_bytes: int = 0  # Largest total size of the site tensors seen after any gate.


# A mutable matrix product state on num_qubits qubits, starting in |0...0>.
class MPS:
    def __init__(
        self,
        num_qubits: int,
        *,
        layout: Sequence[int] | None = None,
        max_bond_dimension: int | None = None,
        cutoff: float = DEFAULT_CUTOFF,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("an MPS needs at least one qubit")
        if max_bond_dimension is not None and max_bond_dimension < 1:
            raise ValueError(f"max_bond_dimension must be positive, got {max_bond_dimension}")
        if cutoff < 0:
            raise ValueError(f"cutoff must be non-negative, got {cutoff}")

        layout = list(range(num_qubits)) if layout is None else [int(q) for q in layout]
        if sorted(layout) != list(range(num_qubits)):
            raise ValueError(f"layout must be a permutation of range({num_qubits}), got {layout}")

        self.layout = layout  # layout[s] is the qubit at site s.
        # site_of[q] is the site holding qubit q (inverse of layout).
        self.site_of = [0] * num_qubits
        for site, q in enumerate(layout):
            self.site_of[q] = site

        zero = np.zeros((1, 2, 1), dtype=np.complex128)
        zero[0, 0, 0] = 1.0
        self.tensors = [zero.copy() for _ in range(num_qubits)]
        self.center = 0
        self.max_bond_dimension = max_bond_dimension
        self.cutoff = cutoff
        self.stats = MPSStats(peak_state_bytes=self.state_bytes)

    # --- inspection ------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return len(self.tensors)

    # Bond dimension between site s and s + 1, for each s.
    @property
    def bond_dimensions(self) -> list[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    @property
    def state_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    # Contract into a flat little-endian statevector (2**n entries; small n only).
    def to_statevector(self) -> np.ndarray:
        n = self.num_qubits
        psi = self.tensors[0].reshape(2, -1)
        for t in self.tensors[1:]:
            psi = (psi @ t.reshape(t.shape[0], -1)).reshape(-1, t.shape[2])
        # Axis s is site s (site 0 most significant). Qiskit wants qubit q on axis n-1-q.
        psi = psi.reshape((2,) * n).transpose([self.site_of[n - 1 - a] for a in range(n)])
        return psi.reshape(-1)

    # --- gates -----------------------------------------------------------------

    def apply_gate(self, gate: Gate) -> None:
        if gate.num_qubits == 1:
            self._apply_one(gate)
        else:
            self._apply_block(gate)
        self.stats.peak_state_bytes = max(self.stats.peak_state_bytes, self.state_bytes)

    def _apply_one(self, gate: Gate) -> None:
        site = self.site_of[gate.qubits[0]]
        a = self.tensors[site]
        if gate.kind is GateKind.DIAGONAL:
            self.tensors[site] = a * gate.diagonal()[None, :, None]
        else:
            self.tensors[site] = np.einsum("ps,lsr->lpr", gate.matrix, a)

    def _apply_block(self, gate: Gate) -> None:
        k = gate.num_qubits
        start = self._gather(gate.qubits)
        self._move_center_into(start, start + k - 1)

        theta = self.tensors[start]
        for t in self.tensors[start + 1 : start + k]:
            theta = np.tensordot(theta, t, axes=(theta.ndim - 1, 0))
        # theta axes: [left bond, site start, ..., site start+k-1, right bond].

        # Gate bit b acts on qubit gate.qubits[b]. Reshaped, the matrix's axes are
        # [out b_{k-1}..b_0, in b_{k-1}..b_0], so match "in" axes to theta in that order.
        u = gate.matrix.reshape((2,) * (2 * k))
        theta_axes = [1 + self.site_of[q] - start for q in reversed(gate.qubits)]
        out = np.tensordot(u, theta, axes=(list(range(k, 2 * k)), theta_axes))
        theta = np.moveaxis(out, list(range(k)), theta_axes)

        self._split(theta, start, center_left=False)

    # SWAP qubits onto consecutive sites. Returns the first site of the block.
    #
    # The leftmost qubit stays put and every other one is moved left until
    # it's next to the previous one. Each qubit only passes over sites that
    # don't hold gate qubits, so already-placed ones never move again.
    def _gather(self, qubits: Sequence[int]) -> int:
        sites = sorted(self.site_of[q] for q in qubits)
        start = sites[0]
        for m, site in enumerate(sites[1:], start=1):
            for s in range(site - 1, start + m - 1, -1):
                self._swap(s)
        return start

    # Exchange the qubits on sites s and s + 1.
    def _swap(self, s: int) -> None:
        self._move_center_into(s, s + 1)
        theta = np.tensordot(self.tensors[s], self.tensors[s + 1], axes=(2, 0))
        # Moving left, so leave the center on the left site, where the next swap needs it.
        self._split(theta.transpose(0, 2, 1, 3), s, center_left=True)

        qa, qb = self.layout[s], self.layout[s + 1]
        self.layout[s], self.layout[s + 1] = qb, qa
        self.site_of[qa], self.site_of[qb] = s + 1, s
        self.stats.num_swaps += 1

    # --- canonical form and truncation -----------------------------------------

    # Move the orthogonality center (with QR steps) until lo <= center <= hi.
    def _move_center_into(self, lo: int, hi: int) -> None:
        while self.center < lo:
            c = self.center
            a = self.tensors[c]
            q, r = np.linalg.qr(a.reshape(-1, a.shape[2]))
            self.tensors[c] = q.reshape(a.shape[0], 2, -1)
            self.tensors[c + 1] = np.tensordot(r, self.tensors[c + 1], axes=(1, 0))
            self.center += 1
        while self.center > hi:
            c = self.center
            a = self.tensors[c]
            # a = r^H q^H, where q^H has orthonormal rows (an RQ decomposition).
            q, r = np.linalg.qr(a.reshape(a.shape[0], -1).conj().T)
            self.tensors[c] = q.conj().T.reshape(-1, 2, a.shape[2])
            self.tensors[c - 1] = np.tensordot(self.tensors[c - 1], r.conj().T, axes=(2, 0))
            self.center -= 1

    # Split block theta (shape (chi_l, 2, ..., 2, chi_r)) back into sites.
    #
    # Sweeps toward the end where the orthogonality center should be left:
    # the first site of the block if center_left, otherwise the last.
    def _split(self, theta: np.ndarray, start: int, *, center_left: bool) -> None:
        k = theta.ndim - 2
        chi_l, chi_r = theta.shape[0], theta.shape[-1]
        if not center_left:
            for m in range(k - 1):
                u, s, vh = self._svd(theta.reshape(chi_l * 2, -1))
                self.tensors[start + m] = u.reshape(chi_l, 2, -1)
                chi_l = s.size
                theta = s[:, None] * vh
            self.tensors[start + k - 1] = theta.reshape(chi_l, 2, chi_r)
            self.center = start + k - 1
        else:
            for m in range(k - 1, 0, -1):
                u, s, vh = self._svd(theta.reshape(-1, 2 * chi_r))
                self.tensors[start + m] = vh.reshape(-1, 2, chi_r)
                chi_r = s.size
                theta = u * s[None, :]
            self.tensors[start] = theta.reshape(chi_l, 2, chi_r)
            self.center = start

    # Truncated SVD, renormalized so the state keeps its norm. Updates stats.
    def _svd(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, s, vh = _svd(matrix)
        weights = s**2
        total = weights.sum()
        keep = _num_to_keep(weights, total, self.cutoff, self.max_bond_dimension)

        self.stats.num_svds += 1
        if keep < s.size:
            kept = weights[:keep].sum()
            discarded = float((total - kept) / total) if total > 0 else 0.0
            u, s, vh = u[:, :keep], s[:keep] * np.sqrt(total / kept), vh[:keep]
            self.stats.discarded_weight += discarded
            self.stats.fidelity_estimate *= 1.0 - discarded
        self.stats.max_bond_dimension = max(self.stats.max_bond_dimension, keep)
        return u, s, vh

    # --- sampling ----------------------------------------------------------------

    # Draw shots computational-basis samples of all qubits.
    #
    # Returns (bits, weights): bits[i, q] is qubit q's value in distinct
    # outcome i, and weights[i] is how many shots produced it (summing to shots).
    #
    # Sites are sampled left to right with the rest of the chain
    # right-canonical, so each site's conditional probabilities come from
    # one small contraction. Shots are carried as weighted branches and split
    # with binomial draws, so the work grows with the number of *distinct*
    # prefixes rather than with shots (e.g. 2 branches for GHZ).
    def sample(self, shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        self._move_center_into(0, 0)
        envs = np.ones((1, 1), dtype=np.complex128)  # (branches, chi_left), unit norm
        weights = np.array([shots], dtype=np.int64)
        bits = np.zeros((1, 0), dtype=np.uint8)

        for t in self.tensors:
            chi_l, _, chi_r = t.shape
            v = (envs @ t.reshape(chi_l, -1)).reshape(-1, 2, chi_r)
            p = np.einsum("bpr,bpr->bp", v, v.conj()).real  # unnormalized P(prefix, bit)
            p1 = np.clip(p[:, 1] / p.sum(axis=1), 0.0, 1.0)
            n1 = rng.binomial(weights, p1)
            n0 = weights - n1

            children_envs, children_weights, children_bits = [], [], []
            for bit, counts in ((0, n0), (1, n1)):
                alive = counts > 0
                children_envs.append(v[alive, bit, :] / np.sqrt(p[alive, bit])[:, None])
                children_weights.append(counts[alive])
                new_bits = np.full((alive.sum(), 1), bit, dtype=np.uint8)
                children_bits.append(np.hstack([bits[alive], new_bits]))
            envs = np.concatenate(children_envs)
            weights = np.concatenate(children_weights)
            bits = np.concatenate(children_bits)

        return bits[:, self.site_of], weights


# --- engine functions ----------------------------------------------------------


# Run every gate of circuit on a fresh MPS and return it (with .stats).
def simulate(
    circuit: Circuit,
    *,
    max_bond_dimension: int | None = None,
    cutoff: float = DEFAULT_CUTOFF,
    layout: Sequence[int] | None = None,
) -> MPS:
    mps = MPS(
        circuit.num_qubits,
        layout=layout,
        max_bond_dimension=max_bond_dimension,
        cutoff=cutoff,
    )
    for gate in circuit.gates:
        mps.apply_gate(gate)
    return mps


# Final state as a flat little-endian vector, global phase included. Small n only.
# options are passed to simulate().
def statevector(circuit: Circuit, **options) -> np.ndarray:
    mps = simulate(circuit, **options)
    return np.exp(1j * circuit.global_phase) * mps.to_statevector()


# Simulate and sample. Returns the counts and the final MPS, whose .stats say what it cost.
def run(
    circuit: Circuit, shots: int, seed: SeedLike = None, **options
) -> tuple[dict[str, int], MPS]:
    check_sampling_args(circuit, shots)
    mps = simulate(circuit, **options)
    bits, weights = mps.sample(shots, np.random.default_rng(seed))
    measured = [meas.qubit for meas in circuit.measurements]
    return counts_from_samples(bits[:, measured], circuit, weights=weights), mps


# Simulate circuit and sample its terminal measurements (same output format as the dense engine).
def sample_counts(circuit: Circuit, shots: int, seed: SeedLike = None, **options) -> dict[str, int]:
    counts, _ = run(circuit, shots, seed, **options)
    return counts


# --- helpers -------------------------------------------------------------------


def _svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        # LAPACK's default divide-and-conquer driver occasionally fails to
        # converge; the slower QR-iteration driver is more robust.
        return scipy.linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")


# How many of the (descending) singular values to keep.
def _num_to_keep(
    weights: np.ndarray, total: float, cutoff: float, max_bond: int | None
) -> int:
    tail = np.cumsum(weights[::-1])  # tail[i] = total weight of the i+1 smallest values.
    droppable = int(np.searchsorted(tail, cutoff * total, side="right"))
    keep = max(1, weights.size - droppable)
    if max_bond is not None:
        keep = min(keep, max_bond)
    return keep
