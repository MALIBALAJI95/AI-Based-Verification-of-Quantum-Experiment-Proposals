"""
Quantum Circuit Simulator using Qiskit with exact mathematical verification.
Includes statevector calculation, normalization checks, measurement probabilities,
and an exact NumPy tensor simulator fallback for robust testing.
"""

import time
import math
import cmath
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..schemas.proposal import CircuitSpec, GateOperation
from ..schemas.verification import SimulationOutput, StatevectorComponent

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


class SimulationError(RuntimeError):
    """Raised when circuit simulation encounters a numerical or structural failure."""
    pass


class QuantumSimulator:
    """
    Simulates supported ideal gate-based quantum circuits (1-5 qubits target, up to 10 qubits).
    Calculates exact statevectors, amplitudes, measurement probabilities, and normalization.
    """

    def __init__(self, force_numpy: bool = False):
        self.force_numpy = force_numpy or (not HAS_QISKIT)

    def simulate(self, circuit_spec: CircuitSpec) -> SimulationOutput:
        """
        Execute statevector simulation for the given CircuitSpec.
        """
        start_time = time.perf_counter()
        num_qubits = circuit_spec.num_qubits

        if num_qubits < 1 or num_qubits > 10:
            raise SimulationError(f"Supported qubit range is 1 to 10, got {num_qubits}.")

        if not self.force_numpy and HAS_QISKIT:
            try:
                return self._simulate_qiskit(circuit_spec, start_time)
            except Exception as e:
                # Fallback to NumPy engine if Qiskit fails
                return self._simulate_numpy(circuit_spec, start_time, fallback_note=str(e))
        else:
            return self._simulate_numpy(circuit_spec, start_time)

    def _simulate_qiskit(self, circuit_spec: CircuitSpec, start_time: float) -> SimulationOutput:
        num_qubits = circuit_spec.num_qubits
        qc = QuantumCircuit(num_qubits)

        for idx, gate in enumerate(circuit_spec.gates):
            name = gate.name.upper()
            targets = gate.target_qubits
            params = gate.params or []

            # Verify bounds
            for q in targets:
                if q < 0 or q >= num_qubits:
                    raise SimulationError(
                        f"Gate #{idx+1} ({name}) targets qubit {q}, which exceeds circuit size of {num_qubits} qubits."
                    )

            if name == "H":
                qc.h(targets[0])
            elif name == "X":
                qc.x(targets[0])
            elif name == "Y":
                qc.y(targets[0])
            elif name == "Z":
                qc.z(targets[0])
            elif name == "S":
                qc.s(targets[0])
            elif name == "T":
                qc.t(targets[0])
            elif name == "SDG":
                qc.sdg(targets[0])
            elif name == "TDG":
                qc.tdg(targets[0])
            elif name in ("I", "ID"):
                qc.id(targets[0])
            elif name == "RX":
                if not params:
                    raise SimulationError(f"RX gate #{idx+1} requires an angle parameter.")
                qc.rx(params[0], targets[0])
            elif name == "RY":
                if not params:
                    raise SimulationError(f"RY gate #{idx+1} requires an angle parameter.")
                qc.ry(params[0], targets[0])
            elif name == "RZ":
                if not params:
                    raise SimulationError(f"RZ gate #{idx+1} requires an angle parameter.")
                qc.rz(params[0], targets[0])
            elif name in ("CX", "CNOT"):
                qc.cx(targets[0], targets[1])
            elif name == "CZ":
                qc.cz(targets[0], targets[1])
            elif name == "SWAP":
                qc.swap(targets[0], targets[1])
            else:
                raise SimulationError(f"Gate '{name}' is not supported in the simulation engine.")

        # Compute statevector
        sv = Statevector.from_instruction(qc)
        sv_data = np.array(sv.data, dtype=complex)

        # Normalization check
        total_prob = float(np.sum(np.abs(sv_data) ** 2))
        norm_error = abs(total_prob - 1.0)
        is_normalized = norm_error <= 1e-6

        # Extract probability dictionary
        # In Qiskit, probabilities_dict() keys are bitstrings with q_{n-1}...q_0
        raw_probs = sv.probabilities_dict()
        probabilities: Dict[str, float] = {}
        for bitstring, p in raw_probs.items():
            probabilities[bitstring] = round(float(p), 6)

        # Fill all 2^n basis states even if 0 probability for complete distribution
        dim = 2 ** num_qubits
        for i in range(dim):
            bstring = format(i, f"0{num_qubits}b")
            if bstring not in probabilities:
                probabilities[bstring] = 0.0

        # Build detailed StatevectorComponent list
        components: List[StatevectorComponent] = []
        for i in range(dim):
            # In Qiskit, index i corresponds to bitstring reversed (little endian)
            bstring = format(i, f"0{num_qubits}b")
            # Index in sv_data corresponds to integer value of binary string reversed
            # e.g. for 2 qubits: bstring '01' (q1=0, q0=1) -> index 1.
            amp = sv_data[i]
            mag = float(abs(amp))
            prob = float(mag ** 2)
            phase = float(cmath.phase(amp))
            components.append(
                StatevectorComponent(
                    basis_state=bstring,
                    real=round(float(amp.real), 6),
                    imag=round(float(amp.imag), 6),
                    magnitude=round(mag, 6),
                    probability=round(prob, 6),
                    phase_rad=round(phase, 4)
                )
            )

        circuit_diagram = str(qc.draw(output="text"))
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SimulationOutput(
            num_qubits=num_qubits,
            statevector_components=components,
            probabilities=probabilities,
            is_normalized=is_normalized,
            normalization_error=round(norm_error, 9),
            circuit_diagram_text=circuit_diagram,
            execution_time_ms=round(elapsed, 2)
        )

    def _simulate_numpy(
        self, circuit_spec: CircuitSpec, start_time: float, fallback_note: Optional[str] = None
    ) -> SimulationOutput:
        """
        Exact tensor-product mathematical statevector simulation.
        Ensures 100% testable verification without external service or wheel issues.
        """
        num_qubits = circuit_spec.num_qubits
        dim = 2 ** num_qubits

        # Standard initial state |0...0>
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 + 0.0j

        # Single qubit gate unitary matrices (2x2)
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
        TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

        def rx_mat(theta: float) -> np.ndarray:
            return np.array([
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)]
            ], dtype=complex)

        def ry_mat(theta: float) -> np.ndarray:
            return np.array([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)]
            ], dtype=complex)

        def rz_mat(theta: float) -> np.ndarray:
            return np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=complex)

        # Helper to apply 1-qubit gate to statevector in little-endian ordering
        def apply_1q(u: np.ndarray, target: int, st: np.ndarray) -> np.ndarray:
            # Reshape into tensor of shape (2,)*num_qubits with axis 0 being q0
            shape = [2] * num_qubits
            st_tensor = st.reshape(shape)
            # Contract along axis 'target'
            # target axis contracted with axis 1 of u
            new_tensor = np.tensordot(u, st_tensor, axes=([1], [target]))
            # np.tensordot puts new axis at 0, move it back to target
            new_tensor = np.moveaxis(new_tensor, 0, target)
            return new_tensor.reshape(dim)

        # Helper to apply 2-qubit gate (control, target)
        def apply_cx(ctrl: int, tgt: int, st: np.ndarray) -> np.ndarray:
            new_st = np.copy(st)
            for idx in range(dim):
                # Check if control bit is 1
                if (idx >> ctrl) & 1:
                    # Flip target bit
                    flipped_idx = idx ^ (1 << tgt)
                    # Only do swap if idx < flipped_idx to avoid double swapping
                    if idx < flipped_idx:
                        tmp = new_st[idx]
                        new_st[idx] = new_st[flipped_idx]
                        new_st[flipped_idx] = tmp
            return new_st

        def apply_cz(ctrl: int, tgt: int, st: np.ndarray) -> np.ndarray:
            new_st = np.copy(st)
            for idx in range(dim):
                if ((idx >> ctrl) & 1) and ((idx >> tgt) & 1):
                    new_st[idx] = -new_st[idx]
            return new_st

        def apply_swap(q1: int, q2: int, st: np.ndarray) -> np.ndarray:
            new_st = np.copy(st)
            for idx in range(dim):
                b1 = (idx >> q1) & 1
                b2 = (idx >> q2) & 1
                if b1 != b2:
                    flipped = idx ^ (1 << q1) ^ (1 << q2)
                    if idx < flipped:
                        tmp = new_st[idx]
                        new_st[idx] = new_st[flipped]
                        new_st[flipped] = tmp
            return new_st

        for idx, gate in enumerate(circuit_spec.gates):
            name = gate.name.upper()
            targets = gate.target_qubits
            params = gate.params or []

            for q in targets:
                if q < 0 or q >= num_qubits:
                    raise SimulationError(
                        f"Gate #{idx+1} ({name}) targets qubit {q}, which exceeds circuit size of {num_qubits} qubits."
                    )

            if name == "H":
                state = apply_1q(H, targets[0], state)
            elif name == "X":
                state = apply_1q(X, targets[0], state)
            elif name == "Y":
                state = apply_1q(Y, targets[0], state)
            elif name == "Z":
                state = apply_1q(Z, targets[0], state)
            elif name == "S":
                state = apply_1q(S, targets[0], state)
            elif name == "T":
                state = apply_1q(T, targets[0], state)
            elif name == "SDG":
                state = apply_1q(SDG, targets[0], state)
            elif name == "TDG":
                state = apply_1q(TDG, targets[0], state)
            elif name in ("I", "ID"):
                pass
            elif name == "RX":
                if not params:
                    raise SimulationError(f"RX gate #{idx+1} requires an angle parameter.")
                state = apply_1q(rx_mat(params[0]), targets[0], state)
            elif name == "RY":
                if not params:
                    raise SimulationError(f"RY gate #{idx+1} requires an angle parameter.")
                state = apply_1q(ry_mat(params[0]), targets[0], state)
            elif name == "RZ":
                if not params:
                    raise SimulationError(f"RZ gate #{idx+1} requires an angle parameter.")
                state = apply_1q(rz_mat(params[0]), targets[0], state)
            elif name in ("CX", "CNOT"):
                state = apply_cx(targets[0], targets[1], state)
            elif name == "CZ":
                state = apply_cz(targets[0], targets[1], state)
            elif name == "SWAP":
                state = apply_swap(targets[0], targets[1], state)
            else:
                raise SimulationError(f"Unsupported gate '{name}'.")

        # Normalization check
        total_prob = float(np.sum(np.abs(state) ** 2))
        norm_error = abs(total_prob - 1.0)
        is_normalized = norm_error <= 1e-6

        # Formulate probabilities and components
        probabilities: Dict[str, float] = {}
        components: List[StatevectorComponent] = []

        for i in range(dim):
            # Bitstring format: standard little-endian representation (q_{n-1}...q_0)
            bitstring = ""
            for q in range(num_qubits - 1, -1, -1):
                bitstring += "1" if (i >> q) & 1 else "0"

            amp = state[i]
            mag = float(abs(amp))
            prob = float(mag ** 2)
            phase = float(cmath.phase(amp))

            probabilities[bitstring] = round(prob, 6)
            components.append(
                StatevectorComponent(
                    basis_state=bitstring,
                    real=round(float(amp.real), 6),
                    imag=round(float(amp.imag), 6),
                    magnitude=round(mag, 6),
                    probability=round(prob, 6),
                    phase_rad=round(phase, 4)
                )
            )

        # ASCII diagram representation
        diagram_lines = [f"q_{i}: |0> ---" for i in range(num_qubits)]
        for g in circuit_spec.gates:
            if len(g.target_qubits) == 1:
                q = g.target_qubits[0]
                diagram_lines[q] += f"[{g.name}]---"
            elif len(g.target_qubits) == 2:
                c, t = g.target_qubits
                diagram_lines[c] += f"-[*({g.name})]-"
                diagram_lines[t] += f"-[+({g.name})]-"
        diagram = "\n".join(diagram_lines)
        if fallback_note:
            diagram += f"\n(Simulated using exact NumPy tensor engine: {fallback_note})"

        elapsed = (time.perf_counter() - start_time) * 1000.0

        return SimulationOutput(
            num_qubits=num_qubits,
            statevector_components=components,
            probabilities=probabilities,
            is_normalized=is_normalized,
            normalization_error=round(norm_error, 9),
            circuit_diagram_text=diagram,
            execution_time_ms=round(elapsed, 2)
        )
