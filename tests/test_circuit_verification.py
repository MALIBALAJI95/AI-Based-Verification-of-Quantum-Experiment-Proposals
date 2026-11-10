"""
Automated Pytest Suite: Quantum Circuit Verification & Simulation.
Tests Case 1 (Bell state PASS), Case 2 (Incorrect distribution FAIL),
Case 3 (Invalid qubit index FAIL), and malformed gate syntax rejection.
"""

import pytest
import numpy as np

from src.schemas.proposal import CircuitSpec, GateOperation, ExpectedOutcome, ProposalSpecification
from src.circuit.parser import parse_circuit_dsl, CircuitParseError
from src.circuit.simulator import QuantumSimulator, SimulationError
from src.circuit.metrics import compare_outcomes, total_variation_distance, compute_fidelity
from src.verification.rules import check_qubit_count, check_circuit_syntax, check_outcome_distribution


@pytest.fixture
def simulator():
    return QuantumSimulator()


def test_case1_bell_state_ideal_verification(simulator):
    """
    Mandatory Case 1: Two-qubit Bell-state circuit:
    |00> -> H on qubit 0 -> CX(0,1).
    Expected ideal state is (|00> + |11>)/sqrt(2), with measurement probabilities 0.5 for 00 and 0.5 for 11.
    Status: MUST PASS.
    """
    spec = CircuitSpec(
        num_qubits=2,
        gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1])
        ]
    )

    sim_output = simulator.simulate(spec)

    assert sim_output.is_normalized is True
    assert sim_output.normalization_error < 1e-6
    assert abs(sim_output.probabilities["00"] - 0.5) < 1e-4
    assert abs(sim_output.probabilities["11"] - 0.5) < 1e-4
    assert sim_output.probabilities["01"] < 1e-4
    assert sim_output.probabilities["10"] < 1e-4

    # Outcome comparison check
    expected = [
        ExpectedOutcome(basis_state="00", expected_probability=0.5, tolerance=0.05),
        ExpectedOutcome(basis_state="11", expected_probability=0.5, tolerance=0.05)
    ]
    comparisons, all_matched, max_error = compare_outcomes(expected, sim_output.probabilities)

    assert all_matched is True
    assert max_error < 0.05
    assert len(comparisons) == 2


def test_case2_deliberately_incorrect_distribution(simulator):
    """
    Mandatory Case 2: Deliberately incorrect expected measurement distribution for Bell circuit.
    Hypothesized: 0.8 on 00 and 0.2 on 11.
    Status: MUST DETECT DISCREPANCY (FAIL).
    """
    spec = CircuitSpec(
        num_qubits=2,
        gates=[
            GateOperation(name="H", target_qubits=[0]),
            GateOperation(name="CX", target_qubits=[0, 1])
        ]
    )

    sim_output = simulator.simulate(spec)

    # Biased expected probabilities
    expected = [
        ExpectedOutcome(basis_state="00", expected_probability=0.8, tolerance=0.05),
        ExpectedOutcome(basis_state="11", expected_probability=0.2, tolerance=0.05)
    ]
    comparisons, all_matched, max_error = compare_outcomes(expected, sim_output.probabilities)

    assert all_matched is False
    assert max_error >= 0.25  # Expected 0.8 vs 0.5 computed => error = 0.3

    # Verification check rule must mark FAIL
    prop = ProposalSpecification(
        title="Biased Bell Test",
        objective="Test discrepancy detection",
        num_qubits=2,
        circuit=spec,
        expected_outcomes=expected
    )
    chk, comps, tvd = check_outcome_distribution(prop, sim_output)
    assert chk.status.value == "FAIL"
    assert "Discrepancy detected" in chk.details


def test_case3_invalid_qubit_index():
    """
    Mandatory Case 3: Invalid qubit index (e.g. qubit 2 on 2-qubit circuit).
    Status: MUST REJECT WITH MEANINGFUL ERROR.
    """
    # Test via parser DSL
    with pytest.raises(CircuitParseError) as exc_info:
        parse_circuit_dsl("H 0\nCX 0 2", num_qubits=2)
    assert "qubit index 2" in str(exc_info.value)
    assert "2 qubits" in str(exc_info.value)

    # Test via Pydantic model validation
    with pytest.raises(ValueError) as exc_info2:
        CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 2])
            ]
        )
    assert "out of bounds" in str(exc_info2.value).lower()


def test_malformed_gate_syntax():
    """
    Test rejection of unknown gate names, invalid arity, or identical control-target.
    """
    # Unknown gate
    with pytest.raises(CircuitParseError):
        parse_circuit_dsl("FOO 0", num_qubits=1)

    # Identical control and target in CNOT
    with pytest.raises(CircuitParseError) as exc_cnot:
        parse_circuit_dsl("CX 0 0", num_qubits=2)
    assert "cannot be identical" in str(exc_cnot.value)

    # Missing rotation parameter in RZ
    with pytest.raises(ValueError):
        GateOperation(name="RZ", target_qubits=[0], params=None)


def test_ghz_three_qubit_circuit(simulator):
    """
    Test 3-qubit GHZ state generation (|000> + |111>)/sqrt(2).
    """
    dsl = "H 0\nCX 0 1\nCX 1 2"
    spec = parse_circuit_dsl(dsl, num_qubits=3)
    out = simulator.simulate(spec)

    assert out.is_normalized is True
    assert abs(out.probabilities["000"] - 0.5) < 1e-4
    assert abs(out.probabilities["111"] - 0.5) < 1e-4
    assert out.probabilities["001"] < 1e-4


def test_fidelity_and_tvd_metrics():
    """
    Test numerical properties of fidelity and Total Variation Distance.
    """
    # Identical states => Fidelity = 1.0
    sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    fid = compute_fidelity(sv, sv)
    assert abs(fid - 1.0) < 1e-6

    # Orthogonal states => Fidelity = 0.0
    sv_ortho = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
    fid_ortho = compute_fidelity(sv, sv_ortho)
    assert abs(fid_ortho - 0.0) < 1e-6

    # TVD
    p1 = {"00": 0.5, "11": 0.5}
    p2 = {"00": 0.8, "11": 0.2}
    tvd = total_variation_distance(p1, p2)
    assert abs(tvd - 0.3) < 1e-6


def test_toffoli_ccx_and_fredkin_cswap(simulator):
    """
    Test 3-qubit Toffoli (CCX) and Fredkin (CSWAP) gate parsing and simulation.
    X 0, X 1 -> |011> in little-endian (q0=1, q1=1, q2=0).
    CCX 0 1 2 -> flips q2 because q0=1 and q1=1 -> state becomes |111>.
    """
    dsl = "X 0\nX 1\nCCX 0 1 2"
    spec = parse_circuit_dsl(dsl, num_qubits=3)
    out = simulator.simulate(spec)
    assert out.is_normalized is True
    # in little endian, q2=1, q1=1, q0=1 is bitstring '111'
    assert abs(out.probabilities["111"] - 1.0) < 1e-5

    # Test NumPy fallback engine also matches
    sim_np = QuantumSimulator(force_numpy=True)
    out_np = sim_np.simulate(spec)
    assert out_np.is_normalized is True
    assert abs(out_np.probabilities["111"] - 1.0) < 1e-5

    # Test CSWAP: |001> (q0=1, q1=0, q2=0) -> CSWAP 0 1 2 -> swaps q1 and q2 because q0=1 -> |010>? No, q1 and q2 are both 0.
    # Set q0=1, q1=1, q2=0 -> bitstring '011' (q2=0, q1=1, q0=1).
    # CSWAP 0 1 2 -> swaps q1 and q2 -> q2=1, q1=0, q0=1 -> bitstring '101'
    dsl_swap = "X 0\nX 1\nCSWAP 0 1 2"
    spec_swap = parse_circuit_dsl(dsl_swap, num_qubits=3)
    out_swap = simulator.simulate(spec_swap)
    assert abs(out_swap.probabilities["101"] - 1.0) < 1e-5
