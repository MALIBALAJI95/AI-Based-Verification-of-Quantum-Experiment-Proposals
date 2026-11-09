"""
Independent Scientific Verification Rules.
Defines modular checks for structural validity, gate bounds, statevector normalization,
probability distribution matching, and literature grounding.
Each rule strictly returns PASS, FAIL, INCONCLUSIVE, or NOT_RUN.
"""

from typing import List, Tuple, Optional, Dict, Any
from ..schemas.proposal import ProposalSpecification, CircuitSpec
from ..schemas.verification import CheckStatus, CheckCategory, VerificationCheck, SimulationOutput, OutcomeComparison
from ..circuit.metrics import compare_outcomes, total_variation_distance
from ..circuit.parser import ALL_SUPPORTED_GATES


def check_qubit_count(proposal: ProposalSpecification) -> VerificationCheck:
    """Verifies that system size (number of qubits) is explicitly specified and within valid prototype bounds (1-10)."""
    if proposal.num_qubits is None:
        if proposal.circuit and proposal.circuit.num_qubits:
            proposal.num_qubits = proposal.circuit.num_qubits
        else:
            return VerificationCheck(
                check_id="CHK-STRUC-01",
                name="Qubit Count Specification",
                category=CheckCategory.STRUCTURAL,
                status=CheckStatus.INCONCLUSIVE,
                description="Verify that proposal specifies an explicit number of qubits.",
                details="Proposal omitted the number of qubits. Automated verification cannot proceed without system dimension.",
                error_message="Missing specification: 'num_qubits' is null."
            )

    n = proposal.num_qubits
    if n < 1 or n > 10:
        return VerificationCheck(
            check_id="CHK-STRUC-01",
            name="Qubit Count Specification",
            category=CheckCategory.STRUCTURAL,
            status=CheckStatus.FAIL,
            description="Verify that proposal specifies an explicit number of qubits within prototype limits (1-10).",
            details=f"Invalid qubit count {n}. Prototype supports 1-10 qubits.",
            error_message=f"Value {n} is out of bounds [1, 10]."
        )

    return VerificationCheck(
        check_id="CHK-STRUC-01",
        name="Qubit Count Specification",
        category=CheckCategory.STRUCTURAL,
        status=CheckStatus.PASS,
        details=f"Proposal targets an allocated {n}-qubit quantum register (Hilbert space dimension 2^{n} = {2**n}).",
        description="Verify that proposal specifies an explicit number of qubits."
    )


def check_circuit_syntax(proposal: ProposalSpecification) -> VerificationCheck:
    """Validates gate vocabulary, arity, qubit indices, and continuous parameter requirements."""
    if not proposal.circuit or not proposal.circuit.gates:
        # Check if an invalid circuit error was captured during extraction or validation
        circuit_errors = [
            err for err in (proposal.potential_inconsistencies + proposal.extraction_warnings)
            if any(term in err.lower() for term in ("circuit specification error:", "gate syntax error:", "out of bounds", "out-of-bounds", "invalid qubit index"))
        ]
        if circuit_errors:
            return VerificationCheck(
                check_id="CHK-CIRC-01",
                name="Circuit Gate Specification",
                category=CheckCategory.STRUCTURAL,
                status=CheckStatus.FAIL,
                description="Check that an executable gate sequence is provided with valid gate operations.",
                details=f"Detected critical circuit specification error: {circuit_errors[0]}",
                error_message="; ".join(circuit_errors)
            )

        return VerificationCheck(
            check_id="CHK-CIRC-01",
            name="Circuit Gate Specification",
            category=CheckCategory.STRUCTURAL,
            status=CheckStatus.INCONCLUSIVE,
            description="Check that an executable gate sequence is provided with valid gate operations.",
            details="No gate operations found in proposal specification.",
            error_message="Missing circuit or empty gate sequence."
        )

    num_qubits = proposal.num_qubits or proposal.circuit.num_qubits
    errors = []

    for idx, gate in enumerate(proposal.circuit.gates, start=1):
        gname = gate.name.strip().upper()
        if gname not in ALL_SUPPORTED_GATES:
            errors.append(f"Gate #{idx} '{gname}' is not in supported vocabulary: {sorted(list(ALL_SUPPORTED_GATES))}")

        for q in gate.target_qubits:
            if q < 0 or q >= num_qubits:
                errors.append(
                    f"Gate #{idx} ({gname}) targets qubit index {q}, which is invalid for a {num_qubits}-qubit circuit (valid: 0..{num_qubits - 1})."
                )

        if gname in ("CX", "CNOT", "CZ", "SWAP"):
            if len(gate.target_qubits) != 2:
                errors.append(f"Two-qubit gate #{idx} ({gname}) requires 2 qubits, got {len(gate.target_qubits)}.")
            elif gate.target_qubits[0] == gate.target_qubits[1]:
                errors.append(f"Two-qubit gate #{idx} ({gname}) has identical control and target qubit: {gate.target_qubits[0]}.")

        if gname in ("RX", "RY", "RZ") and (not gate.params or len(gate.params) < 1):
            errors.append(f"Parametric rotation gate #{idx} ({gname}) requires an angle parameter in radians.")

    if errors:
        return VerificationCheck(
            check_id="CHK-CIRC-01",
            name="Circuit Gate Specification",
            category=CheckCategory.STRUCTURAL,
            status=CheckStatus.FAIL,
            description="Check that an executable gate sequence is provided with valid gate operations.",
            details=f"Detected {len(errors)} circuit specification error(s).",
            error_message="; ".join(errors)
        )

    return VerificationCheck(
        check_id="CHK-CIRC-01",
        name="Circuit Gate Specification",
        category=CheckCategory.STRUCTURAL,
        status=CheckStatus.PASS,
        description="Check that an executable gate sequence is provided with valid gate operations.",
        details=f"All {len(proposal.circuit.gates)} gate operations validated successfully (indices 0..{num_qubits-1}, valid arity and parameters)."
    )


def check_statevector_normalization(sim_output: Optional[SimulationOutput]) -> VerificationCheck:
    """Verifies that the simulated quantum statevector satisfies unity normalization (sum |c_i|^2 = 1.0)."""
    if sim_output is None:
        return VerificationCheck(
            check_id="CHK-SIM-01",
            name="Statevector Normalization",
            category=CheckCategory.SIMULATION,
            status=CheckStatus.NOT_RUN,
            description="Verify that quantum unitary evolution preserves norm unity.",
            details="Simulation was not executed due to preceding structural errors or missing circuit."
        )

    if not sim_output.is_normalized:
        return VerificationCheck(
            check_id="CHK-SIM-01",
            name="Statevector Normalization",
            category=CheckCategory.SIMULATION,
            status=CheckStatus.FAIL,
            description="Verify that quantum unitary evolution preserves norm unity.",
            details=f"Statevector normalization failed with error {sim_output.normalization_error:.2e} > 1e-6.",
            error_message="Non-unitary transformation or numerical divergence."
        )

    return VerificationCheck(
        check_id="CHK-SIM-01",
        name="Statevector Normalization",
        category=CheckCategory.SIMULATION,
        status=CheckStatus.PASS,
        description="Verify that quantum unitary evolution preserves norm unity.",
        details=f"Statevector norm strictly preserved (error: {sim_output.normalization_error:.2e} <= 1e-6)."
    )


def check_outcome_distribution(
    proposal: ProposalSpecification,
    sim_output: Optional[SimulationOutput]
) -> Tuple[VerificationCheck, List[OutcomeComparison], Optional[float]]:
    """
    Compares author-hypothesized outcome probabilities with computed simulation probabilities.
    Returns (VerificationCheck, List[OutcomeComparison], total_variation_distance).
    """
    if not proposal.expected_outcomes:
        return (
            VerificationCheck(
                check_id="CHK-DIST-01",
                name="Measurement Outcome Distribution",
                category=CheckCategory.DISTRIBUTION_MATCH,
                status=CheckStatus.INCONCLUSIVE,
                description="Verify computed measurement probabilities against author-hypothesized outcomes.",
                details="Proposal did not specify quantitative expected measurement probabilities.",
                error_message="Missing specification: 'expected_outcomes' is empty."
            ),
            [],
            None
        )

    if sim_output is None:
        return (
            VerificationCheck(
                check_id="CHK-DIST-01",
                name="Measurement Outcome Distribution",
                category=CheckCategory.DISTRIBUTION_MATCH,
                status=CheckStatus.NOT_RUN,
                description="Verify computed measurement probabilities against author-hypothesized outcomes.",
                details="Simulation could not run; outcome verification skipped."
            ),
            [],
            None
        )

    # Perform rigorous probability comparison
    comparisons, all_matched, max_error = compare_outcomes(
        proposal.expected_outcomes,
        sim_output.probabilities
    )

    # Compute Total Variation Distance if possible
    expected_dict = {o.basis_state: o.expected_probability for o in proposal.expected_outcomes}
    tvd = total_variation_distance(expected_dict, sim_output.probabilities)

    if not all_matched:
        mismatched_details = [
            f"State |{c.basis_state}>: expected {c.expected_probability}, computed {c.computed_probability} (diff {c.absolute_difference} > tol {c.tolerance})"
            for c in comparisons if not c.is_match
        ]
        return (
            VerificationCheck(
                check_id="CHK-DIST-01",
                name="Measurement Outcome Distribution",
                category=CheckCategory.DISTRIBUTION_MATCH,
                status=CheckStatus.FAIL,
                description="Verify computed measurement probabilities against author-hypothesized outcomes.",
                details=f"Discrepancy detected in measurement probabilities. Max error: {max_error:.4f}, TVD: {tvd:.4f}. Details: {'; '.join(mismatched_details)}",
                error_message=f"Hypothesized outcomes contradict quantum mechanics statevector simulation: {'; '.join(mismatched_details)}"
            ),
            comparisons,
            round(tvd, 4)
        )

    return (
        VerificationCheck(
            check_id="CHK-DIST-01",
            name="Measurement Outcome Distribution",
            category=CheckCategory.DISTRIBUTION_MATCH,
            status=CheckStatus.PASS,
            description="Verify computed measurement probabilities against author-hypothesized outcomes.",
            details=f"All {len(comparisons)} specified basis outcome probabilities match simulation within tolerance (Max error: {max_error:.4f}, TVD: {tvd:.4f})."
        ),
        comparisons,
        round(tvd, 4)
    )


def check_physical_hardware_feasibility(proposal: ProposalSpecification) -> VerificationCheck:
    """
    Evaluates physical NISQ implementation feasibility and warns about decoherence and noise.
    Ideal simulation never certifies real physical hardware execution.
    """
    depth = len(proposal.circuit.gates) if (proposal.circuit and proposal.circuit.gates) else 0
    num_q = proposal.num_qubits or 0

    return VerificationCheck(
        check_id="CHK-PHYS-01",
        name="Physical Hardware Implementation Disclaimer",
        category=CheckCategory.PHYSICAL_FEASIBILITY,
        status=CheckStatus.PASS,
        description="Evaluate circuit depth and provide physical noise and decoherence assessment.",
        details=(
            f"Circuit depth is {depth} gates over {num_q} qubits. "
            "IMPORTANT SCIENTIFIC NOTICE: Computational statevector simulation models ideal unitary evolution. "
            "Real NISQ quantum processors suffer from finite coherence times (T1, T2*), gate infidelities (~0.1-1.0% per 2-qubit gate), "
            "readout error, and cross-talk. Execution on physical hardware requires pulse calibration, dynamical decoupling, and error mitigation."
        )
    )
