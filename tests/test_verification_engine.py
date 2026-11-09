"""
Automated Pytest Suite: Modular Verification Engine.
Tests Case 4 (Incomplete proposal INCONCLUSIVE), computational precedence rule,
and audit report generation.
"""

import pytest
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.verification import CheckStatus
from src.verification.engine import VerificationEngine
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever
from src.llm.client import ConfigurableLLMClient


@pytest.fixture
def engine():
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    llm = ConfigurableLLMClient(demo_mode=True)
    return VerificationEngine(retriever=retriever, llm_client=llm)


def test_case4_incomplete_proposal(engine):
    """
    Mandatory Case 4: An incomplete proposal.
    The system must report missing information rather than inventing details.
    Status: MUST RETURN INCONCLUSIVE.
    """
    incomplete_spec = ProposalSpecification(
        title="Incomplete Quantum Algorithm Proposal",
        objective="Propose a quantum optimization method without parameters.",
        hypothesis="Superposition speeds up search.",
        num_qubits=None,
        circuit=None,
        expected_outcomes=[],
        missing_information=[
            "Proposal omitted the number of qubits.",
            "Proposal provides no gate sequence or circuit specification."
        ]
    )

    report = engine.verify_proposal(incomplete_spec)

    assert report.summary_counts["INCONCLUSIVE"] >= 1
    assert "INCONCLUSIVE" in report.overall_summary
    assert len(report.missing_information) >= 2
    assert report.simulation is None  # Never silently executed or hallucinated


def test_computational_precedence_rule(engine):
    """
    Rule: A computational check failure CANNOT be overridden by LLM or literature arguments.
    Even if literature supports the concept, a discrepancy in expected probabilities results in FAIL.
    """
    biased_spec = ProposalSpecification(
        title="Bell State with Discrepancy",
        objective="Prepare Bell state with biased probabilities",
        num_qubits=2,
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1])
            ]
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.8, tolerance=0.05),
            ExpectedOutcome(basis_state="11", expected_probability=0.2, tolerance=0.05)
        ]
    )

    report = engine.verify_proposal(biased_spec)

    # Must contain at least 1 failing check
    assert report.summary_counts["FAIL"] >= 1
    assert "FAILED" in report.overall_summary

    # Find the distribution check
    dist_chk = next(c for c in report.checks if c.check_id == "CHK-DIST-01")
    assert dist_chk.status == CheckStatus.FAIL

    # In claims analysis, computational status must flag contradiction
    outcome_claim = next((c for c in report.claims_analysis if "outcome" in c.claim_category.lower()), None)
    if outcome_claim:
        assert "Contradicted" in outcome_claim.computational_status


def test_hardware_feasibility_disclaimer_presence(engine):
    """
    Verify that report includes scientific disclaimers distinguishing ideal simulation
    from noisy physical quantum hardware.
    """
    spec = ProposalSpecification(
        title="Valid Bell State",
        objective="Demonstrate entanglement",
        num_qubits=2,
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1])
            ]
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.5, tolerance=0.05),
            ExpectedOutcome(basis_state="11", expected_probability=0.5, tolerance=0.05)
        ]
    )

    report = engine.verify_proposal(spec)
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 4
    assert "NISQ" in report.limitations_and_disclaimer
    assert "ideal" in report.limitations_and_disclaimer.lower()
