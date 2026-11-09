"""
End-to-end Integration Test for the Quantum Proposal Verification Workflow.
Validates the entire pipeline: document extraction, semantic retrieval,
Qiskit circuit simulation, verification engine checklist, and report generation.
"""

import os
import json
import pytest
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.verification import CheckStatus, VerificationReport
from src.document_processing.pdf_extractor import PDFExtractor
from src.document_processing.chunker import TraceableChunker
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever
from src.circuit.simulator import QuantumSimulator
from src.verification.engine import VerificationEngine
from src.reporting.generator import ReportGenerator


def test_end_to_end_pipeline_bell_state_pass():
    """Verify complete end-to-end execution on valid Bell state proposal."""
    # 1. Knowledge Base & Retriever
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)

    # 2. Proposal Specification
    proposal = ProposalSpecification(
        title="Verification of EPR Pair Bell State Generation",
        objective="Prepare (|00> + |11>)/sqrt(2) using H and CNOT gates.",
        hypothesis="Hadamard followed by CX transforms |00> into maximally entangled state.",
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
        ],
        assumptions=["Ideal unitary evolution"]
    )

    # 3. Verification Engine
    engine = VerificationEngine(retriever=retriever)
    report = engine.verify_proposal(proposal)

    # 4. Assertions on Audit Checklist
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 4
    assert "PASSED" in report.overall_summary

    # 5. Assertions on Qiskit Simulation
    assert report.simulation is not None
    assert report.simulation.num_qubits == 2
    assert report.simulation.is_normalized is True
    assert abs(report.simulation.probabilities["00"] - 0.5) < 1e-4
    assert abs(report.simulation.probabilities["11"] - 0.5) < 1e-4

    # 6. Assertions on Outcome Comparison
    assert len(report.outcome_comparisons) == 2
    assert all(oc.is_match for oc in report.outcome_comparisons)
    assert report.total_variation_distance is not None
    assert report.total_variation_distance < 0.05

    # 7. Assertions on Traceable Literature Citations
    assert len(report.citations) >= 1
    for cit in report.citations:
        assert cit.source_title
        assert cit.filename
        assert cit.chunk_id
        assert cit.passage
        assert cit.similarity_score >= 0.0

    # 8. Assertions on Report Generation
    json_export = ReportGenerator.to_json(report)
    md_export = ReportGenerator.to_markdown(report)

    parsed_json = json.loads(json_export)
    assert parsed_json["report_id"] == report.report_id
    assert "software_environment" in parsed_json
    assert "Quantum Experiment Proposal Verification Report" in md_export
    assert "NISQ" in md_export


def test_end_to_end_pipeline_biased_distribution_fail():
    """Verify that biased probability distribution strictly causes overall FAIL."""
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)

    proposal = ProposalSpecification(
        title="Biased Bell State Proposal",
        objective="Prepare Bell state but expect 0.8/0.2 probabilities.",
        hypothesis="Claims asymmetric probabilities for symmetric Bell state.",
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

    engine = VerificationEngine(retriever=retriever)
    report = engine.verify_proposal(proposal)

    # Must FAIL
    assert report.summary_counts["FAIL"] >= 1
    assert "FAILED" in report.overall_summary

    dist_chk = next(c for c in report.checks if c.check_id == "CHK-DIST-01")
    assert dist_chk.status == CheckStatus.FAIL
    assert "Discrepancy detected" in dist_chk.details


def test_end_to_end_pipeline_out_of_bounds_fail():
    """Verify that gate referencing unallocated qubit strictly causes overall FAIL."""
    # Proposal with 2 qubits but targeting qubit 2
    raw_data = {
        "title": "Out-of-Bounds Qubit Register Proposal",
        "objective": "Apply CX on qubit 2 on a 2-qubit device",
        "num_qubits": 2,
        "circuit": {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "target_qubits": [0]},
                {"name": "CX", "target_qubits": [0, 2]}
            ]
        }
    }
    proposal = ProposalSpecification(**raw_data)
    engine = VerificationEngine()
    report = engine.verify_proposal(proposal)

    assert report.summary_counts["FAIL"] >= 1
    assert "FAILED" in report.overall_summary
    circ_chk = next(c for c in report.checks if c.check_id == "CHK-CIRC-01")
    assert circ_chk.status == CheckStatus.FAIL
