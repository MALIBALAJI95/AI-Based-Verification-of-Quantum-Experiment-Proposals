"""
Automated Pytest Suite: Verification Report Generator.
Tests JSON and Markdown report export formatting, completeness, and audit integrity.
"""

import json
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.verification import VerificationReport
from src.verification.engine import VerificationEngine
from src.reporting.generator import ReportGenerator


def test_report_json_and_markdown_exports():
    engine = VerificationEngine()
    spec = ProposalSpecification(
        title="Test Bell Verification",
        objective="Verify state |Phi+>",
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

    # 1. JSON Export
    json_str = ReportGenerator.to_json(report)
    parsed = json.loads(json_str)
    assert parsed["report_id"] == report.report_id
    assert parsed["summary_counts"]["PASS"] >= 4
    assert "software_environment" in parsed

    # 2. Markdown Export
    md_str = ReportGenerator.to_markdown(report)
    assert "# Quantum Experiment Proposal Verification Report" in md_str
    assert f"**Report ID**: `{report.report_id}`" in md_str
    assert "Verification Checklist" in md_str
    assert "Simulated Statevector & Measurement Probabilities" in md_str
    assert "Limitations and Research Disclaimer" in md_str
    assert "NISQ" in md_str
