"""
Tests for Two Proposal Input Modes (PDF & DOCX Upload vs Manual Structured Input)
and the Common Confirmation Workflow.
Validates extraction of all required fields, non-hallucination of missing operations,
schema compatibility across both modes, and researcher confirmation requirements.
"""

import os
import io
import pytest
from src.document_processing.document_extractor import DocumentExtractor, ExtractedDocument
from src.document_processing.docx_extractor import DOCXExtractor
from src.document_processing.pdf_extractor import PDFExtractor
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.verification import CheckStatus
from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import ProposalExtractor
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever
from src.verification.engine import VerificationEngine
from src.circuit.simulator import QuantumSimulator


@pytest.fixture
def doc_extractor():
    return DocumentExtractor()


@pytest.fixture
def proposal_extractor():
    client = ConfigurableLLMClient(demo_mode=True)
    return ProposalExtractor(client)


@pytest.fixture
def verification_engine():
    client = ConfigurableLLMClient(demo_mode=True)
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    return VerificationEngine(retriever=retriever, llm_client=client)


# -----------------------------------------------------------------------------
# 1. Test PDF Upload & Extraction
# -----------------------------------------------------------------------------
def test_pdf_upload_extraction(doc_extractor, proposal_extractor):
    pdf_path = "data/sample_proposals/sample_bell_state_proposal.pdf"
    assert os.path.exists(pdf_path), "Sample PDF should exist in data/sample_proposals/"

    doc = doc_extractor.extract_from_file(pdf_path)
    assert isinstance(doc, ExtractedDocument)
    assert len(doc.pages) >= 1
    assert doc.filename == "sample_bell_state_proposal.pdf"
    assert "objective" in doc.sections
    assert "expected_results" in doc.sections

    spec = proposal_extractor.extract(
        raw_text=doc.full_text,
        document_title=doc.sections.get("title", doc.filename),
        extracted_doc=doc,
        input_mode="upload"
    )

    assert isinstance(spec, ProposalSpecification)
    assert spec.input_mode == "upload"
    assert spec.source_filename == "sample_bell_state_proposal.pdf"
    assert len(spec.source_pages) >= 1
    assert spec.num_qubits == 2
    assert spec.circuit is not None
    assert len(spec.circuit.gates) == 2
    assert len(spec.expected_outcomes) == 2


# -----------------------------------------------------------------------------
# 2. Test DOCX Upload & Extraction
# -----------------------------------------------------------------------------
def test_docx_upload_extraction(doc_extractor, proposal_extractor):
    docx_path = "data/sample_proposals/sample_bell_state_proposal.docx"
    assert os.path.exists(docx_path), "Sample DOCX should exist in data/sample_proposals/"

    doc = doc_extractor.extract_from_file(docx_path)
    assert isinstance(doc, ExtractedDocument)
    assert len(doc.pages) >= 2  # Has page break between page 1 and page 2
    assert doc.filename == "sample_bell_state_proposal.docx"
    assert "methodology" in doc.sections
    assert "quantum_system" in doc.sections

    spec = proposal_extractor.extract(
        raw_text=doc.full_text,
        document_title=doc.sections.get("title", doc.filename),
        extracted_doc=doc,
        input_mode="upload"
    )

    assert isinstance(spec, ProposalSpecification)
    assert spec.input_mode == "upload"
    assert spec.source_filename == "sample_bell_state_proposal.docx"
    assert len(spec.source_pages) == 2
    assert spec.quantum_system is not None
    assert spec.initial_state is not None
    assert spec.measurement_basis is not None
    assert spec.circuit is not None
    assert len(spec.circuit.gates) == 2

    # Verify source page references in author claims
    for claim in spec.author_claims:
        assert claim.source_reference is not None
        assert "Page" in claim.source_reference


# -----------------------------------------------------------------------------
# 3. Test Manual Structured Input
# -----------------------------------------------------------------------------
def test_manual_structured_input_mode(proposal_extractor, verification_engine):
    manual_spec = ProposalSpecification(
        title="Three-Qubit GHZ State Preparation",
        objective="Prepare genuine tripartite entanglement (|000> + |111>)/sqrt(2).",
        hypothesis="H(0) cascaded with CNOT(0,1) and CNOT(1,2) produces GHZ state.",
        methodology="Hadamard on qubit 0, CX(0,1), CX(1,2).",
        quantum_system="Trapped Ion",
        num_qubits=3,
        initial_state="|000>",
        measurement_basis="computational (Z)",
        circuit=CircuitSpec(
            num_qubits=3,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1]),
                GateOperation(name="CX", target_qubits=[1, 2])
            ],
            initial_state="|000>",
            measurement_basis="computational (Z)"
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="000", expected_probability=0.5, tolerance=0.05),
            ExpectedOutcome(basis_state="111", expected_probability=0.5, tolerance=0.05)
        ],
        assumptions=["3-qubit coherent register"],
        input_mode="manual",
        confirmed_by_researcher=True
    )

    assert manual_spec.input_mode == "manual"
    assert manual_spec.num_qubits == 3
    assert len(manual_spec.circuit.gates) == 3
    assert manual_spec.confirmed_by_researcher is True

    # Test verification on manually specified proposal
    report = verification_engine.verify_proposal(manual_spec)
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 4
    assert "PASSED" in report.overall_summary


# -----------------------------------------------------------------------------
# 4. Test Compatibility Across Both Input Modes
# -----------------------------------------------------------------------------
def test_input_mode_schema_compatibility(doc_extractor, proposal_extractor, verification_engine):
    """Ensure both upload and manual modes produce identical, compatible ProposalSpecification schemas."""
    # Upload mode
    doc = doc_extractor.extract_from_file("data/sample_proposals/sample_bell_state_proposal.docx")
    upload_spec = proposal_extractor.extract(doc.full_text, doc.filename, extracted_doc=doc, input_mode="upload")

    # Manual mode
    manual_spec = ProposalSpecification(
        title=upload_spec.title,
        objective=upload_spec.objective,
        hypothesis=upload_spec.hypothesis,
        methodology=upload_spec.methodology,
        quantum_system=upload_spec.quantum_system,
        num_qubits=upload_spec.num_qubits,
        initial_state=upload_spec.initial_state,
        measurement_basis=upload_spec.measurement_basis,
        circuit=upload_spec.circuit,
        expected_outcomes=upload_spec.expected_outcomes,
        assumptions=upload_spec.assumptions,
        input_mode="manual"
    )

    # Both must share the same schema fields
    assert set(upload_spec.model_dump().keys()) == set(manual_spec.model_dump().keys())

    # Both must verify cleanly through verification engine
    report_upload = verification_engine.verify_proposal(upload_spec)
    report_manual = verification_engine.verify_proposal(manual_spec)

    assert report_upload.summary_counts["PASS"] == report_manual.summary_counts["PASS"]
    assert report_upload.summary_counts["FAIL"] == report_manual.summary_counts["FAIL"]


# -----------------------------------------------------------------------------
# 5. Test Incomplete & Ambiguous Proposals (No Invention, INCONCLUSIVE)
# -----------------------------------------------------------------------------
def test_incomplete_proposal_non_hallucination(doc_extractor, proposal_extractor, verification_engine):
    """
    Incomplete proposals must NOT invent missing gates or expected probabilities,
    and must return INCONCLUSIVE verification status.
    """
    docx_path = "data/sample_proposals/sample_incomplete_proposal.docx"
    doc = doc_extractor.extract_from_file(docx_path)
    spec = proposal_extractor.extract(doc.full_text, doc.filename, extracted_doc=doc, input_mode="upload")

    # Verify nothing was hallucinated
    assert spec.circuit is None
    assert len(spec.expected_outcomes) == 0
    assert len(spec.missing_information) >= 2
    assert any("qubit" in m.lower() for m in spec.missing_information)

    # Verify engine output
    report = verification_engine.verify_proposal(spec)
    assert report.summary_counts["INCONCLUSIVE"] >= 2
    assert "INCONCLUSIVE" in report.overall_summary
    assert report.simulation is None  # Simulation must not run


# -----------------------------------------------------------------------------
# 6. Test Editable Confirmation Workflow
# -----------------------------------------------------------------------------
def test_researcher_confirmation_and_editing_flow(proposal_extractor, verification_engine):
    """Test that modifying an extracted specification before confirmation updates simulation & verification."""
    raw_dsl = "H 0\nCX 0 1"
    initial_spec = ProposalSpecification(
        title="Unconfirmed Draft Proposal",
        objective="Prepare Bell state.",
        num_qubits=2,
        circuit=None,  # Missing initially
        expected_outcomes=[],  # Missing initially
        missing_information=["Gate sequence missing", "Expected outcomes missing"],
        confirmed_by_researcher=False
    )

    assert initial_spec.confirmed_by_researcher is False

    # Simulate researcher filling in the missing details in the confirmation screen
    from src.circuit.parser import parse_circuit_dsl
    confirmed_circuit = parse_circuit_dsl(raw_dsl, num_qubits=2)
    confirmed_outcomes = [
        ExpectedOutcome(basis_state="00", expected_probability=0.5),
        ExpectedOutcome(basis_state="11", expected_probability=0.5)
    ]

    confirmed_spec = initial_spec.model_copy(update={
        "circuit": confirmed_circuit,
        "expected_outcomes": confirmed_outcomes,
        "missing_information": [],
        "confirmed_by_researcher": True
    })

    assert confirmed_spec.confirmed_by_researcher is True
    assert confirmed_spec.circuit is not None
    assert len(confirmed_spec.expected_outcomes) == 2

    # Now verify confirmed specification
    report = verification_engine.verify_proposal(confirmed_spec)
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 4
    assert report.simulation is not None
    assert abs(report.simulation.probabilities["00"] - 0.5) < 1e-4
