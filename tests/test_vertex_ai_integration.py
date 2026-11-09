"""
Comprehensive Integration and Unit Tests for Google Cloud Vertex AI Integration.
Tests:
1. Service Account JSON credential detection, Application Default Credentials (ADC), and environment loading
2. Project ID, location (global), model (gemini-3.8-flash), and service account email parsing without key exposure
3. Live / sandboxed Vertex AI API connectivity probe handling (billing, quota, and authentication diagnostics)
4. PDF processing of 's42256-025-01153-0.pdf' into ExtractedDocument with page provenance
5. Structured Pydantic validation into ResearchPaperAnalysis and ProposalSpecification (no 42257 qubit hallucination)
6. Independent Qiskit simulation verification precedence over Vertex AI model output
7. Initial AI-backed verification report generation
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from src.llm.client import ConfigurableLLMClient, LLMConfigurationError
from src.llm.extractor import ProposalExtractor, extract_doi, extract_strict_num_qubits
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.research_paper import ResearchPaperAnalysis, QuantumExperimentRecord
from src.schemas.verification import CheckStatus
from src.document_processing.document_extractor import DocumentExtractor, ExtractedDocument
from src.circuit.simulator import QuantumSimulator
from src.verification.engine import VerificationEngine
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever


SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", r"C:\GoogleCloudCredentials\vertex-service-account.json")
TEST_PDF_PATH = os.path.join("data", "sample_proposals", "s42256-025-01153-0.pdf")


# -----------------------------------------------------------------------------
# 1. Credentials, Configuration & Security Tests
# -----------------------------------------------------------------------------
def test_vertex_ai_credentials_and_adc_detection():
    """Verify service-account JSON is detected, parsed safely, and configured for ADC."""
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        pytest.skip(f"Vertex AI service-account credential file not found at {SERVICE_ACCOUNT_PATH}")
    
    with open(SERVICE_ACCOUNT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Must contain essential Google Service Account fields
    assert data.get("type") == "service_account"
    assert data.get("project_id") == "acquired-winter-511114-c2"
    assert "api-extraction@acquired-winter-511114-c2.iam.gserviceaccount.com" in data.get("client_email", "")
    assert "private_key" in data

    # Client must initialize with Vertex AI provider
    client = ConfigurableLLMClient(demo_mode=False)
    assert client.provider_name == "vertex"
    assert client.model_name == "gemini-3.8-flash"
    assert client.vertex_project == "acquired-winter-511114-c2"
    assert client.vertex_location == "global"
    assert client.service_account_email == "api-extraction@acquired-winter-511114-c2.iam.gserviceaccount.com"
    assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == SERVICE_ACCOUNT_PATH


def test_vertex_ai_security_never_exposes_private_keys():
    """Verify that is_configured, status strings, and connectivity outputs NEVER leak private keys or tokens."""
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        pytest.skip(f"Vertex AI service-account credential file not found at {SERVICE_ACCOUNT_PATH}")
    client = ConfigurableLLMClient(demo_mode=False)
    is_ready, status_desc = client.is_configured()
    
    assert is_ready is True
    # Must not contain private key markers
    assert "-----BEGIN PRIVATE KEY-----" not in status_desc
    assert "private_key" not in status_desc

    diag = client.test_connectivity()
    diag_str = json.dumps(diag)
    assert "-----BEGIN PRIVATE KEY-----" not in diag_str
    assert "BEGIN RSA PRIVATE KEY" not in diag_str


def test_vertex_ai_connectivity_probe_graceful_diagnostics():
    """Verify connectivity probe returns structured diagnostic dictionary and handles sandbox/billing/quota gracefully."""
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        pytest.skip(f"Vertex AI service-account credential file not found at {SERVICE_ACCOUNT_PATH}")
    client = ConfigurableLLMClient(demo_mode=False)
    diag = client.test_connectivity()
    
    assert isinstance(diag, dict)
    assert diag["provider"] == "Google Cloud Vertex AI"
    assert diag["model"] == "gemini-3.8-flash"
    assert diag["project_id"] == "acquired-winter-511114-c2"
    assert diag["location"] == "global"
    assert "service_account" in diag
    assert "latency_ms" in diag
    assert "status" in diag

    # In sandbox or if billing is required, diagnostic message must be human-readable and actionable
    status = diag["status"]
    assert any(term in status for term in ["Connected", "Billing", "disabled", "resolution", "quota", "Verified"])


# -----------------------------------------------------------------------------
# 2. Document Processing & Anti-Hallucination on s42256-025-01153-0.pdf
# -----------------------------------------------------------------------------
def test_nature_paper_pdf_parsing_and_provenance():
    """Verify PyMuPDF extracts full text, sections, and page references from s42256-025-01153-0.pdf."""
    assert os.path.exists(TEST_PDF_PATH), f"Target paper {TEST_PDF_PATH} must exist"
    
    with open(TEST_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    doc_extractor = DocumentExtractor()
    extracted_doc = doc_extractor.extract_from_bytes(pdf_bytes, filename="s42256-025-01153-0.pdf")

    assert isinstance(extracted_doc, ExtractedDocument)
    assert extracted_doc.filename == "s42256-025-01153-0.pdf"
    assert len(extracted_doc.pages) >= 5
    assert len(extracted_doc.full_text) > 10000
    assert "abstract" in extracted_doc.sections or "methodology" in extracted_doc.sections

    # Verify DOI extraction specifically parses 10.1038/s42256-025-01153-0
    doi = extract_doi(extracted_doc.full_text, filename="s42256-025-01153-0.pdf")
    assert doi is not None
    assert "s42256" in doi


def test_qubit_count_extraction_does_not_hallucinate_42257():
    """Verify strict semantic extraction guarantees num_qubits is NOT 42257 or DOI-derived."""
    with open(TEST_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    
    doc = DocumentExtractor().extract_from_bytes(pdf_bytes, filename="s42256-025-01153-0.pdf")
    
    # Check strict qubit extractor
    extracted_q = extract_strict_num_qubits(doc.full_text)
    if extracted_q is not None:
        assert extracted_q <= 10
        assert extracted_q != 42257

    # Run proposal extractor
    client = ConfigurableLLMClient(demo_mode=False)
    extractor = ProposalExtractor(client)
    spec = extractor.extract(doc.full_text, document_title="Nature Article s42256", extracted_doc=doc)

    assert isinstance(spec, ProposalSpecification)
    assert spec.num_qubits != 42257
    if spec.num_qubits is not None:
        assert 1 <= spec.num_qubits <= 10

    # Structured paper extractor
    paper = extractor.extract_research_paper(doc.full_text, document_title="Nature Article s42256", extracted_doc=doc)
    assert isinstance(paper, ResearchPaperAnalysis)
    assert paper.document_type == "research_paper"
    assert paper.title is not None
    assert len(paper.experiments) >= 1
    for exp in paper.experiments:
        if exp.num_qubits is not None:
            assert 1 <= exp.num_qubits <= 10
            assert exp.num_qubits != 42257


# -----------------------------------------------------------------------------
# 3. Structured JSON Validation & Vertex AI Fallback Compatibility
# -----------------------------------------------------------------------------
def test_structured_vertex_extraction_pydantic_validation():
    """Verify that structured JSON response from Vertex AI Gemini parses into validated Pydantic model."""
    mock_vertex_json = json.dumps({
        "title": "Quantum Algorithmic Architectures on Superconducting Processors",
        "objective": "Verify two-qubit entanglement fidelity and GHZ state scaling.",
        "hypothesis": "H and CNOT create maximal Bell entanglement with P(00)=0.5 and P(11)=0.5.",
        "methodology": "Implement quantum circuits in Qiskit and simulate measurement distributions.",
        "quantum_system": "Superconducting Transmon",
        "num_qubits": 2,
        "initial_state": "|00>",
        "measurement_basis": "computational (Z)",
        "circuit": {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "target_qubits": [0]},
                {"name": "CX", "target_qubits": [0, 1]}
            ]
        },
        "expected_outcomes": [
            {"basis_state": "00", "expected_probability": 0.5},
            {"basis_state": "11", "expected_probability": 0.5}
        ],
        "assumptions": ["Ideal state preparation", "Zero thermal excitation"],
        "missing_information": [],
        "potential_inconsistencies": []
    })

    client = ConfigurableLLMClient(demo_mode=False)
    client._provider = "vertex"
    client._vertex_project = "acquired-winter-511114-c2"
    with patch.object(client, "_call_vertex", return_value=mock_vertex_json), \
         patch.object(client, "is_configured", return_value=(True, "Vertex AI Configured")):
        extractor = ProposalExtractor(client)
        spec = extractor.extract("Mock proposal text for Vertex AI test.")

        assert spec.title == "Quantum Algorithmic Architectures on Superconducting Processors"
        assert spec.num_qubits == 2
        assert len(spec.circuit.gates) == 2
        assert len(spec.expected_outcomes) == 2
        assert spec.expected_outcomes[0].basis_state == "00"
        assert spec.expected_outcomes[0].expected_probability == 0.5


# -----------------------------------------------------------------------------
# 4. Computational Simulation Precedence & Initial Verification Report
# -----------------------------------------------------------------------------
def test_qiskit_computational_precedence_over_vertex_ai():
    """
    CRITICAL SCIENTIFIC REQUIREMENT:
    Computational Qiskit statevector simulation must take absolute precedence over
    any claims made by the LLM or proposal author.
    """
    # Create proposal with false claim (P(01)=1.0 for a Bell state)
    spec = ProposalSpecification(
        title="Flawed Claim Bell State Test",
        objective="Prepare Bell state with impossible outcome claims.",
        num_qubits=2,
        initial_state="|00>",
        measurement_basis="computational (Z)",
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1])
            ]
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="01", expected_probability=1.0)  # Mathematically FALSE
        ],
        confirmed_by_researcher=True
    )

    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=False)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    report = engine.verify_proposal(spec)

    # Simulation MUST report failure on probability distribution check regardless of LLM
    dist_check = next((c for c in report.checks if c.check_id == "CHK-DIST-01"), None)
    assert dist_check is not None
    assert dist_check.status == CheckStatus.FAIL

    # Overall verdict summary counts must have at least one FAIL
    assert report.summary_counts["FAIL"] >= 1


def test_initial_verification_report_generation():
    """Verify that an initial evidence-backed verification report is generated with citations and summary."""
    spec = ProposalSpecification(
        title="Two-Qubit Bell State Verification",
        objective="Prepare (|00> + |11>)/sqrt(2) state using H and CNOT gates.",
        hypothesis="H and CX gates produce an entangled state with equal superposition.",
        quantum_system="Superconducting Transmon",
        num_qubits=2,
        initial_state="|00>",
        measurement_basis="computational (Z)",
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1])
            ]
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.5),
            ExpectedOutcome(basis_state="11", expected_probability=0.5)
        ],
        assumptions=["T1, T2 >> gate execution times"],
        confirmed_by_researcher=True
    )

    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=False)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    report = engine.verify_proposal(spec)

    assert report.proposal_title == "Two-Qubit Bell State Verification"
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 1
    assert report.simulation is not None
    assert report.simulation.num_qubits == 2
    assert len(report.citations) >= 1
    assert len(report.checks) >= 6
    assert report.overall_summary is not None
