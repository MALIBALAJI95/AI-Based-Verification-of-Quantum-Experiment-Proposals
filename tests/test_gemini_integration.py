"""
Comprehensive Integration and Unit Tests for Google Gemini 3.8 Flash Integration.
Tests:
1. Actual Gemini API connectivity & model availability
2. Structured quantum proposal extraction
3. Invalid and missing API credentials handling
4. Malformed model responses handling
5. RAG context integration & citation preservation
6. Independent quantum verification precedence (computational checks cannot be overridden by LLM)
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from src.llm.client import ConfigurableLLMClient, LLMConfigurationError
from src.llm.extractor import ProposalExtractor
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.verification import CheckStatus
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever
from src.circuit.simulator import QuantumSimulator
from src.verification.engine import VerificationEngine


# -----------------------------------------------------------------------------
# 1. Real API Connectivity & Model Availability Tests
# -----------------------------------------------------------------------------
def test_real_gemini_model_availability():
    """Verify that model 'gemini-3.8-flash' is retrieved from Google's official model catalog."""
    client = ConfigurableLLMClient(demo_mode=False)
    is_ready, _ = client.is_configured()
    if not is_ready or client.provider_name != "gemini":
        pytest.skip("GEMINI_API_KEY not configured in environment for live test.")

    avail, msg, details = client.verify_model_availability("gemini-3.8-flash")
    if not avail and ("getaddrinfo failed" in msg or "ConnectError" in msg):
        pytest.skip("Offline / sandboxed environment without external DNS resolution.")

    assert avail is True, f"Expected gemini-3.8-flash to be available in catalog: {msg}"
    assert "gemini-3.8-flash" in details.get("name", "")
    assert "generateContent" in details.get("supported_actions", [])


def test_real_gemini_connectivity_test():
    """Verify actual Gemini API connectivity test probe returns genuine diagnostic response."""
    client = ConfigurableLLMClient(demo_mode=False)
    is_ready, _ = client.is_configured()
    if not is_ready or client.provider_name != "gemini":
        pytest.skip("GEMINI_API_KEY not configured in environment for live test.")

    diag = client.test_connectivity()
    if not diag.get("success") and ("getaddrinfo failed" in diag.get("status", "") or "ConnectError" in diag.get("status", "")):
        pytest.skip("Offline / sandboxed environment without external DNS resolution.")

    assert diag["success"] is True
    assert diag["provider"] == "Google Gemini"
    assert diag["model"] == "gemini-3.8-flash"
    assert diag["latency_ms"] > 0
    assert "Verified" in diag["status"]
    # Verify API key is NEVER exposed in diagnostics
    diag_str = json.dumps(diag)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        assert api_key not in diag_str


# -----------------------------------------------------------------------------
# 2. Security & Credentials Error Handling
# -----------------------------------------------------------------------------
def test_missing_api_key_handling(monkeypatch):
    """When GEMINI_API_KEY is missing, application must report clear status and not fabricate output."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    client = ConfigurableLLMClient(demo_mode=False)
    is_ready, desc = client.is_configured()
    assert is_ready is False
    assert "GEMINI_API_KEY" in desc
    assert "DEMO MODE" in desc

    with pytest.raises(LLMConfigurationError):
        client.complete("System", "User Proposal")


def test_invalid_api_credentials_handling(monkeypatch):
    """Invalid credentials must return an error status and never crash or expose the secret."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "INVALID_MOCK_KEY_12345")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    client = ConfigurableLLMClient(demo_mode=False)
    assert client.provider_name == "gemini"
    assert "INVALID_MOCK_KEY_12345" not in client.is_configured()[1]  # Must be masked

    diag = client.test_connectivity()
    assert diag["success"] is False
    assert "INVALID_MOCK_KEY_12345" not in json.dumps(diag)


# -----------------------------------------------------------------------------
# 3. Unit Tests with Mocked Gemini Responses
# -----------------------------------------------------------------------------
def test_structured_quantum_proposal_extraction_mocked():
    """Test structured Pydantic extraction from simulated Gemini 3.8 Flash response."""
    mock_gemini_json = json.dumps({
        "title": "Two-Qubit Bell State Preparation",
        "objective": "Prepare (|00> + |11>)/sqrt(2) state.",
        "hypothesis": "H on qubit 0 and CX on (0,1) generates maximal entanglement.",
        "methodology": "Initialize in |00>, apply H(0), CX(0,1), measure in Z basis.",
        "quantum_system": "Superconducting Transmon",
        "num_qubits": 2,
        "initial_state": "|00>",
        "measurement_basis": "computational (Z)",
        "circuit": {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "target_qubits": [0]},
                {"name": "CX", "target_qubits": [0, 1]}
            ],
            "initial_state": "|00>",
            "measurement_basis": "computational (Z)"
        },
        "expected_outcomes": [
            {"basis_state": "00", "expected_probability": 0.5, "tolerance": 0.05},
            {"basis_state": "11", "expected_probability": 0.5, "tolerance": 0.05}
        ],
        "assumptions": ["Ideal gate fidelities"],
        "constraints": [],
        "missing_information": [],
        "potential_inconsistencies": [],
        "author_claims": [
            {"claim_id": "CLM-01", "claim_type": "circuit", "statement": "H(0) then CX(0,1) creates Bell state.", "source_reference": "Page 1"}
        ]
    })

    mock_client = MagicMock(spec=ConfigurableLLMClient)
    mock_client.complete.return_value = mock_gemini_json

    extractor = ProposalExtractor(mock_client)
    spec = extractor.extract("Raw proposal text", document_title="Bell Experiment")

    assert spec.title == "Two-Qubit Bell State Preparation"
    assert spec.num_qubits == 2
    assert spec.initial_state == "|00>"
    assert len(spec.circuit.gates) == 2
    assert spec.circuit.gates[0].name == "H"
    assert spec.circuit.gates[1].name == "CX"
    assert len(spec.expected_outcomes) == 2
    assert spec.expected_outcomes[0].basis_state == "00"
    assert spec.expected_outcomes[0].expected_probability == 0.5


def test_malformed_model_response_handling():
    """When LLM returns malformed JSON or invalid format, extractor must fall back gracefully."""
    mock_client = MagicMock(spec=ConfigurableLLMClient)
    # Malformed non-JSON output
    mock_client.complete.return_value = "This is a conversational reply without any valid JSON: ``` {broken "

    extractor = ProposalExtractor(mock_client)
    raw_text = "Proposal: Bell state generation using H 0 and CX 0 1 on 2 qubits."
    spec = extractor.extract(raw_text, document_title="Fallback Test")

    # Fallback extractor must salvage title and gates without crashing
    assert spec is not None
    assert isinstance(spec, ProposalSpecification)
    assert spec.title == "Fallback Test"
    assert spec.num_qubits == 2


# -----------------------------------------------------------------------------
# 4. RAG Integration & Citation Preservation
# -----------------------------------------------------------------------------
def test_rag_context_integration_and_citation_preservation():
    """Verify that literature retrieval supplies authentic citations with exact source tracking."""
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)

    proposal = ProposalSpecification(
        title="EPR Bell State Generation",
        objective="Prepare maximally entangled Bell state.",
        hypothesis="Hadamard and CNOT gates prepare EPR state.",
        num_qubits=2
    )

    citations = retriever.retrieve_for_proposal(proposal)
    assert len(citations) >= 1
    for cit in citations:
        assert cit.citation_id.startswith("CIT-")
        assert cit.source_title is not None
        assert cit.filename is not None
        assert cit.page_number >= 1
        assert len(cit.passage) > 20
        assert 0.0 <= cit.similarity_score <= 1.0


# -----------------------------------------------------------------------------
# 5. Independent Quantum Verification Precedence
# -----------------------------------------------------------------------------
def test_computational_precedence_over_gemini():
    """
    Core Scientific Principle:
    Even if LLM or literature claims success, computational discrepancy in Qiskit simulation strictly yields FAIL.
    """
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=True)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    # Deliberately incorrect probability: expects 80% 00 and 20% 11 on ideal Bell state
    biased_proposal = ProposalSpecification(
        title="Biased Bell Proposal",
        objective="Test computational precedence rule.",
        hypothesis="Bell state with asymmetric measurement distribution.",
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

    report = engine.verify_proposal(biased_proposal)

    # Must FAIL regardless of any LLM claims
    assert report.summary_counts["FAIL"] >= 1
    assert "FAILED" in report.overall_summary
    outcome_check = next(c for c in report.checks if c.check_id == "CHK-DIST-01")
    assert outcome_check.status == CheckStatus.FAIL
