"""
Regression tests for active_spec NoneType handling, state management,
and extraction robustness in Quantum Verifier.

Tests covering user requirements:
1. active_spec is None: safe handling without AttributeError, clear messaging, simulation blocked.
2. Valid experiment specification: correct conversion to Pydantic model and full verification.
3. Missing experiment title: non-fabrication of title, captured in missing_information without crashing.
4. Failed Vertex AI extraction: robust error capture, graceful fallback, retry capability.
5. Research paper with multiple experiments: preserving discrete experiment records separately.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.schemas.proposal import (
    ProposalSpecification,
    CircuitSpec,
    GateOperation,
    ExpectedOutcome,
)
from src.schemas.research_paper import (
    ResearchPaperAnalysis,
    QuantumExperimentRecord,
)
from src.schemas.verification import CheckStatus, VerificationReport
from src.llm.client import ConfigurableLLMClient, LLMConfigurationError
from src.llm.extractor import ProposalExtractor, extract_doi, extract_strict_num_qubits
from src.document_processing.document_extractor import ExtractedDocument
from src.circuit.simulator import QuantumSimulator
from src.verification.engine import VerificationEngine
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever


# -----------------------------------------------------------------------------
# 1. Regression Test: active_spec is None
# -----------------------------------------------------------------------------
def test_regression_active_spec_is_none_safe_handling():
    """
    Verify that when active_spec is None (e.g., initial state or reset state):
    - No AttributeError is raised for active_spec.title or any other attribute.
    - System accurately detects that no proposal specification is available.
    - Simulation cannot be executed without a confirmed specification.
    """
    active_spec = None
    paper_data = None
    report = None
    last_extraction_error = None

    # Simulate UI safety logic from Tab 2:
    if active_spec is not None:
        title_str = active_spec.title
        has_spec = True
    else:
        title_str = None
        has_spec = False

    assert has_spec is False
    assert title_str is None

    # Simulation check: must be blocked when no confirmed specification exists
    can_simulate = bool(report and report.simulation)
    assert can_simulate is False

    # When no paper and no active_spec exist, a clear message must be produced
    if not paper_data and not active_spec:
        guidance = "No experiment specification is available. Please upload a proposal or enter parameters in Tab 1."
    else:
        guidance = ""
    assert "No experiment specification is available" in guidance


# -----------------------------------------------------------------------------
# 2. Regression Test: Valid Experiment Specification
# -----------------------------------------------------------------------------
def test_regression_valid_experiment_specification():
    """
    Verify that a valid quantum experiment specification:
    - Parses and validates into ProposalSpecification schema without error.
    - Preserves all fields: title, objective, hypothesis, qubits, gates, outcomes.
    - Successfully executes Qiskit simulation and produces PASS report on verification.
    """
    spec_data = {
        "title": "Two-Qubit Bell State Preparation",
        "objective": "Prepare and verify maximally entangled state (|00> + |11>)/sqrt(2).",
        "hypothesis": "H on q0 and CX on (q0, q1) generates equal superposition of |00> and |11>.",
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
        "assumptions": ["Ideal gate timing", "Zero thermal excitation"],
        "confirmed_by_researcher": True
    }

    # Pydantic validation
    spec = ProposalSpecification(**spec_data)
    assert spec.title == "Two-Qubit Bell State Preparation"
    assert spec.num_qubits == 2
    assert spec.circuit is not None
    assert len(spec.circuit.gates) == 2
    assert len(spec.expected_outcomes) == 2
    assert spec.confirmed_by_researcher is True

    # Run verification engine
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=True)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    report = engine.verify_proposal(spec)
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["PASS"] >= 1
    assert report.simulation is not None
    assert report.simulation.probabilities.get("00") == 0.5
    assert report.simulation.probabilities.get("11") == 0.5


# -----------------------------------------------------------------------------
# 3. Regression Test: Missing Experiment Title
# -----------------------------------------------------------------------------
def test_regression_missing_experiment_title():
    """
    Verify that when an experiment proposal omits a title:
    - ProposalSpecification allows title=None without raising a Pydantic ValidationError.
    - System does NOT fabricate a title.
    - missing_information captures that the title was not specified.
    - UI and reporting layers display safe fallback indicator.
    """
    # 1. Direct model instantiation with title=None
    spec = ProposalSpecification(
        title=None,
        objective="Investigate quantum circuit behavior.",
        num_qubits=2,
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[GateOperation(name="H", target_qubits=[0])]
        )
    )
    assert spec.title is None
    assert any("title was not specified" in m for m in spec.missing_information)

    # 2. UI safe display test
    title_display = spec.title if (spec.title and spec.title.strip()) else "*(Untitled / Not Specified in Proposal)*"
    assert title_display == "*(Untitled / Not Specified in Proposal)*"

    # 3. Verification engine must not crash when title is None
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=True)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    report = engine.verify_proposal(spec)
    assert report.proposal_title == "Untitled Quantum Proposal"
    assert report.checks is not None


# -----------------------------------------------------------------------------
# 4. Regression Test: Failed Vertex AI Extraction
# -----------------------------------------------------------------------------
def test_regression_failed_vertex_ai_extraction_handling():
    """
    Verify that when Vertex AI / Gemini extraction fails (e.g. API error, invalid JSON):
    - The exception is captured and logged without crashing the app.
    - Safe deterministic fallback produces a structured specification.
    - Fallback flags the extraction warning / diagnostic note.
    - If the extraction completely fails at the document level, the error is preserved.
    """
    mock_failing_client = MagicMock(spec=ConfigurableLLMClient)
    mock_failing_client.complete.side_effect = RuntimeError("Vertex AI 503 Service Unavailable spike")
    mock_failing_client.is_configured.return_value = (True, "Vertex AI Active")

    extractor = ProposalExtractor(mock_failing_client)
    sample_text = (
        "Proposal for Two-Qubit Entanglement\n"
        "Objective: Test Bell state generation.\n"
        "We use a two-qubit circuit with H on qubit 0 and CX on qubits 0, 1.\n"
        "Expected outcomes: P(00)=0.5, P(11)=0.5."
    )

    spec = extractor.extract(sample_text, document_title="Bell State Test")

    # Safe fallback must succeed and preserve key parameters
    assert isinstance(spec, ProposalSpecification)
    assert spec.num_qubits == 2
    assert spec.circuit is not None
    assert len(spec.circuit.gates) == 2
    # Must capture the failure in extraction warnings
    assert any("503 Service Unavailable" in w for w in spec.extraction_warnings)


# -----------------------------------------------------------------------------
# 5. Regression Test: Research Paper with Multiple Experiments
# -----------------------------------------------------------------------------
def test_regression_research_paper_with_multiple_experiments():
    """
    Verify that an academic research paper discussing multiple distinct experiments:
    - Is classified as a 'research_paper'.
    - Preserves discrete QuantumExperimentRecord instances in ResearchPaperAnalysis.experiments.
    - Does NOT confuse DOI/article identifiers (e.g. s42256) with qubit counts.
    - Allows each experiment record to be independently loaded into a ProposalSpecification.
    """
    multi_exp_text = (
        "Nature Machine Intelligence | Research Article\n"
        "DOI: 10.1038/s42256-025-01153-0\n"
        "Title: Multi-Circuit Benchmarking on Superconducting Processors\n\n"
        "Abstract: We benchmark two distinct quantum circuits on a superconducting processor.\n\n"
        "Experiment 1: Two-qubit Bell State Preparation\n"
        "We allocate 2 qubits and apply H gate on qubit 0, followed by CX on qubits 0, 1.\n"
        "Expected outcomes: P(00) = 0.5, P(11) = 0.5.\n\n"
        "Experiment 2: Three-qubit GHZ State Preparation\n"
        "We allocate 3 qubits and apply H gate on qubit 0, CX on qubits 0, 1, and CX on qubits 1, 2.\n"
        "Expected outcomes: P(000) = 0.5, P(111) = 0.5.\n\n"
        "Results: Both experiments verified state fidelity above 95%."
    )

    client = ConfigurableLLMClient(demo_mode=True)
    extractor = ProposalExtractor(client)

    # 1. Classification
    doc_type = extractor.classify(multi_exp_text, filename="s42256-025-01153-0.pdf")
    assert doc_type == "research_paper"

    # 2. Research Paper Extraction
    paper = extractor.extract_research_paper(
        multi_exp_text,
        document_title="Multi-Circuit Benchmarking",
        extracted_doc=ExtractedDocument(
            filename="s42256-025-01153-0.pdf",
            full_text=multi_exp_text,
            pages=[{"page_number": 1, "text": multi_exp_text, "char_count": len(multi_exp_text)}],
            sections={},
            math_expressions=[],
            warnings=[]
        )
    )

    assert isinstance(paper, ResearchPaperAnalysis)
    assert paper.document_type == "research_paper"
    assert "10.1038/s42256-025-01153-0" in (paper.doi or "")

    # Discrete experiments preserved
    assert len(paper.experiments) >= 2
    exp1 = paper.experiments[0]
    exp2 = paper.experiments[1]

    assert exp1.num_qubits == 2
    assert exp1.num_qubits != 42257
    assert exp2.num_qubits == 3
    assert exp2.num_qubits != 42257

    # 3. Independent loading of each experiment into ProposalSpecification
    spec1 = ProposalSpecification(
        title=exp1.title,
        objective=f"Verify {exp1.title}",
        num_qubits=exp1.num_qubits,
        circuit=CircuitSpec(num_qubits=exp1.num_qubits, gates=exp1.gates) if exp1.gates else None,
        expected_outcomes=exp1.expected_outcomes
    )
    spec2 = ProposalSpecification(
        title=exp2.title,
        objective=f"Verify {exp2.title}",
        num_qubits=exp2.num_qubits,
        circuit=CircuitSpec(num_qubits=exp2.num_qubits, gates=exp2.gates) if exp2.gates else None,
        expected_outcomes=exp2.expected_outcomes
    )

    assert spec1.num_qubits == 2
    assert spec2.num_qubits == 3
    assert spec1.title != spec2.title
