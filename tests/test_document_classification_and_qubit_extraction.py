"""
Automated Regression Tests for Document Classification and Strict Qubit Count Extraction.
Verifies:
1. Paper with DOI 10.1038/s42256-025-01153-0 does not produce an extracted qubit count of 42257.
2. Document explicitly specifying a two-qubit circuit extracts 2.
3. Document with no qubit information returns num_qubits = null.
4. Paper discussing multiple circuits preserves the separate experiments.
5. Paper containing years (2024, 2025) and citations does not confuse them with qubit counts.
6. An incomplete proposal does not crash the application and marks the qubit count as inconclusive.
7. A valid two-qubit Bell-state proposal still passes extraction and verification.
"""

import pytest
from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import (
    ProposalExtractor,
    extract_doi,
    extract_strict_num_qubits,
    extract_strict_gates,
    classify_document,
)
from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.research_paper import ResearchPaperAnalysis, QuantumExperimentRecord
from src.schemas.verification import CheckStatus
from src.verification.engine import VerificationEngine
from src.verification.rules import check_qubit_count


@pytest.fixture
def mock_extractor():
    client = ConfigurableLLMClient(demo_mode=True)
    return ProposalExtractor(llm_client=client)


@pytest.fixture
def verification_engine():
    client = ConfigurableLLMClient(demo_mode=True)
    return VerificationEngine(llm_client=client)


def test_doi_paper_does_not_extract_42257(mock_extractor):
    """
    Test 1: A paper with DOI 10.1038/s42256-025-01153-0 must NOT extract 42257
    as the number of qubits or an S gate on qubit 42256.
    """
    doi_paper_text = (
        "Nature Machine Intelligence\n"
        "Article: Quantum Graph Representation Learning on Superconducting Processors\n"
        "DOI: 10.1038/s42256-025-01153-0\n"
        "Published online: 24 February 2025\n"
        "Abstract: We introduce a scalable graph neural architecture for variational quantum circuits. "
        "The framework overcomes traditional ansatz expressibility limits by embedding topology into pulse schedules.\n"
        "Methodology: Theoretical formulation and empirical simulations on noisy devices.\n"
        "References: [1] Preskill, J. (2018). Quantum Computing in the NISQ era and beyond. Quantum 2, 79."
    )

    # 1. DOI extraction
    doi = extract_doi(doi_paper_text, filename="s42256-025-01153-0.pdf")
    assert doi == "10.1038/s42256-025-01153-0"

    # 2. Strict qubit extraction must reject 42256 / 42257
    extracted_q = extract_strict_num_qubits(doi_paper_text)
    assert extracted_q != 42257
    assert extracted_q != 42256
    assert extracted_q is None

    # 3. Strict gate extraction must not match 's' in s42256 as an S gate on qubit 42256
    extracted_gates = extract_strict_gates(doi_paper_text)
    for g in extracted_gates:
        for t in g.target_qubits:
            assert t < 10
            assert t != 42256

    # 4. Document classification
    doc_type = classify_document(doi_paper_text, filename="s42256-025-01153-0.pdf")
    assert doc_type == "research_paper"

    # 5. Full extractor pipeline does not crash with validation error
    spec = mock_extractor.extract(doi_paper_text, document_title="Nature Article s42256", input_mode="upload")
    assert isinstance(spec, ProposalSpecification)
    assert spec.num_qubits is None
    assert spec.num_qubits != 42257
    assert spec.circuit is None
    assert spec.doi == "10.1038/s42256-025-01153-0"
    assert spec.document_type == "research_paper"


def test_explicit_two_qubit_circuit_extracts_2(mock_extractor):
    """
    Test 2: A document explicitly specifying a two-qubit circuit extracts 2.
    """
    two_qubit_text = (
        "Quantum Experiment Proposal: Entanglement Demonstration\n"
        "Objective: Prepare a Bell state on a two-qubit quantum register.\n"
        "Hypothesis: A Hadamard gate followed by CNOT creates equal superposition.\n"
        "System: Superconducting 2-qubit processor.\n"
        "Gate Sequence:\n"
        "H 0\n"
        "CX 0 1\n"
        "Expected measurement distribution:\n"
        "00: 0.5\n"
        "11: 0.5\n"
    )

    extracted_q = extract_strict_num_qubits(two_qubit_text)
    assert extracted_q == 2

    spec = mock_extractor.extract(two_qubit_text, input_mode="upload")
    assert spec.num_qubits == 2
    if spec.circuit:
        assert spec.circuit.num_qubits == 2


def test_no_qubit_information_returns_null(mock_extractor):
    """
    Test 3: A document with no qubit information returns num_qubits = null.
    """
    conceptual_text = (
        "Theoretical Analysis of Quantum Annealing Landscapes\n"
        "Abstract: We explore topological obstructions in non-convex Hamiltonian landscapes. "
        "The spectral gap behavior near phase transitions is evaluated theoretically using perturbation methods.\n"
        "Conclusion: Further architectural developments are needed to achieve runtime advantages."
    )

    extracted_q = extract_strict_num_qubits(conceptual_text)
    assert extracted_q is None

    spec = mock_extractor.extract(conceptual_text, input_mode="upload")
    assert spec.num_qubits is None
    assert spec.circuit is None
    assert any("Target system size" in m for m in spec.missing_information)


def test_paper_discussing_multiple_circuits_preserves_separate_experiments(mock_extractor):
    """
    Test 4: A paper discussing multiple circuits preserves the separate experiments.
    """
    multi_experiment_text = (
        "Nature Quantum Information | Article\n"
        "DOI: 10.1038/s41534-024-00123-4\n"
        "Title: Multi-Scale Quantum Benchmarks Across Superconducting Processors\n\n"
        "Abstract: We benchmark diverse multi-qubit workloads on transmon architectures.\n\n"
        "Experiment 1: Two-qubit Bell State Preparation\n"
        "We implement a two-qubit entanglement benchmark with H gate on qubit 0 and CX gate on qubits 0, 1.\n"
        "Expected measurement probabilities: P(00)=0.5, P(11)=0.5.\n\n"
        "Experiment 2: Three-qubit GHZ Tripartite Entanglement\n"
        "We evaluate a three-qubit GHZ state register with H gate on qubit 0 and CX on qubits 0, 1 and 1, 2.\n"
        "Expected outcomes: P(000)=0.5, P(111)=0.5.\n\n"
        "Experiment 3: Four-qubit Quantum Fourier Transform\n"
        "We evaluate a four-qubit register circuit assessing phase estimation accuracy.\n\n"
        "Results: Two-qubit fidelity reached 99.1%, while four-qubit fidelity decayed to 87.4% due to two-qubit gate cross-talk.\n"
        "Limitations: Readout error and residual ZZ coupling degrade deeper circuits."
    )

    paper_analysis = mock_extractor.extract_research_paper(multi_experiment_text, document_title="Multi-Scale Benchmarks")
    assert isinstance(paper_analysis, ResearchPaperAnalysis)
    assert len(paper_analysis.experiments) >= 2

    # Verify separate experiment records
    exp_ids = [e.experiment_id for e in paper_analysis.experiments]
    assert "EXP-01" in exp_ids
    assert "EXP-02" in exp_ids

    # Verify distinct titles / circuits
    assert any("two" in e.title.lower() or "bell" in e.title.lower() for e in paper_analysis.experiments)
    assert any("three" in e.title.lower() or "ghz" in e.title.lower() for e in paper_analysis.experiments)


def test_years_and_citations_not_confused_with_qubits(mock_extractor):
    """
    Test 5: A paper containing years (e.g. 2024, 2025) and citations does not confuse them with qubit counts.
    """
    academic_citation_text = (
        "Recent Progress in Superconducting Quantum Computing (2025 Review)\n"
        "Published in Journal of Quantum Tech, Vol. 42, pp. 115-130, 2024.\n"
        "Prior studies in 2021 [1], 2022 [15], 2023 [88], and 2024 [127] established baseline coherence limits. "
        "As seen in Figure 3 and Equation (12), gate fidelity varies non-linearly with pulse duration. "
        "Reference [2025] provides a comprehensive survey of NISQ limitations.\n"
        "Page 42 contains supplementary notes.\n"
    )

    extracted_q = extract_strict_num_qubits(academic_citation_text)
    # Must NOT extract 2024, 2025, 42, 115, 130, 15, 88, 127 as qubit count
    assert extracted_q is None

    extracted_gates = extract_strict_gates(academic_citation_text)
    assert len(extracted_gates) == 0

    spec = mock_extractor.extract(academic_citation_text, input_mode="upload")
    assert spec.num_qubits is None
    assert spec.circuit is None


def test_incomplete_proposal_does_not_crash_and_marks_inconclusive(mock_extractor, verification_engine):
    """
    Test 6: An incomplete proposal does not crash the application and marks the qubit count as inconclusive.
    """
    incomplete_text = (
        "Quantum Chemistry Acceleration Proposal\n"
        "Objective: Calculate ground state energy of LiH molecule.\n"
        "Hypothesis: VQE ansatz reaches chemical accuracy faster than classical Hartree-Fock."
    )

    # 1. Extraction must succeed without error
    spec = mock_extractor.extract(incomplete_text, input_mode="upload")
    assert spec.num_qubits is None
    assert spec.circuit is None

    # 2. Rule check CHK-STRUC-01 returns INCONCLUSIVE
    qubit_chk = check_qubit_count(spec)
    assert qubit_chk.status == CheckStatus.INCONCLUSIVE
    assert "CHK-STRUC-01" == qubit_chk.check_id

    # 3. Verification engine runs safely without crashing
    report = verification_engine.verify_proposal(spec)
    assert report is not None
    assert report.simulation is None  # Simulation safely bypassed
    assert report.summary_counts.get("INCONCLUSIVE", 0) >= 1
    assert "INCONCLUSIVE" in report.overall_summary


from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever


def test_valid_bell_state_proposal_still_passes(mock_extractor):
    """
    Test 7: A valid two-qubit Bell-state proposal still passes extraction and verification.
    """
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    retriever = QuantumRetriever(kb)
    client = ConfigurableLLMClient(demo_mode=True)
    engine = VerificationEngine(retriever=retriever, llm_client=client)

    bell_proposal = ProposalSpecification(
        title="Valid Bell State Verification",
        objective="Prepare maximally entangled state (|00> + |11>)/sqrt(2).",
        hypothesis="Hadamard followed by CNOT creates the |Phi+> state with 50% probability on |00> and |11>.",
        quantum_system="Superconducting Transmon",
        num_qubits=2,
        initial_state="|00>",
        measurement_basis="computational (Z)",
        circuit=CircuitSpec(
            num_qubits=2,
            gates=[
                GateOperation(name="H", target_qubits=[0]),
                GateOperation(name="CX", target_qubits=[0, 1]),
            ],
            initial_state="|00>",
            measurement_basis="computational (Z)"
        ),
        expected_outcomes=[
            ExpectedOutcome(basis_state="00", expected_probability=0.5, tolerance=0.05),
            ExpectedOutcome(basis_state="11", expected_probability=0.5, tolerance=0.05)
        ],
        assumptions=["Ideal noiseless simulation"],
        input_mode="manual",
        confirmed_by_researcher=True
    )

    report = engine.verify_proposal(bell_proposal)
    assert report is not None
    assert report.simulation is not None
    assert report.simulation.is_normalized is True
    assert report.simulation.num_qubits == 2
    assert report.simulation.probabilities.get("00", 0.0) == pytest.approx(0.5, abs=1e-4)
    assert report.simulation.probabilities.get("11", 0.0) == pytest.approx(0.5, abs=1e-4)
    assert report.summary_counts.get("FAIL", 0) == 0
    assert "PASS" in report.overall_summary
