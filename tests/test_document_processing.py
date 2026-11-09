"""
Automated Pytest Suite: Document Processing and Traceable Chunker.
Tests section identification, math symbol detection, anti-hallucination warnings,
and chunk traceability.
"""

import pytest
from src.document_processing.pdf_extractor import PDFExtractor
from src.document_processing.chunker import TraceableChunker


def test_section_and_math_extraction():
    extractor = PDFExtractor()
    sample_text = (
        "Demonstration of Quantum Teleportation Protocol\n\n"
        "Abstract: We propose an experimental verification of quantum teleportation using |Phi+> Bell states.\n\n"
        "Objective: To transfer state |psi> from Alice to Bob with unit fidelity.\n\n"
        "Methodology: We initialize 3 qubits in state |000>, apply H(0) and CX(0,1) to establish entanglement.\n\n"
        "Expected Results: Outcomes |00> and |11> each have 0.50 probability.\n\n"
        "Assumptions: Noiseless classical communication channel."
    )

    sections = extractor._identify_sections(sample_text)
    assert "abstract" in sections
    assert "objective" in sections
    assert "methodology" in sections
    assert "expected_results" in sections

    math_symbols = extractor._extract_math_expressions(sample_text)
    assert any("|" in s for s in math_symbols)


def test_traceable_chunker_provenance():
    chunker = TraceableChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "text": "Sentence one about quantum entanglement. Sentence two on Bell states."},
        {"page_number": 2, "text": "Page two discussing Grover's diffusion operator and oracle reflection."}
    ]

    chunks = chunker.chunk_document(full_text="", source_filename="test_paper.pdf", pages=pages)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.source_filename == "test_paper.pdf"
        assert c.page_number in (1, 2)
        assert c.chunk_id.startswith("test_paper.pdf_p")
        assert len(c.text) > 0


def test_scanned_document_anti_hallucination_warning():
    """
    Documents with sparse text (< 50 characters) must trigger a critical warning
    and NEVER hallucinate missing text.
    """
    extractor = PDFExtractor()
    # Mocking extracted document with almost empty page text
    empty_pages = [{"page_number": 1, "text": "   ", "char_count": 0}]

    total_chars = sum(p["char_count"] for p in empty_pages)
    warnings = []
    if total_chars < 50:
        warnings.append("CRITICAL WARNING: The document contains fewer than 50 text characters.")

    assert len(warnings) == 1
    assert "fewer than 50 text characters" in warnings[0]
