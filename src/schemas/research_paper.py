"""
Pydantic schemas for Academic Research Paper Analysis.
Distinguishes broad academic research papers from single executable quantum experiment proposals.
Supports multi-experiment papers with individual experiment records, methodology,
algorithmic frameworks, experimental setup, findings, and limitations.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

from .proposal import GateOperation, ExpectedOutcome


class QuantumExperimentRecord(BaseModel):
    """An individual quantum experiment, benchmark, or circuit described in a research paper."""
    experiment_id: str = Field(..., description="Unique identifier, e.g. EXP-01")
    title: str = Field(..., description="Experiment or benchmark title")
    quantum_system: Optional[str] = Field(default=None, description="Physical system (e.g. Superconducting, Trapped Ion, Neutral Atom)")
    num_qubits: Optional[int] = Field(default=None, ge=1, le=10, description="Qubit register size if explicit and <= 10")
    circuit_summary: Optional[str] = Field(default=None, description="Summary of gates, pulse sequences, or circuit layout")
    has_executable_circuit: bool = Field(default=False, description="Whether this experiment specifies an executable small gate sequence (<= 10 qubits)")
    gates: List[GateOperation] = Field(default_factory=list, description="Extracted gate operations if available")
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list, description="Hypothesized or computed basis probabilities")
    findings: Optional[str] = Field(default=None, description="Observed experimental results or metrics")


class ResearchPaperAnalysis(BaseModel):
    """Structured scientific analysis for academic research papers."""
    document_type: Literal["research_paper"] = "research_paper"
    title: str = Field(..., min_length=2, description="Paper title")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier (e.g. 10.1038/s42256-025-01153-0)")
    publication_info: Optional[str] = Field(default=None, description="Journal, conference, or year of publication")
    authors: List[str] = Field(default_factory=list, description="Author names if extractable")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    
    # Core Academic Paper Sections
    research_problem: str = Field(..., description="Core scientific problem or challenge addressed by authors")
    research_objective: str = Field(..., description="Primary research objective and goal")
    methodology: str = Field(..., description="Theoretical and experimental methodology")
    algorithms_and_frameworks: List[str] = Field(default_factory=list, description="Algorithms, quantum software, or frameworks discussed (e.g. VQE, QAOA, Qiskit)")
    experimental_setup: str = Field(default="", description="Experimental platform, hardware devices, or simulation settings")
    quantum_systems_discussed: List[str] = Field(default_factory=list, description="Quantum architectures discussed in the work")
    
    # Multiple experiments support
    experiments: List[QuantumExperimentRecord] = Field(default_factory=list, description="List of discrete quantum experiments or benchmarks discussed")
    
    results_and_findings: List[str] = Field(default_factory=list, description="Key experimental results, findings, and conclusions")
    limitations: List[str] = Field(default_factory=list, description="Stated limitations, noise impacts, decoherence, or architectural limits")
    
    # Traceable Provenance & Warnings
    evidence_and_citations: List[Dict[str, Any]] = Field(default_factory=list, description="Traceable literature evidence matching knowledge base")
    raw_text: Optional[str] = Field(default=None, description="Raw document text")
    source_filename: Optional[str] = Field(default=None, description="Source PDF or DOCX filename")
    source_pages: List[Dict[str, Any]] = Field(default_factory=list, description="Page-by-page text mapping")
    extraction_warnings: List[str] = Field(default_factory=list, description="Warnings generated during extraction")
