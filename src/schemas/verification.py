"""
Pydantic schemas for the Independent Scientific Verification Engine.
Standardizes check statuses (PASS, FAIL, INCONCLUSIVE, NOT_RUN) and report structures.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"


class CheckCategory(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    SIMULATION = "SIMULATION"
    DISTRIBUTION_MATCH = "DISTRIBUTION_MATCH"
    LITERATURE_GROUNDING = "LITERATURE_GROUNDING"
    PHYSICAL_FEASIBILITY = "PHYSICAL_FEASIBILITY"


class VerificationCheck(BaseModel):
    """An individual verification check with evidence and audit status."""
    check_id: str = Field(..., description="Unique check identifier, e.g. 'CHK-CIRC-01'.")
    name: str = Field(..., description="Short descriptive name.")
    category: CheckCategory = Field(..., description="Category of verification check.")
    status: CheckStatus = Field(..., description="Result of check: PASS, FAIL, INCONCLUSIVE, or NOT_RUN.")
    description: str = Field(..., description="What this check tests.")
    details: str = Field(..., description="Factual evaluation outcome and observed values.")
    error_message: Optional[str] = Field(default=None, description="Detailed error trace if check failed.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary numeric or structured diagnostics.")


class StatevectorComponent(BaseModel):
    """Component of the simulated quantum statevector."""
    basis_state: str
    real: float
    imag: float
    magnitude: float
    probability: float
    phase_rad: float


class SimulationOutput(BaseModel):
    """Exact computational output from Qiskit quantum circuit simulation."""
    num_qubits: int
    statevector_components: List[StatevectorComponent] = Field(default_factory=list)
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Computational basis bitstring -> probability.")
    is_normalized: bool
    normalization_error: float
    circuit_diagram_text: Optional[str] = None
    execution_time_ms: float = 0.0


class OutcomeComparison(BaseModel):
    """Comparison between author-hypothesized outcome and computationally verified outcome."""
    basis_state: str
    expected_probability: float
    computed_probability: float
    absolute_difference: float
    tolerance: float
    is_match: bool


class LiteratureCitation(BaseModel):
    """Traceable citation to verified quantum literature stored in the knowledge base."""
    citation_id: str
    source_title: str
    filename: str
    page_number: Optional[int] = None
    chunk_id: str
    passage: str
    similarity_score: float
    relevance_comment: str


class ClaimVerificationPair(BaseModel):
    """Explicit distinction between author claim, literature evidence, and computational validation."""
    claim_id: str
    claim_text: str
    claim_category: str
    literature_support_status: str  # Supported, Contradicted, Unaddressed
    supporting_citations: List[str] = Field(default_factory=list)
    computational_status: str       # Verified, Contradicted, Not Applicable
    explanation: str


class VerificationReport(BaseModel):
    """Complete, downloadable, explainable verification report."""
    report_id: str
    proposal_title: str
    timestamp: str
    software_environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Software environment info (Python, Qiskit, OS, LLM provider/mode)."
    )
    overall_summary: str
    checks: List[VerificationCheck] = Field(default_factory=list)
    summary_counts: Dict[str, int] = Field(
        default_factory=lambda: {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0, "NOT_RUN": 0}
    )
    simulation: Optional[SimulationOutput] = None
    outcome_comparisons: List[OutcomeComparison] = Field(default_factory=list)
    fidelity_score: Optional[float] = None
    total_variation_distance: Optional[float] = None
    citations: List[LiteratureCitation] = Field(default_factory=list)
    claims_analysis: List[ClaimVerificationPair] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    detected_inconsistencies: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)
    limitations_and_disclaimer: str = Field(
        default=(
            "RESEARCH PROTOTYPE DISCLAIMER: Computational verification uses ideal mathematical statevector simulation. "
            "A PASS status does not imply physical experiment feasibility on noisy quantum hardware (NISQ), "
            "nor does it account for decoherence, cross-talk, gate synthesis overhead, or optical bench alignment. "
            "Expert human scientific review is required before physical execution."
        )
    )
