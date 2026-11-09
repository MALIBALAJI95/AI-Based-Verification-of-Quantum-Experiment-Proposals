"""
Pydantic schemas for Quantum Experiment Proposals and extracted specifications.
Provides strict validation, sensible defaults, and field constraints.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator


class GateOperation(BaseModel):
    """Represents a discrete gate operation in a gate-based quantum circuit."""
    name: str = Field(..., description="Gate identifier, e.g., 'H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ', 'CX', 'CZ', 'SWAP'")
    target_qubits: List[int] = Field(..., min_length=1, description="Zero-indexed qubit indices involved in this operation.")
    params: Optional[List[float]] = Field(default=None, description="Continuous rotation parameters (angles in radians) if applicable.")
    description: Optional[str] = Field(default=None, description="Human-readable description or notation from proposal.")

    @model_validator(mode="after")
    def validate_gate(self) -> "GateOperation":
        norm_name = self.name.strip().upper()
        single_qubit_gates = {"H", "X", "Y", "Z", "S", "T", "SDG", "TDG", "RX", "RY", "RZ", "I", "ID"}
        two_qubit_gates = {"CX", "CNOT", "CZ", "SWAP"}
        parametric_gates = {"RX", "RY", "RZ"}

        for q in self.target_qubits:
            if q < 0 or q >= 10:
                raise ValueError(f"Target qubit index {q} is out of bounds for prototype register (supported: 0-9).")

        if norm_name in single_qubit_gates and len(self.target_qubits) != 1:
            raise ValueError(f"Gate '{norm_name}' requires exactly 1 target qubit, but got {len(self.target_qubits)}: {self.target_qubits}")
        
        if norm_name in two_qubit_gates and len(self.target_qubits) != 2:
            raise ValueError(f"Gate '{norm_name}' requires exactly 2 target qubits (control, target), but got {len(self.target_qubits)}: {self.target_qubits}")

        if norm_name in two_qubit_gates and self.target_qubits[0] == self.target_qubits[1]:
            raise ValueError(f"Two-qubit gate '{norm_name}' cannot have identical control and target qubit: {self.target_qubits[0]}")

        if norm_name in parametric_gates and (not self.params or len(self.params) < 1):
            raise ValueError(f"Parametric gate '{norm_name}' requires at least 1 angle parameter in radians.")

        return self


class CircuitSpec(BaseModel):
    """Structured specification of a supported gate-based quantum circuit."""
    num_qubits: int = Field(..., ge=1, le=10, description="Total number of qubits (prototype supports 1-5 natively, up to 10 for small ideal simulation).")
    gates: List[GateOperation] = Field(default_factory=list, description="Ordered sequence of gate operations.")
    initial_state: str = Field(default="|0...0>", description="Initial state label, e.g., '|00>', '|000>'.")
    measurement_basis: str = Field(default="computational (Z)", description="Measurement basis.")
    notes: Optional[str] = Field(default=None, description="Author notes regarding layout or connectivity constraints.")

    @model_validator(mode="after")
    def validate_qubit_bounds(self) -> "CircuitSpec":
        for idx, gate in enumerate(self.gates):
            for q in gate.target_qubits:
                if q < 0 or q >= self.num_qubits:
                    raise ValueError(
                        f"Gate #{idx + 1} ({gate.name}) targets qubit index {q}, "
                        f"which is out of bounds for an allocated {self.num_qubits}-qubit circuit (indices 0..{self.num_qubits - 1})."
                    )
        return self


class ExpectedOutcome(BaseModel):
    """Expected measurement probability for a specific computational basis state."""
    basis_state: str = Field(..., description="Bitstring for computational basis state, e.g. '00', '11', '010'.")
    expected_probability: float = Field(..., ge=0.0, le=1.0, description="Hypothesized or theoretical measurement probability in [0, 1].")
    tolerance: float = Field(default=0.05, ge=0.0, le=0.5, description="Allowed deviation threshold between simulation and expected value.")


class ExtractedClaim(BaseModel):
    """An author claim or empirical assertion extracted from the proposal."""
    claim_id: str = Field(..., description="Unique claim identifier, e.g., 'CLM-01'.")
    claim_type: Literal["hypothesis", "methodology", "circuit", "outcome", "hardware_assumption"] = Field(
        ..., description="Category of scientific claim."
    )
    statement: str = Field(..., description="Exact claim text extracted from document.")
    source_reference: Optional[str] = Field(default=None, description="Section, page, or paragraph citation from proposal.")
    confidence_level: Literal["high", "medium", "low"] = Field(default="high")


class ProposalSpecification(BaseModel):
    """Comprehensive structured representation of a quantum experiment proposal."""
    title: Optional[str] = Field(default=None, description="Experiment title.")
    abstract: Optional[str] = Field(default=None, description="Abstract or executive summary.")
    objective: str = Field(default="Quantum experiment proposal.", description="Scientific goal and hypothesis of proposal.")
    hypothesis: Optional[str] = Field(default=None, description="Core physical or algorithmic hypothesis.")
    methodology: Optional[str] = Field(default=None, description="Experimental methodology and procedure.")
    quantum_system: Optional[str] = Field(default=None, description="Target quantum architecture (e.g., Superconducting Transmon, Trapped Ion, Photonic, Ideal Simulation).")
    num_qubits: Optional[int] = Field(default=None, ge=1, le=10, description="Target quantum system size.")
    initial_state: Optional[str] = Field(default=None, description="Initial state label, e.g. |00>, |000>.")
    measurement_basis: Optional[str] = Field(default=None, description="Measurement basis, e.g. computational (Z), Bell basis.")
    circuit: Optional[CircuitSpec] = Field(default=None, description="Parsed or specified circuit sequence.")
    expected_outcomes: List[ExpectedOutcome] = Field(default_factory=list, description="Expected state measurement distributions.")
    assumptions: List[str] = Field(default_factory=list, description="Physical/hardware or mathematical assumptions stated by author.")
    constraints: List[str] = Field(default_factory=list, description="Hardware constraints, e.g. gate fidelity, coherence limits.")
    missing_information: List[str] = Field(default_factory=list, description="Identified gaps or ambiguous specifications in proposal.")
    potential_inconsistencies: List[str] = Field(default_factory=list, description="Potential inconsistencies detected during extraction.")
    author_claims: List[ExtractedClaim] = Field(default_factory=list, description="List of discrete author assertions.")
    raw_text: Optional[str] = Field(default=None, description="Full raw text of proposal or user prompt.")
    extraction_warnings: List[str] = Field(default_factory=list, description="Warnings generated during PDF or text ingestion.")
    input_mode: Optional[str] = Field(default=None, description="Source input mode ('upload' or 'manual').")
    source_filename: Optional[str] = Field(default=None, description="Filename of original uploaded document.")
    source_pages: List[Dict[str, Any]] = Field(default_factory=list, description="Preserved page-by-page text mapping from source document.")
    confirmed_by_researcher: bool = Field(default=False, description="Whether researcher reviewed and confirmed specification before verification.")
    document_type: Literal["experiment_proposal", "research_paper"] = Field(default="experiment_proposal", description="Detected or selected document type.")
    doi: Optional[str] = Field(default=None, description="Publication DOI if detected in document.")
    experiments_discussed: List[Dict[str, Any]] = Field(default_factory=list, description="Multiple experiments/benchmarks discussed in document.")

    @model_validator(mode="before")
    @classmethod
    def handle_circuit_validation_errors(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check title presence without fabricating
            raw_title = data.get("title")
            if raw_title is not None and not str(raw_title).strip():
                data["title"] = None
            if not data.get("title"):
                missing = data.get("missing_information") or []
                if isinstance(missing, list) and not any("title" in m.lower() for m in missing):
                    missing.append("Experiment title was not specified in proposal.")
                    data["missing_information"] = missing

            # Check objective presence
            raw_obj = data.get("objective")
            if not raw_obj or not str(raw_obj).strip():
                data["objective"] = "Quantum experiment proposal."
                missing = data.get("missing_information") or []
                if isinstance(missing, list) and not any("objective" in m.lower() for m in missing):
                    missing.append("Scientific objective was not specified in proposal.")
                    data["missing_information"] = missing

            # Sync initial state and measurement basis if circuit is defined
            raw_circuit = data.get("circuit")
            if isinstance(raw_circuit, dict):
                if not data.get("initial_state") and raw_circuit.get("initial_state"):
                    data["initial_state"] = raw_circuit.get("initial_state")
                if not data.get("measurement_basis") and raw_circuit.get("measurement_basis"):
                    data["measurement_basis"] = raw_circuit.get("measurement_basis")

                try:
                    CircuitSpec(**raw_circuit)
                except Exception as e:
                    # Capture syntax or bounds errors into potential_inconsistencies
                    inconsistencies = data.get("potential_inconsistencies") or []
                    if isinstance(inconsistencies, list):
                        inconsistencies.append(f"Circuit specification error: {e}")
                        data["potential_inconsistencies"] = inconsistencies
                    data["circuit"] = None
        return data
