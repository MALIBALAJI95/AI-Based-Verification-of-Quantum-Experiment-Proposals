"""
Modular Verification Engine.
Coordinates structural validation, quantum simulation, probability matching,
literature retrieval, and claim cross-examination.
Enforces the core scientific principle: Computational verification cannot be overridden by an LLM.
"""

import sys
import platform
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from ..schemas.proposal import ProposalSpecification
from ..schemas.verification import (
    CheckStatus,
    CheckCategory,
    VerificationCheck,
    SimulationOutput,
    VerificationReport,
    LiteratureCitation,
    ClaimVerificationPair,
)
from ..circuit.simulator import QuantumSimulator, SimulationError
from ..circuit.metrics import compute_fidelity
from ..knowledge_base.retriever import QuantumRetriever
from ..llm.analyzer import ScientificAnalyzer
from ..llm.client import ConfigurableLLMClient
from .rules import (
    check_qubit_count,
    check_circuit_syntax,
    check_statevector_normalization,
    check_outcome_distribution,
    check_physical_hardware_feasibility,
)


class VerificationEngine:
    """
    End-to-end scientific verification pipeline for quantum experiment proposals.
    """

    def __init__(
        self,
        retriever: Optional[QuantumRetriever] = None,
        llm_client: Optional[ConfigurableLLMClient] = None
    ):
        self.simulator = QuantumSimulator()
        self.retriever = retriever
        self.llm_client = llm_client or ConfigurableLLMClient(demo_mode=True)
        self.analyzer = ScientificAnalyzer(self.llm_client)

    def verify_proposal(self, proposal: ProposalSpecification) -> VerificationReport:
        """
        Execute full multi-tier verification pipeline on a proposal.
        """
        report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        checks: List[VerificationCheck] = []
        simulation_output: Optional[SimulationOutput] = None
        outcome_comparisons = []
        tvd_score: Optional[float] = None
        fidelity_score: Optional[float] = None

        # -------------------------------------------------------------
        # 1. Structural Checks
        # -------------------------------------------------------------
        qubit_chk = check_qubit_count(proposal)
        checks.append(qubit_chk)

        circuit_chk = check_circuit_syntax(proposal)
        checks.append(circuit_chk)

        can_simulate = (qubit_chk.status == CheckStatus.PASS and circuit_chk.status == CheckStatus.PASS)

        # -------------------------------------------------------------
        # 2. Quantum Circuit Simulation (Ideal Statevector)
        # -------------------------------------------------------------
        sim_error_msg = None
        if can_simulate and proposal.circuit:
            try:
                simulation_output = self.simulator.simulate(proposal.circuit)
            except SimulationError as se:
                sim_error_msg = str(se)
            except Exception as e:
                sim_error_msg = f"Unexpected simulation runtime failure: {e}"

        # -------------------------------------------------------------
        # 3. Statevector Normalization Check
        # -------------------------------------------------------------
        norm_chk = check_statevector_normalization(simulation_output)
        if sim_error_msg:
            norm_chk.status = CheckStatus.FAIL
            norm_chk.details = f"Circuit simulation failed: {sim_error_msg}"
            norm_chk.error_message = sim_error_msg
        checks.append(norm_chk)

        # -------------------------------------------------------------
        # 4. Expected vs Computed Outcome Comparison
        # -------------------------------------------------------------
        dist_chk, comparisons, tvd = check_outcome_distribution(proposal, simulation_output)
        checks.append(dist_chk)
        outcome_comparisons = comparisons
        tvd_score = tvd

        # -------------------------------------------------------------
        # 5. Literature Retrieval & Grounding Check
        # -------------------------------------------------------------
        citations: List[LiteratureCitation] = []
        if self.retriever:
            citations = self.retriever.retrieve_for_proposal(proposal)

        if citations:
            lit_chk = VerificationCheck(
                check_id="CHK-LIT-01",
                name="Literature Evidence Grounding",
                category=CheckCategory.LITERATURE_GROUNDING,
                status=CheckStatus.PASS,
                description="Cross-reference proposal claims against genuine peer-reviewed quantum literature.",
                details=f"Retrieved {len(citations)} traceable literature passages supporting or contextualizing claims."
            )
        else:
            lit_chk = VerificationCheck(
                check_id="CHK-LIT-01",
                name="Literature Evidence Grounding",
                category=CheckCategory.LITERATURE_GROUNDING,
                status=CheckStatus.INCONCLUSIVE,
                description="Cross-reference proposal claims against genuine peer-reviewed quantum literature.",
                details="No directly matching passages retrieved from knowledge base. Proposal contains ungrounded claims or requires literature ingestion.",
                error_message="Missing literature grounding: 0 matching passages."
            )
        checks.append(lit_chk)

        # -------------------------------------------------------------
        # 6. Physical Hardware Feasibility Disclaimer Check
        # -------------------------------------------------------------
        phys_chk = check_physical_hardware_feasibility(proposal)
        checks.append(phys_chk)

        # -------------------------------------------------------------
        # 7. Scientific Claims Analysis
        # -------------------------------------------------------------
        claims_analysis = self.analyzer.cross_examine_claims(
            proposal=proposal,
            citations=citations,
            simulation=simulation_output,
            outcome_comparisons=outcome_comparisons
        )

        # -------------------------------------------------------------
        # 8. Summary Statistics & Overall Synthesis
        # -------------------------------------------------------------
        counts = {
            "PASS": sum(1 for c in checks if c.status == CheckStatus.PASS),
            "FAIL": sum(1 for c in checks if c.status == CheckStatus.FAIL),
            "INCONCLUSIVE": sum(1 for c in checks if c.status == CheckStatus.INCONCLUSIVE),
            "NOT_RUN": sum(1 for c in checks if c.status == CheckStatus.NOT_RUN),
        }

        # Build detected errors list
        detected_errors = [c.error_message for c in checks if c.status == CheckStatus.FAIL and c.error_message]
        if proposal.potential_inconsistencies:
            detected_errors.extend(proposal.potential_inconsistencies)

        # Build missing information list
        missing_info = list(proposal.missing_information)
        for c in checks:
            if c.status == CheckStatus.INCONCLUSIVE and c.error_message:
                if c.error_message not in missing_info:
                    missing_info.append(c.error_message)

        # Recommended next steps
        next_steps = self._formulate_next_steps(counts, detected_errors, missing_info)

        # Formulate overall summary
        overall_summary = self._synthesize_overall_summary(counts, detected_errors, missing_info)

        # Environment metadata
        is_live, provider_info = self.llm_client.is_configured()
        software_env = {
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "llm_provider": provider_info,
            "verification_engine_version": "1.0.0"
        }

        return VerificationReport(
            report_id=report_id,
            proposal_title=proposal.title or "Untitled Quantum Proposal",
            timestamp=timestamp,
            software_environment=software_env,
            overall_summary=overall_summary,
            checks=checks,
            summary_counts=counts,
            simulation=simulation_output,
            outcome_comparisons=outcome_comparisons,
            fidelity_score=fidelity_score,
            total_variation_distance=tvd_score,
            citations=citations,
            claims_analysis=claims_analysis,
            missing_information=missing_info,
            detected_inconsistencies=detected_errors,
            recommended_next_steps=next_steps
        )

    def _synthesize_overall_summary(
        self, counts: Dict[str, int], errors: List[str], missing: List[str]
    ) -> str:
        if counts["FAIL"] > 0:
            return (
                f"VERIFICATION STATUS: FAILED ({counts['FAIL']} failing check(s)). "
                f"Computational simulation or structural bounds identified critical discrepancies. "
                f"The proposal cannot be physically certified in its current state."
            )
        elif counts["INCONCLUSIVE"] > 0:
            return (
                f"VERIFICATION STATUS: INCONCLUSIVE ({counts['INCONCLUSIVE']} inconclusive check(s)). "
                f"Essential parameters (such as qubit count, gate sequence, or probability distributions) "
                f"were omitted or ambiguous. Human review and additional specification are required."
            )
        else:
            return (
                f"VERIFICATION STATUS: PASSED ({counts['PASS']} checks passed). "
                f"Ideal quantum statevector simulation confirmed all mathematical operations and probability distributions. "
                f"Notice: Passing ideal simulation does not guarantee feasibility on noisy physical NISQ hardware."
            )

    def _formulate_next_steps(
        self, counts: Dict[str, int], errors: List[str], missing: List[str]
    ) -> List[str]:
        steps = []
        if errors:
            steps.append("Rectify mathematical discrepancies identified in circuit specification and expected outcomes.")
        if missing:
            steps.append("Provide missing experimental parameters (exact gate sequences, rotation angles, measurement basis).")
        if counts["FAIL"] == 0 and counts["INCONCLUSIVE"] == 0:
            steps.append("Formulate open quantum systems noise model (T1/T2 decoherence, gate crosstalk) for target hardware.")
            steps.append("Perform error-mitigation analysis (Zero-Noise Extrapolation or Readout Error Mitigation).")
            steps.append("Submit proposal to peer scientific review prior to hardware runtime allocation.")
        return steps
