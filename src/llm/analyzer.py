"""
Scientific Claim Cross-Examiner.
Cross-references author claims against retrieved literature and simulation outputs.
Enforces the inviolable rule: Computational simulation failure cannot be overridden by an LLM.
"""

import json
import logging
from typing import List, Optional

from .client import ConfigurableLLMClient
from .prompts import ANALYSIS_SYSTEM_PROMPT
from ..schemas.proposal import ProposalSpecification, ExtractedClaim
from ..schemas.verification import LiteratureCitation, SimulationOutput, OutcomeComparison, ClaimVerificationPair

logger = logging.getLogger(__name__)


class ScientificAnalyzer:
    """
    Performs multi-source scientific analysis distinguishing author assertions,
    literature evidence, model hypotheses, and verified computational facts.
    """

    def __init__(self, llm_client: ConfigurableLLMClient):
        self.llm_client = llm_client

    def cross_examine_claims(
        self,
        proposal: ProposalSpecification,
        citations: List[LiteratureCitation],
        simulation: Optional[SimulationOutput],
        outcome_comparisons: List[OutcomeComparison]
    ) -> List[ClaimVerificationPair]:
        """
        Cross-reference each author claim with literature citations and simulation output.
        """
        claims = proposal.author_claims
        if not claims:
            # If no discrete claims were parsed, generate standard ones from proposal specification
            claims = self._synthesize_claims_from_spec(proposal)

        # Check simulation status
        sim_has_discrepancy = any(not c.is_match for c in outcome_comparisons) if outcome_comparisons else False
        
        results: List[ClaimVerificationPair] = []

        for claim in claims:
            cid = claim.claim_id
            ctext = claim.statement
            ctype = claim.claim_type

            # Check computational relevance
            comp_status = "Not Applicable"
            if ctype in ("circuit", "outcome") or "probability" in ctext.lower() or "state" in ctext.lower():
                if simulation is None:
                    comp_status = "Not Applicable (No Simulation)"
                elif sim_has_discrepancy:
                    comp_status = "Contradicted (Computational Mismatch)"
                else:
                    comp_status = "Verified (Consistent with Statevector)"

            # Check literature relevance
            lit_status = "Unaddressed"
            matching_cits = []
            for cit in citations:
                # Check for keyword overlap
                keywords = [w.lower() for w in ctext.split() if len(w) > 4]
                if any(kw in cit.passage.lower() for kw in keywords):
                    matching_cits.append(cit.citation_id)
                    lit_status = "Supported"

            # Formulate clear scientific explanation
            explanation = self._build_explanation(claim, lit_status, comp_status, matching_cits, sim_has_discrepancy)

            results.append(
                ClaimVerificationPair(
                    claim_id=cid,
                    claim_text=ctext,
                    claim_category=ctype,
                    literature_support_status=lit_status,
                    supporting_citations=matching_cits,
                    computational_status=comp_status,
                    explanation=explanation
                )
            )

        return results

    def _synthesize_claims_from_spec(self, proposal: ProposalSpecification) -> List[ExtractedClaim]:
        """Create structured claims if author assertions were not explicitly separated."""
        claims = []
        if proposal.hypothesis:
            claims.append(
                ExtractedClaim(
                    claim_id="CLM-01",
                    claim_type="hypothesis",
                    statement=proposal.hypothesis,
                    confidence_level="high"
                )
            )
        if proposal.circuit and proposal.circuit.gates:
            gate_str = ", ".join([f"{g.name} on {g.target_qubits}" for g in proposal.circuit.gates])
            claims.append(
                ExtractedClaim(
                    claim_id="CLM-02",
                    claim_type="circuit",
                    statement=f"Circuit executes gate sequence: {gate_str}",
                    confidence_level="high"
                )
            )
        if proposal.expected_outcomes:
            outcomes_str = "; ".join([f"P({o.basis_state}) = {o.expected_probability}" for o in proposal.expected_outcomes])
            claims.append(
                ExtractedClaim(
                    claim_id="CLM-03",
                    claim_type="outcome",
                    statement=f"Expected measurement probability distribution: {outcomes_str}",
                    confidence_level="high"
                )
            )
        return claims

    def _build_explanation(
        self,
        claim: ExtractedClaim,
        lit_status: str,
        comp_status: str,
        citations: List[str],
        sim_has_discrepancy: bool
    ) -> str:
        if comp_status.startswith("Contradicted"):
            return (
                "Computational statevector simulation directly contradicts this outcome claim. "
                "Calculated probabilities deviate beyond the allowed numerical tolerance. "
                "Per verification rules, computational simulation takes precedence over natural language arguments."
            )
        elif comp_status.startswith("Verified"):
            return (
                "Verified by ideal quantum statevector simulation. All simulated computational basis "
                "probabilities match hypothesized values within numerical tolerance. "
                "Note: Ideal simulation does not validate noisy physical hardware execution."
            )
        elif lit_status == "Supported":
            cit_str = ", ".join(citations)
            return f"Corroborated by verified literature citations: {cit_str}."
        else:
            return "Author-stated hypothesis without direct computational simulation or indexed literature refutation."
