"""
Report Generator for Quantum Experiment Proposal Verification.
Exports structured JSON audit files and readable, publication-ready Markdown reports.
"""

import json
from typing import Dict, Any
from ..schemas.verification import VerificationReport, CheckStatus


class ReportGenerator:
    """
    Produces publication-grade reports in JSON and Markdown formats.
    """

    @staticmethod
    def to_json(report: VerificationReport, indent: int = 2) -> str:
        """Export full verification report to clean JSON."""
        return report.model_dump_json(indent=indent)

    @staticmethod
    def to_markdown(report: VerificationReport) -> str:
        """Export comprehensive readable Markdown report."""
        lines = []

        # Header
        lines.append(f"# Quantum Experiment Proposal Verification Report")
        lines.append(f"**Report ID**: `{report.report_id}`  ")
        lines.append(f"**Proposal**: {report.proposal_title}  ")
        lines.append(f"**Verification Timestamp**: `{report.timestamp}`  ")
        lines.append("")

        # Software Environment
        lines.append("## 1. Execution Environment")
        for k, v in report.software_environment.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        lines.append("")

        # Executive Summary
        lines.append("## 2. Executive Verification Summary")
        status_emoji = "✅" if report.summary_counts.get("FAIL", 0) == 0 and report.summary_counts.get("INCONCLUSIVE", 0) == 0 else ("❌" if report.summary_counts.get("FAIL", 0) > 0 else "⚠️")
        lines.append(f"> {status_emoji} **{report.overall_summary}**")
        lines.append("")
        lines.append(
            f"**Check Results**: "
            f"🟢 PASS: `{report.summary_counts.get('PASS', 0)}` | "
            f"🔴 FAIL: `{report.summary_counts.get('FAIL', 0)}` | "
            f"🟡 INCONCLUSIVE: `{report.summary_counts.get('INCONCLUSIVE', 0)}` | "
            f"⚪ NOT RUN: `{report.summary_counts.get('NOT_RUN', 0)}`"
        )
        lines.append("")

        # Verification Checklist
        lines.append("## 3. Verification Checklist")
        lines.append("| ID | Check Name | Category | Status | Details |")
        lines.append("|---|---|---|---|---|")
        for c in report.checks:
            badge = "🟢 PASS" if c.status == CheckStatus.PASS else ("🔴 FAIL" if c.status == CheckStatus.FAIL else ("🟡 INCONCLUSIVE" if c.status == CheckStatus.INCONCLUSIVE else "⚪ NOT RUN"))
            clean_details = c.details.replace("\n", " ").replace("|", "/")
            lines.append(f"| `{c.check_id}` | {c.name} | `{c.category}` | **{badge}** | {clean_details} |")
        lines.append("")

        # Quantum Circuit & Simulation
        lines.append("## 4. Quantum Simulation Results")
        if report.simulation:
            sim = report.simulation
            lines.append(f"- **Register Size**: {sim.num_qubits} qubits (Hilbert Dimension: {2**sim.num_qubits})")
            lines.append(f"- **Normalization Preserved**: {'Yes' if sim.is_normalized else 'No'} (Error: `{sim.normalization_error:.2e}`)")
            lines.append(f"- **Simulation Execution Time**: `{sim.execution_time_ms} ms`")
            lines.append("")
            
            if sim.circuit_diagram_text:
                lines.append("### Circuit Diagram")
                lines.append("```text")
                lines.append(sim.circuit_diagram_text)
                lines.append("```")
                lines.append("")

            lines.append("### Simulated Statevector & Measurement Probabilities")
            lines.append("| Basis State | Real Amplitude | Imag Amplitude | Magnitude | Probability | Phase (rad) |")
            lines.append("|---|---|---|---|---|---|")
            for comp in sim.statevector_components:
                if comp.probability > 0.0001 or sim.num_qubits <= 3:
                    lines.append(f"| `|{comp.basis_state}>` | {comp.real:.4f} | {comp.imag:.4f} | {comp.magnitude:.4f} | **{comp.probability:.4f}** | {comp.phase_rad:.2f} |")
            lines.append("")
        else:
            lines.append("_No simulation executed. See checklist errors above._\n")

        # Outcome Comparisons
        lines.append("## 5. Expected vs Computed Outcome Comparison")
        if report.outcome_comparisons:
            lines.append("| Basis State | Hypothesized Prob | Computed Prob | Difference | Tolerance | Match? |")
            lines.append("|---|---|---|---|---|---|")
            for oc in report.outcome_comparisons:
                m_str = "✅ PASS" if oc.is_match else "❌ FAIL"
                lines.append(f"| `|{oc.basis_state}>` | {oc.expected_probability:.4f} | {oc.computed_probability:.4f} | {oc.absolute_difference:.4f} | {oc.tolerance:.4f} | **{m_str}** |")
            lines.append("")
            if report.total_variation_distance is not None:
                lines.append(f"**Total Variation Distance (TVD)**: `{report.total_variation_distance:.4f}`")
                lines.append("")
        else:
            lines.append("_No quantitative expected outcomes specified by author._\n")

        # Literature Evidence
        lines.append("## 6. Grounded Quantum Literature Evidence")
        if report.citations:
            for cit in report.citations:
                lines.append(f"### [{cit.citation_id}] {cit.source_title}")
                lines.append(f"- **Source File**: `{cit.filename}` (Page: {cit.page_number or 'N/A'}, Chunk: `{cit.chunk_id}`)")
                lines.append(f"- **Similarity Score**: `{cit.similarity_score:.4f}`")
                lines.append(f"> \"{cit.passage}\"")
                lines.append("")
        else:
            lines.append("_No matching literature citations retrieved._\n")

        # Claim Verification Analysis
        lines.append("## 7. Claim-by-Claim Scientific Audit")
        if report.claims_analysis:
            lines.append("| Claim ID | Category | Statement | Literature Support | Computational Check | Scientific Analysis |")
            lines.append("|---|---|---|---|---|---|")
            for clm in report.claims_analysis:
                clean_stmt = clm.claim_text.replace("|", "/")
                clean_exp = clm.explanation.replace("|", "/")
                lines.append(f"| `{clm.claim_id}` | `{clm.claim_category}` | {clean_stmt} | {clm.literature_support_status} | **{clm.computational_status}** | {clean_exp} |")
            lines.append("")

        # Detected Errors & Missing Info
        if report.detected_inconsistencies:
            lines.append("## 8. Detected Inconsistencies and Errors")
            for err in report.detected_inconsistencies:
                lines.append(f"- ❌ **Anomaly**: {err}")
            lines.append("")

        if report.missing_information:
            lines.append("## 9. Missing Information Requiring Clarification")
            for m in report.missing_information:
                lines.append(f"- ⚠️ **Missing**: {m}")
            lines.append("")

        # Recommendations & Next Steps
        lines.append("## 10. Recommended Next Steps")
        for step in report.recommended_next_steps:
            lines.append(f"1. {step}")
        lines.append("")

        # Disclaimer
        lines.append("## 11. Limitations and Research Disclaimer")
        lines.append(f"> {report.limitations_and_disclaimer}")
        lines.append("")

        return "\n".join(lines)
