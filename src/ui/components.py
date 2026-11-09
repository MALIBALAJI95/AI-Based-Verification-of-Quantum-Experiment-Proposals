"""
Streamlit UI Components for Quantum Experiment Verification.
Renders scientific audit badges, probability distribution charts,
traceable citation cards, and statevector tables.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
from ..schemas.verification import CheckStatus, VerificationCheck, LiteratureCitation, OutcomeComparison, SimulationOutput
from ..schemas.research_paper import ResearchPaperAnalysis


def render_status_badge(status: CheckStatus) -> str:
    """Render colored markdown badge for check statuses."""
    if status == CheckStatus.PASS:
        return "🟢 **PASS**"
    elif status == CheckStatus.FAIL:
        return "🔴 **FAIL**"
    elif status == CheckStatus.INCONCLUSIVE:
        return "🟡 **INCONCLUSIVE**"
    else:
        return "⚪ **NOT RUN**"


def render_checklist_table(checks: List[VerificationCheck]):
    """Display the verification checklist in a structured Streamlit table."""
    data = []
    for c in checks:
        badge = "PASS" if c.status == CheckStatus.PASS else ("FAIL" if c.status == CheckStatus.FAIL else ("INCONCLUSIVE" if c.status == CheckStatus.INCONCLUSIVE else "NOT RUN"))
        data.append({
            "Status": badge,
            "Check ID": c.check_id,
            "Check Name": c.name,
            "Category": c.category.value,
            "Details": c.details,
            "Error Trace": c.error_message or "-"
        })
    df = pd.DataFrame(data)
    
    # Custom styling
    def highlight_status(row):
        if row["Status"] == "PASS":
            return ["background-color: rgba(34, 197, 94, 0.15)"] * len(row)
        elif row["Status"] == "FAIL":
            return ["background-color: rgba(239, 68, 68, 0.20)"] * len(row)
        elif row["Status"] == "INCONCLUSIVE":
            return ["background-color: rgba(245, 158, 11, 0.15)"] * len(row)
        return [""] * len(row)

    styled_df = df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def render_probability_distribution_chart(comparisons: List[OutcomeComparison], computed_probs: Dict[str, float]):
    """Render interactive side-by-side comparison of hypothesized vs simulated probabilities."""
    chart_data = []

    if comparisons:
        for c in comparisons:
            chart_data.append({"Basis State": f"|{c.basis_state}>", "Probability": c.expected_probability, "Type": "Hypothesized (Expected)"})
            chart_data.append({"Basis State": f"|{c.basis_state}>", "Probability": c.computed_probability, "Type": "Computed (Simulation)"})
    else:
        for bstate, prob in computed_probs.items():
            if prob > 0.0001 or len(computed_probs) <= 4:
                chart_data.append({"Basis State": f"|{bstate}>", "Probability": prob, "Type": "Computed (Simulation)"})

    if chart_data:
        df = pd.DataFrame(chart_data)
        st.bar_chart(df, x="Basis State", y="Probability", color="Type", stack=False)


def render_citation_card(citation: LiteratureCitation, idx: int):
    """Render an individual traceable citation with verified provenance."""
    with st.expander(f"📄 [{citation.citation_id}] {citation.source_title}", expanded=(idx == 0)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Document File**: `{citation.filename}`")
            st.markdown(f"**Page Number**: {citation.page_number or 'Unspecified'} | **Chunk ID**: `{citation.chunk_id}`")
        with col2:
            st.metric("Similarity Score", f"{citation.similarity_score:.3f}")

        st.info(f"💬 \"{citation.passage}\"")
        if citation.relevance_comment:
            st.caption(f"Relevance: {citation.relevance_comment}")


def render_statevector_table(sim: SimulationOutput):
    """Render formatted state amplitudes and phases."""
    rows = []
    for c in sim.statevector_components:
        if c.probability > 0.0001 or sim.num_qubits <= 3:
            complex_str = f"{c.real:+.4f} {c.imag:+.4f}j"
            rows.append({
                "Basis State": f"|{c.basis_state}>",
                "Complex Amplitude": complex_str,
                "Magnitude |c|": f"{c.magnitude:.4f}",
                "Probability |c|²": f"{c.probability:.4f}",
                "Phase (rad)": f"{c.phase_rad:.2f}"
            })
    st.table(pd.DataFrame(rows))


def render_research_paper_view(paper: ResearchPaperAnalysis):
    """Render comprehensive structured view of an academic research paper."""
    st.markdown(f"### 📄 {paper.title}")
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        if paper.doi:
            st.markdown(f"🔗 **DOI**: [`{paper.doi}`](https://doi.org/{paper.doi})")
        if paper.publication_info:
            st.markdown(f"🏛️ **Publication**: {paper.publication_info}")
        if paper.authors:
            st.markdown(f"👥 **Authors**: {', '.join(paper.authors)}")
    with col_hdr2:
        st.info("🏷️ **Document Type**\nAcademic Research Paper")

    if paper.abstract:
        with st.expander("📝 Paper Abstract", expanded=True):
            st.write(paper.abstract)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Research Objective")
        st.write(paper.research_objective)

        st.markdown("#### ⚠️ Research Problem")
        st.write(paper.research_problem)

        st.markdown("#### ⚛️ Quantum Systems Discussed")
        for qs in paper.quantum_systems_discussed:
            st.markdown(f"- `{qs}`")

    with col2:
        st.markdown("#### ⚙️ Methodology")
        st.write(paper.methodology)

        st.markdown("#### 💻 Experimental & Simulation Setup")
        st.write(paper.experimental_setup or "Numerical quantum circuit simulation and state evolution.")

        st.markdown("#### 🧩 Algorithms & Frameworks")
        st.write(", ".join(f"`{af}`" for af in paper.algorithms_and_frameworks) if paper.algorithms_and_frameworks else "Quantum circuit model")

    st.markdown("---")
    st.subheader(f"🧪 Experiments & Benchmarks Discussed ({len(paper.experiments)} identified)")
    st.caption("Academic papers typically discuss several experimental evaluations or circuit configurations.")

    for exp in paper.experiments:
        with st.container(border=True):
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.markdown(f"**[{exp.experiment_id}] {exp.title}**")
                st.write(f"Architecture: `{exp.quantum_system or 'Ideal Simulation'}` | Qubits: **{exp.num_qubits if exp.num_qubits else 'Unspecified / Flexible'}**")
                if exp.circuit_summary:
                    st.caption(f"Circuit Summary: {exp.circuit_summary}")
                if exp.findings:
                    st.caption(f"Findings: {exp.findings}")
            with exp_col2:
                if exp.has_executable_circuit:
                    st.success("🟢 Executable Circuit Available")
                else:
                    st.info("⚪ Conceptual / Algorithmic")

    st.markdown("---")
    col_rf1, col_rf2 = st.columns(2)
    with col_rf1:
        st.markdown("#### 📊 Key Results & Findings")
        for r in paper.results_and_findings:
            st.markdown(f"- {r}")
    with col_rf2:
        st.markdown("#### 🛡️ Stated Limitations & Constraints")
        for lim in paper.limitations:
            st.markdown(f"- ⚠️ {lim}")

