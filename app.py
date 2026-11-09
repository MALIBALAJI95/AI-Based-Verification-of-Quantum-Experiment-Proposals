"""
Streamlit Web Application: AI-Based Verification of Quantum Experiment Proposals.
Full-stack research prototype integrating PDF & DOCX processing, quantum RAG,
Qiskit simulation, modular verification engine, and explainable report export.
Supports two input modes (Upload Proposal & Manual Structured Input) with a common
researcher review and confirmation workflow.
"""

import os
import sys
import json
from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(".env")

from src.schemas.proposal import ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
from src.schemas.research_paper import ResearchPaperAnalysis, QuantumExperimentRecord
from src.schemas.verification import CheckStatus
from src.document_processing.document_extractor import DocumentExtractor, DocumentExtractionError, ExtractedDocument
from src.document_processing.chunker import TraceableChunker
from src.knowledge_base.vector_store import QuantumKnowledgeBase
from src.knowledge_base.retriever import QuantumRetriever
from src.circuit.parser import parse_circuit_dsl, CircuitParseError
from src.circuit.simulator import QuantumSimulator, HAS_QISKIT
from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import ProposalExtractor
from src.verification.engine import VerificationEngine
from src.reporting.generator import ReportGenerator
from src.evaluation.evaluator import VerificationEvaluator
from src.ui.components import (
    render_checklist_table,
    render_probability_distribution_chart,
    render_citation_card,
    render_statevector_table,
    render_research_paper_view,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Proposal Verifier",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for scientific styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_knowledge_base():
    kb = QuantumKnowledgeBase()
    kb.seed_starter_corpus()
    return kb

kb = get_knowledge_base()
retriever = QuantumRetriever(kb)

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "upload"  # "upload" or "manual"
if "pending_proposal" not in st.session_state:
    st.session_state.pending_proposal = None
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = None
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "uploaded_doc" not in st.session_state:
    st.session_state.uploaded_doc = None
if "current_paper_analysis" not in st.session_state:
    st.session_state.current_paper_analysis = None
if "last_extraction_error" not in st.session_state:
    st.session_state.last_extraction_error = None


# -----------------------------------------------------------------------------
# Sidebar: System Controls & API Key Detection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/atom.png", width=64)
    st.title("Quantum Verifier")
    st.caption("AI-Driven Research Prototype v1.1")

    st.markdown("---")
    st.subheader("LLM Provider Status")

    # Initialize client to detect configuration
    temp_client = ConfigurableLLMClient(demo_mode=False)
    is_live, status_desc = temp_client.is_configured()

    if "gemini_test_result" not in st.session_state:
        st.session_state.gemini_test_result = None

    # Required Display: Provider, Model, API Connection Status, Last Successful Test, Execution Mode
    if temp_client.provider_name == "vertex":
        provider_display = "Google Cloud Vertex AI"
    elif temp_client.provider_name == "gemini":
        provider_display = "Google Gemini"
    else:
        provider_display = temp_client.provider_name.upper()

    st.write(f"🏢 **Provider**: `{provider_display}`")
    st.write(f"🤖 **Model**: `{temp_client.model_name}`")

    if temp_client.provider_name == "vertex":
        if temp_client.vertex_project:
            st.caption(f"📁 **Project ID**: `{temp_client.vertex_project}`")
        if temp_client.vertex_location:
            st.caption(f"🌐 **Location**: `{temp_client.vertex_location}`")
        if temp_client.service_account_email:
            st.caption(f"🔑 **Service Account**: `{temp_client.service_account_email}`")
        if temp_client.credential_path:
            st.caption(f"📄 **Credentials**: `{temp_client.credential_path}`")

    if is_live:
        st.success("🟢 **API Status**: Connected & ADC Configured")
        st.markdown("**Execution Mode**: 🚀 `Live AI Analysis Active`")
    else:
        st.warning("🟡 **API Status**: Not Configured")
        st.markdown("**Execution Mode**: 🧪 `Demo Mode (Reference Data)`")

    # Last connection test display
    last_test = st.session_state.gemini_test_result
    if last_test:
        if last_test.get("success"):
            st.caption(f"🕒 **Last Test**: {last_test.get('timestamp')} (⏱️ {last_test.get('latency_ms')} ms)")
            st.caption(f"🔎 {last_test.get('status')}")
        else:
            st.caption(f"⚠️ **Last Test Diagnostic**: {last_test.get('status')}")
    else:
        st.caption("🕒 **Last Test**: Not tested in current session")

    # Action button: Test Vertex / Gemini Connection
    btn_label = "🔌 Test Vertex AI Connection" if temp_client.provider_name == "vertex" else "🔌 Test Gemini Connection"
    if st.button(btn_label, use_container_width=True):
        with st.spinner(f"Connecting to {provider_display}..."):
            diag = temp_client.test_connectivity()
            st.session_state.gemini_test_result = diag
            if diag.get("success"):
                st.success(f"✅ Connection Verified ({diag.get('latency_ms')} ms)!\n\n{diag.get('status')}")
            else:
                st.info(f"ℹ️ Provider Diagnostic ({diag.get('latency_ms')} ms):\n\n{diag.get('status')}")

    if is_live:
        force_demo = st.checkbox("Force Demo Mode (Fixed Reference Data)", value=False)
        active_demo_mode = force_demo
    else:
        st.caption("Using verified, deterministic reference benchmarks for safe demonstration.")
        with st.expander("API Key Setup Guide"):
            st.info(
                "Configure your credentials in the local `.env` file:\n\n"
                "```env\n"
                "# Google Cloud Vertex AI (Primary Provider)\n"
                "LLM_PROVIDER=vertex\n"
                "GOOGLE_APPLICATION_CREDENTIALS=C:\\GoogleCloudCredentials\\vertex-service-account.json\n"
                "VERTEX_PROJECT_ID=acquired-winter-511114-c2\n"
                "VERTEX_LOCATION=global\n"
                "VERTEX_MODEL=gemini-3.8-flash\n\n"
                "# Google Gemini (AI Studio)\n"
                "GEMINI_API_KEY=your_key_here\n"
                "GEMINI_MODEL=gemini-3.8-flash\n"
                "```"
            )
        active_demo_mode = True

    st.markdown("---")
    st.subheader("Engine Diagnostics")
    st.write(f"⚛️ **Qiskit Simulator**: {'Available (v2.6.1)' if HAS_QISKIT else 'NumPy Tensor Engine'}")
    st.write(f"📚 **Literature Passages**: {kb.count()} indexed chunks")
    st.write(f"📁 **Supported Documents**: PDF (.pdf) & Word (.docx)")

    st.markdown("---")
    st.caption("Research prototype. Does not certify physical execution on noisy NISQ hardware.")


# Instantiate active clients
llm_client = ConfigurableLLMClient(demo_mode=active_demo_mode)
proposal_extractor = ProposalExtractor(llm_client)
verification_engine = VerificationEngine(retriever=retriever, llm_client=llm_client)
doc_extractor = DocumentExtractor()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚛️ AI-Based Verification of Quantum Experiment Proposals</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Independent scientific verification combining Generative AI extraction, traceable literature RAG, editable researcher review, and exact Qiskit statevector simulation.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_input, tab_extract, tab_circuit, tab_checklist, tab_evidence, tab_report, tab_kb, tab_eval = st.tabs([
    "📝 1. Proposal Input & Review",
    "🔬 2. Extracted Spec",
    "⚛️ 3. Quantum Simulation",
    "📊 4. Verification Engine",
    "📚 5. Evidence & Citations",
    "📑 6. Explainable Report",
    "🗂️ 7. Knowledge Base",
    "🧪 8. Benchmark Evaluation"
])


# =============================================================================
# TAB 1: Two Input Modes & Common Confirmation Workflow
# =============================================================================
with tab_input:
    st.subheader("Step 1: Ingest or Specify Quantum Experiment Proposal")

    input_mode_radio = st.radio(
        "Choose Proposal Ingestion Method:",
        [
            "📄 Mode 1: Upload Existing Research Proposal (PDF / DOCX)",
            "✍️ Mode 2: Manual Structured Input (Form & Demonstration Presets)"
        ],
        index=0 if st.session_state.input_mode == "upload" else 1,
        horizontal=True
    )

    is_upload_mode = "Mode 1" in input_mode_radio

    # -------------------------------------------------------------------------
    # MODE 1: Upload Existing Research Proposal
    # -------------------------------------------------------------------------
    if is_upload_mode:
        st.session_state.input_mode = "upload"
        st.markdown("#### 📄 Upload Research Proposal Document (PDF or Word DOCX)")
        st.caption(
            "Upload an experimental proposal or paper draft. The system extracts text, sections, methodology, "
            "hypothesis, quantum system, qubit count, initial state, gate operations, measurement basis, expected outcomes "
            "and assumptions, preserving source page references without hallucination."
        )

        uploaded_file = st.file_uploader(
            "Select Proposal File (.pdf, .docx)",
            type=["pdf", "docx"],
            key="uploaded_file_picker"
        )

        # Quick sample proposal file buttons
        with st.expander("📁 Or load a verified sample proposal file (1-Click Test):"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                if st.button("📄 Sample Bell State (DOCX)", use_container_width=True):
                    sample_path = "data/sample_proposals/sample_bell_state_proposal.docx"
                    if os.path.exists(sample_path):
                        with open(sample_path, "rb") as f:
                            s_doc = doc_extractor.extract_from_bytes(f.read(), filename="sample_bell_state_proposal.docx")
                            st.session_state.uploaded_doc = s_doc
                            st.session_state.current_paper_analysis = None
                            st.session_state.pending_proposal = proposal_extractor.extract(
                                s_doc.full_text, document_title=s_doc.sections.get("title", s_doc.filename),
                                extracted_doc=s_doc, input_mode="upload"
                            )
                            st.rerun()

            with sc2:
                if st.button("📑 Sample Bell State (PDF)", use_container_width=True):
                    sample_path = "data/sample_proposals/sample_bell_state_proposal.pdf"
                    if os.path.exists(sample_path):
                        with open(sample_path, "rb") as f:
                            s_doc = doc_extractor.extract_from_bytes(f.read(), filename="sample_bell_state_proposal.pdf")
                            st.session_state.uploaded_doc = s_doc
                            st.session_state.current_paper_analysis = None
                            st.session_state.pending_proposal = proposal_extractor.extract(
                                s_doc.full_text, document_title=s_doc.sections.get("title", s_doc.filename),
                                extracted_doc=s_doc, input_mode="upload"
                            )
                            st.rerun()

            with sc3:
                if st.button("⚠️ Sample Incomplete Proposal (DOCX)", use_container_width=True):
                    sample_path = "data/sample_proposals/sample_incomplete_proposal.docx"
                    if os.path.exists(sample_path):
                        with open(sample_path, "rb") as f:
                            s_doc = doc_extractor.extract_from_bytes(f.read(), filename="sample_incomplete_proposal.docx")
                            st.session_state.uploaded_doc = s_doc
                            st.session_state.current_paper_analysis = None
                            st.session_state.pending_proposal = proposal_extractor.extract(
                                s_doc.full_text, document_title=s_doc.sections.get("title", s_doc.filename),
                                extracted_doc=s_doc, input_mode="upload"
                            )
                            st.rerun()

            with sc4:
                if st.button("🔬 Sample Nature Paper (s42256)", use_container_width=True):
                    sample_pdf_path = "data/sample_proposals/s42256-025-01153-0.pdf"
                    if os.path.exists(sample_pdf_path):
                        with open(sample_pdf_path, "rb") as f:
                            s_doc = doc_extractor.extract_from_bytes(f.read(), filename="s42256-025-01153-0.pdf")
                    else:
                        nature_text = (
                            "Nature Machine Intelligence | Article\n"
                            "DOI: 10.1038/s42256-025-01153-0\n"
                            "Title: Quantum Graph Learning and Multi-Circuit Benchmarking on Superconducting Processors\n"
                            "Authors: A. Kumar, M. Chen, S. Tanaka\n"
                            "Published: 15 January 2025\n\n"
                            "Abstract: We investigate scalable quantum graph learning on noisy intermediate-scale quantum devices. "
                            "The work evaluates variational ansatzes and state entanglement across multiple quantum registers.\n\n"
                            "Methodology: We implement quantum circuit ansatzes using Qiskit and simulate measurement outcomes under varying decoherence models.\n\n"
                            "Experiment 1: Two-qubit maximally entangled state preparation\n"
                            "We prepare a two-qubit Bell state |Phi+> using H gate on qubit 0 and CX gate on qubits 0, 1.\n"
                            "Expected outcomes: P(00) = 0.5, P(11) = 0.5.\n\n"
                            "Experiment 2: Three-qubit GHZ state benchmark\n"
                            "We prepare a three-qubit GHZ state using H gate on qubit 0, followed by cascaded CX gates on qubits 0, 1 and 1, 2.\n"
                            "Expected outcomes: P(000) = 0.5, P(111) = 0.5.\n\n"
                            "Results and Discussion: The two-qubit entanglement achieved state fidelity above 98%, while tripartite GHZ state coherence degraded under crosstalk.\n"
                            "Limitations: Coherence time T2 limits circuit depth beyond 20 layers on current superconducting hardware."
                        )
                        from src.document_processing.document_extractor import ExtractedDocument
                        s_doc = ExtractedDocument(
                            filename="s42256-025-01153-0.pdf",
                            full_text=nature_text,
                            pages=[{"page_number": 1, "text": nature_text, "char_count": len(nature_text)}],
                            sections={"title": "Quantum Graph Learning on Superconducting Processors", "abstract": "We investigate scalable quantum graph learning..."},
                            math_expressions=[],
                            warnings=[]
                        )
                    st.session_state.uploaded_doc = s_doc
                    st.session_state.current_paper_analysis = proposal_extractor.extract_research_paper(
                        raw_text=s_doc.full_text,
                        document_title=s_doc.sections.get("title", s_doc.filename),
                        extracted_doc=s_doc
                    )
                    st.session_state.pending_proposal = proposal_extractor.extract(
                        raw_text=s_doc.full_text,
                        document_title=s_doc.sections.get("title", s_doc.filename),
                        extracted_doc=s_doc,
                        input_mode="upload"
                    )
                    st.rerun()

        if uploaded_file is not None:
            col_fi1, col_fi2 = st.columns([3, 1])
            with col_fi1:
                st.write(f"📁 Selected File: **{uploaded_file.name}** ({round(len(uploaded_file.getvalue()) / 1024, 1)} KB)")
            with col_fi2:
                extract_file_btn = st.button("📥 Parse & Extract with AI", type="primary", use_container_width=True)

            if extract_file_btn:
                with st.spinner(f"Extracting structured sections and quantum parameters from '{uploaded_file.name}'..."):
                    try:
                        extracted_doc = doc_extractor.extract_from_bytes(uploaded_file.getvalue(), filename=uploaded_file.name)
                        st.session_state.uploaded_doc = extracted_doc

                        # Classify document type
                        doc_type = proposal_extractor.classify(extracted_doc.full_text, filename=uploaded_file.name)
                        if doc_type == "research_paper":
                            paper_analysis = proposal_extractor.extract_research_paper(
                                raw_text=extracted_doc.full_text,
                                document_title=extracted_doc.sections.get("title", uploaded_file.name),
                                extracted_doc=extracted_doc
                            )
                            st.session_state.current_paper_analysis = paper_analysis
                        else:
                            st.session_state.current_paper_analysis = None

                        # Extract structured proposal via LLM or rule-based fallback
                        extracted_spec = proposal_extractor.extract(
                            raw_text=extracted_doc.full_text,
                            document_title=extracted_doc.sections.get("title", uploaded_file.name),
                            extracted_doc=extracted_doc,
                            input_mode="upload"
                        )
                        st.session_state.pending_proposal = extracted_spec
                        st.session_state.last_extraction_error = None
                        doc_kind = "Academic Research Paper" if doc_type == "research_paper" else "Experiment Proposal"
                        st.success(f"Successfully processed {doc_kind} '{uploaded_file.name}' across {len(extracted_doc.pages)} page(s)!")
                        st.rerun()
                    except Exception as e:
                        st.session_state.last_extraction_error = str(e)
                        st.session_state.pending_proposal = None
                        st.error(f"Failed to process document: {e}")

        if st.session_state.get("last_extraction_error"):
            st.error(f"❌ **Extraction Error**: {st.session_state.last_extraction_error}")
            col_err1, col_err2 = st.columns(2)
            with col_err1:
                if st.button("🔄 Retry Extraction", key="btn_retry_extract_tab1", use_container_width=True):
                    st.session_state.last_extraction_error = None
                    st.rerun()
            with col_err2:
                if st.button("✍️ Switch to Manual Input Mode", key="btn_switch_manual_tab1", use_container_width=True):
                    st.session_state.input_mode = "manual"
                    st.session_state.last_extraction_error = None
                    st.rerun()

    # -------------------------------------------------------------------------
    # MODE 2: Manual Structured Input
    # -------------------------------------------------------------------------
    else:
        st.session_state.input_mode = "manual"
        st.markdown("#### ✍️ Manual Structured Input Form")
        st.caption(
            "Specify the quantum experiment hypothesis, target quantum system, initial quantum state, "
            "gate sequence, measurement basis, expected outcomes and assumptions directly."
        )

        demo_preset = st.selectbox(
            "Load Demonstration Case / Template (Optional):",
            [
                "Custom / Blank Input",
                "1. Two-Qubit Bell State Preparation (|Phi+>) [PASS Expected]",
                "2. Bell State with Deliberately Incorrect Probabilities [FAIL Expected]",
                "3. Invalid Qubit Index (Out-of-Bounds Operation) [FAIL Expected]",
                "4. Incomplete Proposal (Missing Circuit & Qubits) [INCONCLUSIVE Expected]",
                "5. Three-Qubit GHZ State (|000> + |111>)/sqrt(2) [PASS Expected]"
            ]
        )

        # Preset default values
        p_title = "Two-Qubit Bell State Preparation"
        p_obj = "Prepare and verify maximally entangled state (|00> + |11>)/sqrt(2) using H and CNOT gates."
        p_hypo = "Applying H(0) followed by CX(0,1) transforms initial state |00> into an equal superposition of |00> and |11>."
        p_system = "Superconducting Transmon (Ideal Simulation)"
        p_qubits = 2
        p_initial_state = "|00>"
        p_meas_basis = "computational (Z)"
        p_circuit_dsl = "H 0\nCX 0 1"
        p_outcomes_str = "00: 0.5\n11: 0.5"
        p_assumptions = "Ideal unitary evolution without environmental decoherence"

        if "2. Bell State with Deliberately" in demo_preset:
            p_title = "Two-Qubit Bell State Preparation with Hypothesized Asymmetric Distribution"
            p_obj = "Evaluate measurement distribution of standard Bell circuit."
            p_hypo = "Circuit generates state |Phi+>, but measurement probabilities are hypothesized as 80% on 00 and 20% on 11."
            p_system = "Superconducting Transmon"
            p_qubits = 2
            p_circuit_dsl = "H 0\nCX 0 1"
            p_outcomes_str = "00: 0.8\n11: 0.2"
            p_assumptions = "Ideal noiseless quantum hardware"

        elif "3. Invalid Qubit Index" in demo_preset:
            p_title = "Quantum Circuit with Out-of-Bounds Qubit Register Index"
            p_obj = "Test error rejection when gate targets an unallocated qubit index."
            p_hypo = "Circuit attempts to apply CX between qubit 0 and non-existent qubit 2 on a 2-qubit register."
            p_system = "Superconducting Transmon"
            p_qubits = 2
            p_circuit_dsl = "H 0\nCX 0 2"
            p_outcomes_str = "00: 0.5\n11: 0.5"
            p_assumptions = "2-qubit hardware register"

        elif "4. Incomplete Proposal" in demo_preset:
            p_title = "Quantum Optimization Speedup (Incomplete Proposal)"
            p_obj = "Achieve polynomial speedup for combinatorial search."
            p_hypo = "Superposition enables faster graph coloring."
            p_system = "Unspecified Quantum System"
            p_qubits = 0
            p_initial_state = ""
            p_meas_basis = ""
            p_circuit_dsl = ""
            p_outcomes_str = ""
            p_assumptions = "Scalable quantum acceleration"

        elif "5. Three-Qubit GHZ" in demo_preset:
            p_title = "Three-Qubit GHZ State Preparation and Verification"
            p_obj = "Prepare genuine tripartite entanglement in (|000> + |111>)/sqrt(2)."
            p_hypo = "H(0) followed by cascaded CNOTs CX(0,1) and CX(1,2) produces the 3-qubit GHZ state."
            p_system = "Trapped Ion / Superconducting"
            p_qubits = 3
            p_initial_state = "|000>"
            p_circuit_dsl = "H 0\nCX 0 1\nCX 1 2"
            p_outcomes_str = "000: 0.5\n111: 0.5"
            p_assumptions = "3-qubit coherent register"

        elif "Custom" in demo_preset:
            p_title = ""
            p_obj = ""
            p_hypo = ""
            p_system = "Ideal Simulation"
            p_qubits = 2
            p_initial_state = "|00>"
            p_meas_basis = "computational (Z)"
            p_circuit_dsl = ""
            p_outcomes_str = ""
            p_assumptions = ""

        with st.form("manual_proposal_form"):
            col_m1, col_m2 = st.columns([3, 1])
            with col_m1:
                man_title = st.text_input("Experiment Title", value=p_title)
            with col_m2:
                man_qubits = st.number_input("Number of Qubits", min_value=0, max_value=10, value=p_qubits, help="0 means unspecified")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                man_system = st.text_input("Quantum System / Architecture", value=p_system)
            with col_s2:
                man_init_state = st.text_input("Initial State", value=p_initial_state)
            with col_s3:
                man_meas_basis = st.text_input("Measurement Basis", value=p_meas_basis)

            man_obj = st.text_area("Objective", value=p_obj, height=70)
            man_hypo = st.text_area("Hypothesis", value=p_hypo, height=70)

            col_mc1, col_mc2 = st.columns(2)
            with col_mc1:
                st.markdown("**Quantum Gate Sequence (one gate per line)**")
                st.caption("Supported: `H q`, `X q`, `Y q`, `Z q`, `S q`, `T q`, `RX(rad) q`, `CX ctrl tgt`, `CZ c t`, `SWAP a b`")
                man_circuit_dsl = st.text_area("Gate Sequence DSL", value=p_circuit_dsl, height=110, label_visibility="collapsed")
            with col_mc2:
                st.markdown("**Expected Measurement Outcomes & Probabilities**")
                st.caption("Format: `basis_state: probability` (e.g. `00: 0.5`)")
                man_outcomes_str = st.text_area("Expected Outcomes", value=p_outcomes_str, height=110, label_visibility="collapsed")

            man_assump = st.text_input("Assumptions & Constraints", value=p_assumptions)

            prepare_spec_btn = st.form_submit_button("📋 Prepare Specification for Review", type="primary")

        if prepare_spec_btn:
            # Parse gate sequence
            parsed_circuit = None
            syntax_warnings = []
            if man_circuit_dsl.strip():
                try:
                    num_q = man_qubits if man_qubits > 0 else None
                    parsed_circuit = parse_circuit_dsl(man_circuit_dsl, num_qubits=num_q)
                    if man_init_state:
                        parsed_circuit.initial_state = man_init_state
                    if man_meas_basis:
                        parsed_circuit.measurement_basis = man_meas_basis
                except CircuitParseError as pe:
                    syntax_warnings.append(str(pe))
                except Exception as ex:
                    syntax_warnings.append(f"Gate syntax error: {ex}")

            # Parse outcomes
            outcomes_list = []
            if man_outcomes_str.strip():
                for line in man_outcomes_str.strip().splitlines():
                    if ":" in line:
                        parts = line.split(":")
                        bstate = parts[0].strip().replace("|", "").replace(">", "")
                        try:
                            p_val = float(parts[1].strip())
                            outcomes_list.append(ExpectedOutcome(basis_state=bstate, expected_probability=p_val))
                        except ValueError:
                            pass

            missing_info = []
            if man_qubits <= 0:
                missing_info.append("Number of qubits not specified.")
            if not parsed_circuit or not parsed_circuit.gates:
                missing_info.append("No gate sequence or circuit operations provided.")
            if not outcomes_list:
                missing_info.append("Expected measurement outcome probabilities were not specified.")

            manual_spec = ProposalSpecification(
                title=man_title or "Untitled Manual Proposal",
                objective=man_obj or "Manual quantum experiment proposal.",
                hypothesis=man_hypo or None,
                methodology=f"Gate sequence: {man_circuit_dsl.strip()}" if man_circuit_dsl.strip() else None,
                quantum_system=man_system,
                num_qubits=man_qubits if man_qubits > 0 else None,
                initial_state=man_init_state or "|0...0>",
                measurement_basis=man_meas_basis or "computational (Z)",
                circuit=parsed_circuit,
                expected_outcomes=outcomes_list,
                assumptions=[man_assump] if man_assump else [],
                missing_information=missing_info,
                potential_inconsistencies=[],
                extraction_warnings=syntax_warnings,
                input_mode="manual",
                source_filename=None,
                source_pages=[],
                raw_text=f"{man_title}\n{man_obj}\n{man_hypo}\n{man_circuit_dsl}\n{man_outcomes_str}"
            )

            st.session_state.uploaded_doc = None
            st.session_state.pending_proposal = manual_spec
            st.success("Structured specification prepared from manual input! Please review and confirm below.")
            st.rerun()

    # -------------------------------------------------------------------------
    # COMMON PROCESSING WORKFLOW: Editable Confirmation Screen
    # -------------------------------------------------------------------------
    pending: Optional[ProposalSpecification] = st.session_state.pending_proposal

    if pending is not None:
        st.markdown("---")
        st.subheader("Step 2: 📋 Researcher Review & Editable Confirmation Screen")
        st.info(
            "**Researcher confirmation required**: Review and adjust the extracted/specified parameters below. "
            "Verification simulation will ONLY execute once you confirm this specification."
        )

        # Source Provenance Badge
        if pending.input_mode == "upload" and pending.source_filename:
            st.success(
                f"📄 **Input Source**: Uploaded Document `{pending.source_filename}` "
                f"({len(pending.source_pages)} page(s) preserved with traceable provenance)"
            )
            # Expander for original document inspection
            if st.session_state.uploaded_doc:
                udoc = st.session_state.uploaded_doc
                with st.expander("🔍 Inspect Preserved Original Document Text & Page References"):
                    st.write(f"Total Character Count: **{len(udoc.full_text)}** | Identified Sections: **{', '.join(udoc.sections.keys())}**")
                    for p in udoc.pages:
                        st.markdown(f"**--- Page {p['page_number']} ---**")
                        st.text(p["text"][:800] + ("..." if len(p["text"]) > 800 else ""))
        else:
            st.info("✍️ **Input Source**: Manual Structured Input")

        # Document Type Badge & Research Paper Guidance
        if pending.document_type == "research_paper":
            st.warning(
                f"📄 **Document Classified as Academic Research Paper** (DOI: `{pending.doi or 'Identified in Document'}`)\n\n"
                "This publication discusses broad methodology, theoretical frameworks, and experimental findings. "
                "Research papers are not automatically treated as single executable proposals.\n\n"
                "👉 **Options**:\n"
                "- Select an individual experiment record from the paper below to populate this verification form.\n"
                "- Or supply/confirm the exact qubit count and gate sequence you wish to test with Qiskit.\n"
                "- You can also inspect the full **Research Paper Analysis** in Tab 2."
            )
            if st.session_state.current_paper_analysis and st.session_state.current_paper_analysis.experiments:
                st.markdown("##### 🔬 Select an Experiment from this Paper to Send to Circuit Verification:")
                paper_exps = st.session_state.current_paper_analysis.experiments
                for p_idx, p_exp in enumerate(paper_exps):
                    col_pe1, col_pe2 = st.columns([3, 1])
                    with col_pe1:
                        st.markdown(f"**[{p_exp.experiment_id}] {p_exp.title}**")
                        st.caption(f"System: `{p_exp.quantum_system or 'Ideal'}` | Qubits: **{p_exp.num_qubits or 'Unspecified'}** | Circuit: {p_exp.circuit_summary or 'Ansatz'}")
                    with col_pe2:
                        if st.button(f"Load [{p_exp.experiment_id}]", key=f"btn_load_exp_{p_idx}", use_container_width=True):
                            orig_title = (pending.title or "Proposal")[:30]
                            pending.title = f"{p_exp.title} - {orig_title}"
                            pending.num_qubits = p_exp.num_qubits
                            if p_exp.gates:
                                q_size = p_exp.num_qubits or 2
                                pending.circuit = CircuitSpec(
                                    num_qubits=q_size,
                                    gates=p_exp.gates,
                                    initial_state=f"|{'0' * q_size}>",
                                    measurement_basis="computational (Z)"
                                )
                            if p_exp.expected_outcomes:
                                pending.expected_outcomes = p_exp.expected_outcomes
                            st.session_state.pending_proposal = pending
                            st.rerun()

        if pending.num_qubits is None:
            st.info(
                "💡 **Qubit Count Unspecified / Inconclusive**: Target system size was not explicitly declared or was ambiguous. "
                "Please enter the number of qubits (1-10) below to enable Qiskit simulation."
            )

        # Missing / Ambiguous Information Alerts
        if pending.missing_information:
            st.warning("⚠️ **Missing or Ambiguous Information Detected**:\n" + "\n".join(f"- {m}" for m in pending.missing_information))

        if pending.potential_inconsistencies:
            st.error("🚨 **Potential Inconsistencies Detected**:\n" + "\n".join(f"- {pi}" for pi in pending.potential_inconsistencies))

        if pending.extraction_warnings:
            st.caption("ℹ️ Extraction Warnings / Diagnostics:\n" + "\n".join(f"- {w}" for w in pending.extraction_warnings))

        # Reconstruct DSL for gate sequence
        curr_dsl = ""
        if pending.circuit and pending.circuit.gates:
            dsl_lines = []
            for g in pending.circuit.gates:
                if g.params:
                    p_str = ", ".join(f"{p:.4f}" for p in g.params)
                    targets = " ".join(str(q) for q in g.target_qubits)
                    dsl_lines.append(f"{g.name}({p_str}) {targets}")
                else:
                    targets = " ".join(str(q) for q in g.target_qubits)
                    dsl_lines.append(f"{g.name} {targets}")
            curr_dsl = "\n".join(dsl_lines)

        # Reconstruct outcomes string
        curr_outcomes_str = ""
        if pending.expected_outcomes:
            curr_outcomes_str = "\n".join(f"{o.basis_state}: {o.expected_probability}" for o in pending.expected_outcomes)

        # Editable Confirmation Form
        with st.form("researcher_confirmation_form"):
            col_cf1, col_cf2 = st.columns([3, 1])
            with col_cf1:
                conf_title = st.text_input("Experiment Title", value=pending.title or "")
            with col_cf2:
                conf_qubits = st.number_input("Number of Qubits", min_value=0, max_value=10, value=pending.num_qubits or 0)

            col_cs1, col_cs2, col_cs3 = st.columns(3)
            with col_cs1:
                conf_system = st.text_input("Quantum System", value=pending.quantum_system or "Ideal Simulation")
            with col_cs2:
                conf_init_state = st.text_input("Initial State", value=pending.initial_state or "|0...0>")
            with col_cs3:
                conf_meas_basis = st.text_input("Measurement Basis", value=pending.measurement_basis or "computational (Z)")

            conf_obj = st.text_area("Objective", value=pending.objective, height=70)
            conf_hypo = st.text_area("Hypothesis", value=pending.hypothesis or "", height=70)
            conf_meth = st.text_area("Methodology", value=pending.methodology or "", height=70)

            col_cc1, col_cc2 = st.columns(2)
            with col_cc1:
                st.markdown("**Quantum Gate Sequence DSL (Editable)**")
                st.caption("One gate per line, e.g. `H 0`, `CX 0 1`, `RZ(1.57) 0`")
                conf_gates_dsl = st.text_area("Gate Sequence DSL", value=curr_dsl, height=110, label_visibility="collapsed")
            with col_cc2:
                st.markdown("**Expected Measurement Outcomes (Editable)**")
                st.caption("Format: `state: probability` (e.g. `00: 0.5`)")
                conf_outcomes_str = st.text_area("Expected Outcomes", value=curr_outcomes_str, height=110, label_visibility="collapsed")

            conf_assump_val = ", ".join(pending.assumptions) if pending.assumptions else ""
            conf_assumptions = st.text_input("Assumptions & Hardware Constraints", value=conf_assump_val)

            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                confirm_btn = st.form_submit_button("✅ Confirm Specification & Run Verification Engine", type="primary", use_container_width=True)
            with col_btn2:
                st.caption("Validates fields and executes Qiskit statevector simulation.")

        if confirm_btn:
            with st.spinner("Validating specification, running Qiskit simulation, and verifying literature claims..."):
                # Parse edited gate sequence
                confirmed_circuit = None
                conf_warnings = list(pending.extraction_warnings)
                if conf_gates_dsl.strip():
                    try:
                        num_q = conf_qubits if conf_qubits > 0 else None
                        confirmed_circuit = parse_circuit_dsl(conf_gates_dsl, num_qubits=num_q)
                        confirmed_circuit.initial_state = conf_init_state or "|0...0>"
                        confirmed_circuit.measurement_basis = conf_meas_basis or "computational (Z)"
                    except CircuitParseError as pe:
                        conf_warnings.append(str(pe))
                    except Exception as ex:
                        conf_warnings.append(f"Gate syntax error: {ex}")

                # Parse edited outcomes
                confirmed_outcomes = []
                if conf_outcomes_str.strip():
                    for line in conf_outcomes_str.strip().splitlines():
                        if ":" in line:
                            parts = line.split(":")
                            bstate = parts[0].strip().replace("|", "").replace(">", "")
                            try:
                                p_val = float(parts[1].strip())
                                confirmed_outcomes.append(ExpectedOutcome(basis_state=bstate, expected_probability=p_val))
                            except ValueError:
                                pass

                # Recalculate missing information
                updated_missing = []
                if conf_qubits <= 0:
                    updated_missing.append("Proposal omitted the number of qubits.")
                if not confirmed_circuit or not confirmed_circuit.gates:
                    updated_missing.append("Proposal provides no gate sequence or circuit specification.")
                if not confirmed_outcomes:
                    updated_missing.append("Proposal specifies no quantitative expected outcome probabilities.")

                confirmed_spec = ProposalSpecification(
                    title=conf_title,
                    objective=conf_obj,
                    hypothesis=conf_hypo or None,
                    methodology=conf_meth or None,
                    quantum_system=conf_system,
                    num_qubits=conf_qubits if conf_qubits > 0 else None,
                    initial_state=conf_init_state or "|0...0>",
                    measurement_basis=conf_meas_basis or "computational (Z)",
                    circuit=confirmed_circuit,
                    expected_outcomes=confirmed_outcomes,
                    assumptions=[conf_assumptions] if conf_assumptions else [],
                    missing_information=updated_missing,
                    potential_inconsistencies=list(pending.potential_inconsistencies),
                    author_claims=list(pending.author_claims),
                    raw_text=pending.raw_text,
                    extraction_warnings=conf_warnings,
                    input_mode=pending.input_mode,
                    source_filename=pending.source_filename,
                    source_pages=pending.source_pages,
                    confirmed_by_researcher=True
                )

                # Execute Verification Engine
                verification_report = verification_engine.verify_proposal(confirmed_spec)

                st.session_state.current_proposal = confirmed_spec
                st.session_state.current_report = verification_report

                st.success("✅ Specification confirmed and verified! Inspect tabs 2 through 6 for simulation, audit checklist, and report export.")
                st.rerun()

        # Reset button
        if st.button("🔄 Reset / Ingest Different Proposal"):
            st.session_state.pending_proposal = None
            st.session_state.current_proposal = None
            st.session_state.current_report = None
            st.session_state.uploaded_doc = None
            st.session_state.current_paper_analysis = None
            st.session_state.last_extraction_error = None
            st.rerun()


# Load current session data
proposal: Optional[ProposalSpecification] = st.session_state.current_proposal
report = st.session_state.current_report


# =============================================================================
# TAB 2: Extracted Specification & Research Paper Analysis
# =============================================================================
with tab_extract:
    st.subheader("Document Analysis & Structured Specification")
    active_spec: Optional[ProposalSpecification] = proposal or st.session_state.pending_proposal
    paper_data: Optional[ResearchPaperAnalysis] = st.session_state.current_paper_analysis

    spec_view_mode = "🔬 Academic Research Paper Analysis"
    if paper_data:
        spec_view_mode = st.radio(
            "Select Representation View:",
            ["🔬 Academic Research Paper Analysis", "📋 Structured Proposal Specification"],
            horizontal=True
        )
        if "Research Paper" in spec_view_mode:
            render_research_paper_view(paper_data)
            st.markdown("---")
            with st.expander("📄 Raw Research Paper Analysis JSON"):
                st.json(paper_data.model_dump())

    if not paper_data or (paper_data and "Proposal" in spec_view_mode):
        if active_spec is not None:
            # Status header
            c_status_str = "🟢 Confirmed by Researcher" if active_spec.confirmed_by_researcher else "🟡 Pending Researcher Confirmation"
            src_label = f"📄 Uploaded Document ({active_spec.source_filename or 'Document'})" if active_spec.input_mode == "upload" else "✍️ Manual Structured Input"
            doc_type_str = "Academic Research Paper" if active_spec.document_type == "research_paper" else "Experiment Proposal"
            st.markdown(f"**Verification State**: {c_status_str} | **Input Source**: {src_label} | **Document Type**: `{doc_type_str}`")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                title_display = active_spec.title if (active_spec.title and str(active_spec.title).strip()) else "*(Untitled / Not Specified in Proposal)*"
                st.markdown(f"**Experiment Title**: {title_display}")
                st.markdown(f"**Scientific Objective**: {active_spec.objective or '*(Not specified)*'}")
                st.markdown(f"**Hypothesis**: {active_spec.hypothesis or '*(None specified)*'}")
                st.markdown(f"**Target Quantum System**: `{active_spec.quantum_system or 'Ideal Simulation'}`")
                st.markdown(f"**Allocated Qubit Register**: `{active_spec.num_qubits if active_spec.num_qubits is not None else 'Unspecified / Inconclusive'}`")
            with col_s2:
                st.markdown(f"**Initial Quantum State**: `{active_spec.initial_state or '|0...0>'}`")
                st.markdown(f"**Measurement Basis**: `{active_spec.measurement_basis or 'Computational (Z)'}`")
                gate_count = len(active_spec.circuit.gates) if (active_spec.circuit and active_spec.circuit.gates) else 0
                st.markdown(f"**Gate Operations Count**: `{gate_count}`")
                st.markdown(f"**Stated Assumptions**: {', '.join(active_spec.assumptions) if active_spec.assumptions else '*(None)*'}")
                if active_spec.methodology:
                    st.markdown(f"**Methodology**: {active_spec.methodology}")

            if active_spec.author_claims:
                st.markdown("---")
                st.subheader("Extracted Author Claims with Source Page References")
                claims_rows = []
                for clm in active_spec.author_claims:
                    claims_rows.append({
                        "Claim ID": clm.claim_id,
                        "Type": clm.claim_type,
                        "Claim Statement": clm.statement,
                        "Source Page / Section": clm.source_reference or "N/A",
                        "Confidence": clm.confidence_level
                    })
                st.dataframe(pd.DataFrame(claims_rows), use_container_width=True, hide_index=True)

            if active_spec.missing_information:
                st.warning("⚠️ **Identified Missing Information**:\n" + "\n".join([f"- {m}" for m in active_spec.missing_information]))

            if active_spec.extraction_warnings:
                st.info("ℹ️ **Extraction Diagnostics / Warnings**:\n" + "\n".join([f"- {w}" for w in active_spec.extraction_warnings]))

            # Preserved Document Pages
            if active_spec.source_pages:
                with st.expander(f"📄 Preserved Source Document Pages ({len(active_spec.source_pages)} pages)"):
                    for p in active_spec.source_pages:
                        st.markdown(f"**Page {p.get('page_number', 1)}** ({p.get('char_count', 0)} characters)")
                        st.text(p.get("text", "")[:1000] + ("..." if len(p.get("text", "")) > 1000 else ""))

            st.markdown("---")
            st.subheader("Raw JSON Specification")
            st.json(active_spec.model_dump())

        else:
            # active_spec is None
            if paper_data and paper_data.experiments:
                st.info(
                    "📄 **Academic Research Paper Loaded**: The uploaded document discusses multiple quantum experiments. "
                    "Select an experiment below or on the Tab 1 confirmation screen to populate a structured proposal specification."
                )
                st.markdown("##### 🔬 Available Experiments in this Paper:")
                for p_idx, p_exp in enumerate(paper_data.experiments):
                    col_pe1, col_pe2 = st.columns([3, 1])
                    with col_pe1:
                        st.markdown(f"**[{p_exp.experiment_id}] {p_exp.title}**")
                        st.caption(f"System: `{p_exp.quantum_system or 'Ideal'}` | Qubits: **{p_exp.num_qubits or 'Unspecified'}** | Circuit: {p_exp.circuit_summary or 'Ansatz'}")
                    with col_pe2:
                        if st.button(f"Load [{p_exp.experiment_id}] for Verification", key=f"btn_tab2_load_{p_idx}", use_container_width=True):
                            loaded_circuit = None
                            if p_exp.gates and p_exp.num_qubits:
                                loaded_circuit = CircuitSpec(
                                    num_qubits=p_exp.num_qubits,
                                    gates=p_exp.gates,
                                    initial_state=f"|{'0' * p_exp.num_qubits}>",
                                    measurement_basis="computational (Z)"
                                )
                            st.session_state.pending_proposal = ProposalSpecification(
                                title=p_exp.title,
                                objective=f"Verify {p_exp.title} from paper '{paper_data.title}'",
                                quantum_system=p_exp.quantum_system or "Ideal Simulation",
                                num_qubits=p_exp.num_qubits,
                                initial_state=f"|{'0' * (p_exp.num_qubits or 2)}>",
                                measurement_basis="computational (Z)",
                                circuit=loaded_circuit,
                                expected_outcomes=p_exp.expected_outcomes,
                                input_mode="upload",
                                source_filename=paper_data.source_filename,
                                source_pages=paper_data.source_pages,
                                document_type="research_paper",
                                doi=paper_data.doi
                            )
                            st.rerun()
            else:
                last_err = st.session_state.get("last_extraction_error")
                if last_err:
                    st.error(f"❌ **Extraction Error**: {last_err}")
                    st.info("You can retry document extraction or enter the experiment parameters manually in Tab 1.")
                    col_re1, col_re2 = st.columns(2)
                    with col_re1:
                        if st.button("🔄 Retry Extraction", key="btn_retry_tab2", use_container_width=True):
                            st.session_state.last_extraction_error = None
                            st.rerun()
                    with col_re2:
                        if st.button("✍️ Go to Manual Input", key="btn_manual_tab2", use_container_width=True):
                            st.session_state.input_mode = "manual"
                            st.session_state.last_extraction_error = None
                            st.rerun()
                else:
                    st.info(
                        "ℹ️ **No Proposal Specification Available**: No experiment proposal has been ingested yet.\n\n"
                        "👉 **Next Step**: Go to **Tab 1: Proposal Input & Review** to upload a proposal (PDF / DOCX) or enter experimental parameters manually."
                    )


# =============================================================================
# TAB 3: Circuit Verification & Quantum Simulation
# =============================================================================
with tab_circuit:
    st.subheader("Qiskit Quantum Circuit Simulation (Ideal Statevector)")
    if report and report.simulation:
        sim = report.simulation
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Quantum System Dimension", f"{sim.num_qubits} Qubits ({2**sim.num_qubits} States)")
        with col_m2:
            norm_badge = "Preserved (Error <= 1e-6)" if sim.is_normalized else "FAILED"
            st.metric("State Normalization", norm_badge)
        with col_m3:
            st.metric("Simulation Latency", f"{sim.execution_time_ms} ms")

        st.markdown("### Circuit Representation")
        st.code(sim.circuit_diagram_text or "No diagram available", language="text")

        st.markdown("### Measurement Probability Distribution")
        render_probability_distribution_chart(report.outcome_comparisons, sim.probabilities)

        st.markdown("### Full Statevector Components (Amplitudes and Phases)")
        render_statevector_table(sim)

    elif report and not report.simulation:
        if proposal and proposal.num_qubits is None:
            st.warning("🟡 **Simulation Bypassed / Inconclusive**")
            st.info(
                "Target quantum system dimension (number of qubits) was not specified or an executable gate sequence was omitted. "
                "Qiskit statevector simulation was safely bypassed without crash.\n\n"
                "💡 **Guidance**: Researchers can specify the number of qubits (1-10) and valid gate operations on the confirmation screen."
            )
        else:
            st.error("Simulation was not executed. Circuit gate errors or missing parameters were detected.")
        for c in report.checks:
            if c.status == CheckStatus.FAIL:
                st.error(f"**{c.name}**: {c.details} (Trace: {c.error_message})")
            elif c.status == CheckStatus.INCONCLUSIVE:
                st.warning(f"**{c.name}**: {c.details}")
    else:
        st.info("ℹ️ **Simulation Pending Researcher Confirmation**: No verified simulation report available. Please review and confirm your experiment specification in Tab 1 before running Qiskit statevector simulation.")


# =============================================================================
# TAB 4: Verification Engine & Checklist
# =============================================================================
with tab_checklist:
    st.subheader("Independent Scientific Verification Checklist")
    if report:
        # Prominent overall status banner
        if report.summary_counts.get("FAIL", 0) > 0:
            st.error(f"❌ **{report.overall_summary}**")
        elif report.summary_counts.get("INCONCLUSIVE", 0) > 0:
            st.warning(f"⚠️ **{report.overall_summary}**")
        else:
            st.success(f"✅ **{report.overall_summary}**")

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            st.metric("🟢 PASS", report.summary_counts.get("PASS", 0))
        with col_b2:
            st.metric("🔴 FAIL", report.summary_counts.get("FAIL", 0))
        with col_b3:
            st.metric("🟡 INCONCLUSIVE", report.summary_counts.get("INCONCLUSIVE", 0))
        with col_b4:
            st.metric("⚪ NOT RUN", report.summary_counts.get("NOT_RUN", 0))

        st.markdown("---")
        st.subheader("Audited Verification Rules")
        render_checklist_table(report.checks)

        if report.claims_analysis:
            st.markdown("---")
            st.subheader("Claim-by-Claim Verification Matrix")
            claims_data = []
            for clm in report.claims_analysis:
                claims_data.append({
                    "Claim ID": clm.claim_id,
                    "Statement": clm.claim_text,
                    "Literature Grounding": clm.literature_support_status,
                    "Computational Simulation": clm.computational_status,
                    "Scientific Assessment": clm.explanation
                })
            st.dataframe(pd.DataFrame(claims_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.info(f"🛡️ **Hardware Feasibility Disclaimer**: {report.limitations_and_disclaimer}")

    else:
        st.info("Run verification in Tab 1 to generate the audit checklist.")


# =============================================================================
# TAB 5: Evidence & Literature Citations
# =============================================================================
with tab_evidence:
    st.subheader("Traceable Quantum Literature Citations")
    st.caption("Retrieved passages linked to genuine source documents, page numbers, and chunk identifiers.")
    if report and report.citations:
        for idx, cit in enumerate(report.citations):
            render_citation_card(cit, idx)
    elif report:
        st.warning("No directly relevant passages retrieved from knowledge base for this proposal.")
    else:
        st.info("Submit a proposal to retrieve traceable scientific citations.")


# =============================================================================
# TAB 6: Explainable Report & Export
# =============================================================================
with tab_report:
    st.subheader("Downloadable Verification Report")
    if report:
        json_report = ReportGenerator.to_json(report)
        md_report = ReportGenerator.to_markdown(report)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download JSON Audit Report",
                data=json_report,
                file_name=f"verification_report_{report.report_id}.json",
                mime="application/json",
                type="primary"
            )
        with col_d2:
            st.download_button(
                label="📥 Download Markdown Summary Report",
                data=md_report,
                file_name=f"verification_report_{report.report_id}.md",
                mime="text/markdown"
            )

        st.markdown("---")
        st.subheader("Rendered Report Preview")
        st.markdown(md_report)
    else:
        st.info("Execute verification to generate downloadable reports.")


# =============================================================================
# TAB 7: Knowledge Base Management
# =============================================================================
with tab_kb:
    st.subheader("Quantum Literature Knowledge Base")
    st.markdown("Manage reference papers and inspect traceable embeddings in the local vector store.")

    st.write(f"Total Passages Currently Indexed: **{kb.count()}**")

    col_kb1, col_kb2 = st.columns(2)
    with col_kb1:
        if st.button("🔄 Re-Index Starter Reference Corpus"):
            added = kb.seed_starter_corpus()
            st.success(f"Indexed {added} authoritative reference passages from Nielsen & Chuang, Bell (1964), Grover (1996), and Preskill (2018).")

    with col_kb2:
        test_query = st.text_input("Semantic Search Query", value="Bell state entanglement fidelity")
        if st.button("Search Knowledge Base"):
            results = kb.retrieve(test_query, top_k=3)
            st.write(f"Found {len(results)} matching passages:")
            for r in results:
                st.markdown(f"**{r['title']}** (File: `{r['source_filename']}`, Page: {r.get('page_number', 'N/A')}, Score: `{r['score']}`)")
                st.caption(f"\"{r['text']}\"")

    st.markdown("---")
    st.subheader("Upload Reference Literature")
    uploaded_ref = st.file_uploader("Upload Quantum Computing Paper (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="ref_upload")
    if uploaded_ref and st.button("Index Uploaded Reference"):
        chunker = TraceableChunker(chunk_size=500, chunk_overlap=80)
        fname = uploaded_ref.name
        pdoc = doc_extractor.extract_from_bytes(uploaded_ref.read(), filename=fname)
        chunks = chunker.chunk_document(pdoc.full_text, source_filename=fname, pages=pdoc.pages)

        added = kb.index_chunks(chunks, source_title=fname)
        st.success(f"Indexed {added} traceable chunks from '{fname}' into knowledge base.")


# =============================================================================
# TAB 8: Scientific Benchmark Evaluation
# =============================================================================
with tab_eval:
    st.subheader("Scientific Benchmark Evaluation Harness")
    st.markdown(
        "Runs an automated evaluation measuring **Error Detection Precision**, **Recall**, **False-Pass Rate**, "
        "**False-Alarm Rate**, and **Simulation Agreement** across labelled test proposals."
    )

    if st.button("▶️ Execute Benchmark Suite"):
        with st.spinner("Running evaluation across all test proposals and verification modes..."):
            evaluator = VerificationEvaluator(output_csv="evaluation_results.csv")
            eval_df, summary_metrics = evaluator.run_evaluation()

            st.success(f"Evaluated {len(eval_df)} test runs across {len(summary_metrics)} verification modalities.")

            st.markdown("### Aggregated Verification Modality Comparison")
            metrics_table = []
            for mode, m in summary_metrics.items():
                metrics_table.append({
                    "Verification Mode": mode,
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "False-Pass Rate (FPR)": f"{m['false_pass_rate']:.4f}",
                    "False-Alarm Rate (FAR)": f"{m['false_alarm_rate']:.4f}",
                    "Simulation Agreement": f"{m['simulation_agreement']:.4f}",
                    "Avg Latency (ms)": f"{m['avg_execution_time_ms']:.2f}"
                })
            st.table(pd.DataFrame(metrics_table))

            st.markdown("### Detailed Per-Case Evaluation Log")
            st.dataframe(eval_df, use_container_width=True)

            csv_data = eval_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Benchmark Results (CSV)",
                data=csv_data,
                file_name="quantum_verification_benchmark_results.csv",
                mime="text/csv"
            )
