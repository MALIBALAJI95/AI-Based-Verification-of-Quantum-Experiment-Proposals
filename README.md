# AI-Based Verification of Quantum Experiment Proposals

A research-grade verification prototype combining **Generative AI extraction**, **Quantum-Domain Retrieval-Augmented Generation (RAG)**, **Qiskit Statevector Simulation**, and an **Independent Scientific Verification Engine**.

Designed to assist quantum researchers, funding panels, and laboratory directors in vetting quantum experiment proposals, cross-referencing literature claims, detecting mathematical inconsistencies, simulating target quantum circuits, and exporting comprehensive audit reports.

---

## 1. System Architecture

```
AI-Based Verification of Quantum Experiment Proposals/
│
├── .streamlit/                   # Local Streamlit configuration (headless, ports, telemetry disabled)
│   ├── config.toml
│   └── credentials.toml
├── src/
│   ├── schemas/                  # Pydantic v2 data models
│   │   ├── proposal.py           # ProposalSpecification, CircuitSpec, GateOperation, ExpectedOutcome
│   │   └── verification.py       # VerificationCheck, CheckStatus, VerificationReport, LiteratureCitation
│   ├── document_processing/      # Non-hallucinatory document ingestion
│   │   ├── document_extractor.py # Unified polymorphic document extractor (PDF, DOCX, TXT)
│   │   ├── pdf_extractor.py      # PDF parsing, section segmentation, Dirac bra-ket math extraction
│   │   ├── docx_extractor.py     # Pure-Python OpenXML DOCX parser with explicit page-break detection
│   │   └── chunker.py            # Traceable overlapping chunker with page & chunk ID provenance
│   ├── knowledge_base/           # Literature vector store & RAG
│   │   ├── starter_corpus.py     # Verified reference literature (Nielsen & Chuang, Bell 1964, Grover 1996, Preskill 2018)
│   │   ├── vector_store.py       # ChromaDB + SentenceTransformers + Local Cosine Vector Fallback
│   │   └── retriever.py          # Semantic & lexical query retriever producing traceable citations
│   ├── circuit/                  # Computational quantum circuit simulation
│   │   ├── parser.py             # Circuit DSL parser, gate arity, and qubit bounds validation
│   │   ├── simulator.py          # Qiskit Statevector engine (with exact NumPy tensor fallback)
│   │   └── metrics.py            # TVD (Total Variation Distance), phase-invariant fidelity, outcome comparison
│   ├── llm/                      # Multi-provider LLM integration & prompt defense
│   │   ├── client.py             # Configurable client (OpenAI, Anthropic, Gemini, OpenAI-compatible, & Demo Mode)
│   │   ├── prompts.py            # Injection-defended system prompts (<UNTRUSTED_PROPOSAL_TEXT>)
│   │   ├── extractor.py          # Structured proposal parameter extraction into Pydantic models
│   │   └── analyzer.py           # Claim cross-examiner enforcing computational precedence
│   ├── verification/             # Independent verification engine
│   │   ├── rules.py              # Modular check rules (Qubit count, syntax, normalization, distributions, feasibility)
│   │   └── engine.py             # End-to-end verification pipeline (PASS, FAIL, INCONCLUSIVE, NOT RUN)
│   ├── reporting/                # Explainable audit report generation
│   │   └── generator.py          # Export to JSON schema and publication-ready Markdown
│   ├── ui/                       # Streamlit UI visual components
│   │   └── components.py         # Status badges, probability bar charts, citation cards, statevector tables
│   └── evaluation/               # Scientific evaluation & benchmarking
│       ├── benchmark.py          # Labelled benchmark test cases loader
│       └── evaluator.py          # Multi-mode precision, recall, FPR, FAR, and simulation agreement evaluator
├── data/
│   ├── benchmark_cases/          # Labelled benchmark test cases (JSON)
│   └── sample_proposals/         # Sample proposals (Bell State PDF, Bell State DOCX, Incomplete DOCX)
├── tests/                        # Automated Pytest test suite (35 unit & integration tests)
├── app.py                        # Streamlit interactive research dashboard
├── evaluate.py                   # Standalone evaluation & benchmark CLI runner
├── requirements.txt              # Production dependency specifications
├── .env.example                  # Environment credentials template
└── .gitignore                    # Git exclusion rules
```

---

## 2. Mathematical & Quantum Conventions

### Qubit Ordering & Bitstring Representation
- **Register Indexing**: Qubits are zero-indexed (`q_0, q_1, ..., q_{n-1}`).
- **Basis Bitstrings**: Standard computational basis little-endian convention matching Qiskit: $|q_{n-1} q_{n-2} \dots q_1 q_0\rangle$.
  - For 2 qubits, state $|01\rangle$ indicates qubit $0$ is in state $|1\rangle$ and qubit $1$ is in state $|0\rangle$.
  - In symmetric Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, both $|00\rangle$ and $|11\rangle$ have probability $0.50$ regardless of endianness.

### Supported Gate Vocabulary
- **Single-Qubit Gates**: `H`, `X`, `Y`, `Z`, `S`, `T`, `SDG`, `TDG`, `I`
- **Continuous Rotation Gates**: `RX(theta)`, `RY(theta)`, `RZ(theta)` (angles in radians)
- **Two-Qubit Entangling Gates**: `CX` (CNOT), `CZ`, `SWAP`

### Verification Metrics
- **Statevector Normalization**: $\sum_{i=0}^{2^n-1} |c_i|^2 = 1.0$ (numerical tolerance $\epsilon \le 10^{-6}$).
- **Total Variation Distance (TVD)**:
  $$\text{TVD}(P_{\text{exp}}, P_{\text{comp}}) = \frac{1}{2} \sum_{x} |P_{\text{exp}}(x) - P_{\text{comp}}(x)| \in [0.0, 1.0]$$
- **Phase-Invariant State Fidelity**:
  $$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2 \in [0.0, 1.0]$$

---

## 3. Mandatory Demonstration Cases

The prototype includes built-in reproducible demonstration cases accessible directly from the Streamlit UI dropdown and benchmark suite:

| Case ID | Proposal Description | Circuit / Gates | Expected Behavior | Verification Status |
|---|---|---|---|---|
| **Case 1** | Two-qubit Bell State Preparation ($|\Phi^+\rangle$) | `H 0`<br>`CX 0 1` | Ideal state $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$, $P(00)=0.5, P(11)=0.5$ | **PASS** |
| **Case 2** | Bell State with Deliberately Incorrect Probabilities | `H 0`<br>`CX 0 1` | Hypothesizes $P(00)=0.8, P(11)=0.2$. Discrepancy detected ($0.30 > 0.05$). | **FAIL** |
| **Case 3** | Invalid Qubit Index (Out-of-Bounds Operation) | `H 0`<br>`CX 0 2` (on 2 qubits) | Qubit index 2 exceeds register size (0..1). Rejected with structural error. | **FAIL** |
| **Case 4** | Incomplete Proposal (Missing Parameters) | None | Missing qubit count, circuit gates, and expected distributions. | **INCONCLUSIVE** |
| **Case 5** | Three-Qubit GHZ State Preparation | `H 0`<br>`CX 0 1`<br>`CX 1 2` | Tripartite entanglement $\frac{1}{\sqrt{2}}(\|000\rangle + \|111\rangle)$, $P(000)=0.5, P(111)=0.5$ | **PASS** |
| **Case 6** | Unavailable LLM API Key | Any | Safe fallback to explicitly labelled **DEMO MODE** using fixed benchmarks. | **PASS (Demo Mode)** |

---

## 4. Setup and Installation

### Prerequisites
- Python 3.11 or Python 3.12 (64-bit)
- macOS (Terminal / zsh), Linux (bash), or Windows (PowerShell)

### Step 1: Create Virtual Environment
- **macOS / Linux**:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  python -m venv --system-site-packages .venv
  ```

### Step 2: Install Dependencies
- **macOS / Linux**:
  ```bash
  pip install -r requirements.txt
  ```
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\pip.exe install -r requirements.txt
  ```

### Step 3: Configure Environment Variables
Configure your credentials in the local `.env` file (excluded from git):
```env
# Google Cloud Vertex AI Configuration (Primary Provider via Service Account ADC)
LLM_PROVIDER=vertex
GOOGLE_APPLICATION_CREDENTIALS=/path/to/vertex-service-account.json
VERTEX_PROJECT_ID=acquired-winter-511114-c2
VERTEX_LOCATION=global
VERTEX_MODEL=gemini-3.8-flash

# Google Gemini API Fallback (Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

Supported LLM providers (Google Cloud Vertex AI configured as primary provider):
- **Google Cloud Vertex AI** (Primary): Official `google-genai` SDK using `gemini-3.8-flash` via Service Account Application Default Credentials (ADC) JSON.
- **Google Gemini API**: Official `google-genai` SDK using API key from Google AI Studio.
- `OPENAI_API_KEY` (e.g. `gpt-4o`)
- `ANTHROPIC_API_KEY` (e.g. `claude-3-5-sonnet-20241022`)
- `LLM_API_KEY` + `LLM_BASE_URL` (local or OpenAI-compatible server)

*Note: If credentials are not present or external connectivity is unavailable, the system automatically falls back to deterministic extraction and local Qiskit statevector simulation without crashing.*

---

## 5. Running the Application

### Launch Streamlit Research Dashboard
- **macOS / Linux**:
  ```bash
  ./.venv/bin/streamlit run app.py
  ```
- **Windows (PowerShell)**:
  ```powershell
  $env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"; $env:USERPROFILE="$PWD"; .\.venv\Scripts\python.exe -m streamlit run app.py
  ```
Open your browser at: `http://localhost:8501`

### Dual Proposal Input Modes & Confirmation Workflow:
1. **📝 Proposal Input Modes**:
   - **Mode 1: Upload Existing Research Proposal**: Upload academic proposals in **PDF** or Word (**DOCX**) format. Extracts sections, methodology, hypothesis, quantum system, initial state, gate sequences, measurement basis, expected outcomes, and assumptions. Uses the configured LLM (or robust rule-based parser) to populate a Pydantic `ProposalSpecification` with source page references, without inventing missing parameters.
   - **Mode 2: Manual Structured Input**: Enter title, objective, hypothesis, methodology, quantum system, qubit count, initial state, circuit DSL, measurement basis, expected outcomes, and assumptions directly. Includes preloaded 1-click demonstration presets (Bell State, Biased Fail, Out-of-Bounds Fail, GHZ State).
2. **🛡️ Editable Researcher Confirmation Screen**:
   - Both input modes normalize into the identical structured Pydantic schema.
   - Displays source document metadata, full original text in an expander, and any missing information / ambiguity warnings.
   - Allows researchers to inspect and edit every extracted field before computational execution.
   - **Mandatory Confirmation**: Explicit researcher confirmation (`Confirm & Verify Proposal`) is required before sending the specification to Qiskit simulation and the verification engine.
3. **⚛️ Quantum Simulation**: Inspect Qiskit circuit diagrams, statevector components, and measurement probabilities.
4. **📊 Verification Engine**: View audited verification checks with exact PASS, FAIL, or INCONCLUSIVE statuses.
5. **📚 Evidence & Citations**: Inspect retrieved literature passages with provenance (source file, page, chunk ID).
6. **📑 Explainable Report**: Download verification reports as JSON or formatted Markdown.
7. **🗂️ Knowledge Base**: Search literature passages or upload custom research papers.
8. **🧪 Benchmark Evaluation**: Run the automated evaluation suite directly from the UI.

---

## 5. Dual Workflow: Research Paper Analysis vs Proposal Verification

The application distinguishes between two document categories:

### A. Academic Research Paper Analysis (`ResearchPaperAnalysis`)
- **Purpose**: Ingests broad academic literature (e.g. Nature, arXiv, PRX publications).
- **Extracted Fields**:
  - Publication metadata: Title, DOI, Authors, Publication/Journal details.
  - Core Academic Breakdown: Abstract, Research Problem, Research Objective, Methodology.
  - Architecture: Algorithms and Frameworks (Qiskit, PennyLane, VQE, QAOA), Experimental Platform/Setup, Quantum Systems Discussed.
  - Multi-Experiment Support: Preserves discrete `QuantumExperimentRecord` items when papers evaluate multiple benchmarks or circuit configurations.
  - Findings & Limitations: Key scientific results and stated hardware constraints (e.g. $T_2$ decoherence, two-qubit gate error limits).
- **Interactive Verification**: Researchers can inspect the full breakdown in Tab 2 and click `"Load [EXP-ID] for Verification"` to send any individual experiment directly to the Qiskit simulation pipeline.

### B. Quantum Experiment Proposal Verification (`ProposalSpecification`)
- **Purpose**: Target an individual executable quantum experiment.
- **Extracted Parameters**: Initial state, allocated qubit register ($1..10$), gate sequence DSL, measurement basis, expected computational basis probability distribution, and physical assumptions.
- **Verification Workflow**:
  - **Editable Confirmation Screen**: Researchers review and adjust parameters.
  - **Qiskit Statevector Simulation**: Computes exact quantum amplitudes and probability distributions.
  - **Independent Verification Engine**: Evaluates structural bounds, state normalization unity ($\sum |c_i|^2 = 1.0$), probability matching (TVD & fidelity), and literature grounding.
  - **INCONCLUSIVE Status Handling**: If the qubit count or gate sequence is omitted, simulation is safely bypassed without crashing, and the report is transparently marked `INCONCLUSIVE`.

---

## 6. Investigation & Fix: Qubit Count Extraction Bug

### Problem Statement
When uploading academic paper `s42256-025-01153-0.pdf`, the system failed with:
`1 validation error for CircuitSpec: num_qubits Input should be less than or equal to 10 [input_value=42257]`

### Root Cause Analysis
In the heuristic regex fallback in `src/llm/extractor.py`, the gate pattern was defined as:
```python
gate_pattern = re.compile(
    r"\b(H|X|Y|Z|S|T|RX|RY|RZ|CX|CNOT|CZ|SWAP)\s*(\([^\)]+\))?\s*(?:on\s*)?(?:qubit[s]?\s*)?([q\d\s,\[\]]+)",
    re.IGNORECASE
)
```
When encountering the Nature article identifier `s42256-025-01153-0`:
1. `\b(S)` matched the leading letter `s` because `\b` preceded the group but was missing at the end of single-letter gate names.
2. `([q\d\s,\[\]]+)` matched the trailing numerical identifier `42256`.
3. The extractor erroneously generated an `S` gate on target qubit `42256`.
4. Downstream logic derived `num_qubits = max_q + 1 = 42257`, triggering Pydantic validation failure (`num_qubits <= 10`).

### Resolution & Corrective Architecture
1. **Strict Semantic Extraction (`extract_strict_num_qubits`)**:
   - Strips DOIs (`10.xxxx/...`, `s42256-...`), 4-digit years (`2024, 2025`), citations (`[1], [42]`), page numbers, and equation numbers before qubit extraction.
   - Requires explicit contextual evidence (e.g. "two-qubit circuit", "3 qubits", "QuantumRegister(2)").
   - Maps written numerals ("two" $\to 2$, "three" $\to 3$).
   - Rejects registers $> 10$ qubits (respecting simulation prototype bounds).
   - Returns `num_qubits = null` when ambiguous or unspecified.
2. **Strict Gate Matching (`extract_strict_gates`)**:
   - Enforces word boundaries on BOTH sides of gate names (`\b([HXYZST])\b`).
   - Requires explicit gate context (e.g. `H on qubit 0`, `CX on qubits 0, 1`).
   - Enforces target qubit bounds strictly to single digits $0..9$.
3. **Pydantic Bounds & Null Safety**:
   - `CircuitSpec` requires `num_qubits <= 10` and all target qubits $< \text{num\_qubits}$.
   - Downstream simulation engine checks for null `num_qubits` and bypasses execution gracefully, setting status to `INCONCLUSIVE`.

---

## 7. Running Automated Tests

Run the complete Pytest suite (42 unit, regression, and integration tests):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Verifying across the pipeline:
- **Document Classification & Qubit Extraction (Regression Suite)**:
  - Paper with DOI `10.1038/s42256-025-01153-0` does NOT extract 42257
  - Explicit two-qubit document extracts 2
  - Missing qubit information safely returns `num_qubits = null`
  - Multi-experiment papers preserve discrete `QuantumExperimentRecord` items
  - Publication years (2024, 2025) and citations are never confused with qubits
  - Incomplete proposals do not crash and mark status as `INCONCLUSIVE`
  - Valid Bell-state proposal cleanly passes full simulation and verification
- **Google Gemini 3.8 Flash Integration**:
  - Live Gemini model catalog availability (`models/gemini-3.8-flash`)
  - Official `google-genai` SDK connectivity probe and latency diagnostics
  - Missing API key detection without hallucinated AI responses
  - Invalid API key security and error isolation (never exposing credentials)
  - Structured quantum proposal extraction into validated Pydantic models
  - Graceful fallback for malformed model responses
  - RAG context integration and citation provenance preservation
  - Inviolable computational precedence over LLM claims
- **Dual Input Modes & Confirmation**:
  - PDF proposal parsing with section & page-level metadata
  - Pure-Python DOCX parsing with explicit and rendered page-break extraction
  - Manual structured input mode with all experimental parameters
  - Schema compatibility across both input modes
  - Non-hallucination of missing operations/probabilities on incomplete proposals (yielding `INCONCLUSIVE`)
  - Editable researcher review/confirmation flow updating downstream simulation and verification
- **Quantum Circuit Simulation**:
  - Ideal Bell state simulation ($|\Phi^+\rangle$) with $P(00)=0.5, P(11)=0.5$
  - Deliberately incorrect probability rejection
  - Out-of-bounds qubit index handling
  - Malformed gate syntax and angle parameter validation
  - Three-qubit GHZ state tripartite entanglement
  - Normalization unity ($\sum |c_i|^2 = 1.0$)
  - Phase-invariant fidelity & TVD metrics
- **Document Processing & Knowledge Base**:
  - PDF section segmentation and Dirac bra-ket math extraction
  - Document chunking with page-level provenance
  - Anti-hallucination warning for scanned documents
- **Verification Engine Rules**:
  - Verification engine rule precedence (computational failure cannot be overridden by LLM)
  - Physical NISQ hardware feasibility disclaimers
  - Missing parameters producing `INCONCLUSIVE` instead of invented details
- **Reporting & Evaluation**:
  - JSON and Markdown report export completeness
  - Benchmark evaluation harness and CSV export

---

## 8. Running the Scientific Benchmark Evaluation

Run the standalone benchmark evaluator:
```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This runs all 6 labelled benchmark proposals across three verification modalities:
1. **Rule & Simulation Only**: Computational checks without RAG literature grounding.
2. **LLM Analysis (No RAG)**: Generative extraction without retrieved literature.
3. **Hybrid Verification (Rules + Simulation + RAG)**: Full multi-tier verification engine.

### Benchmark Evaluation Results:
```
==================================================================================================
Verification Mode              | Prec.   | Recall  | FPR     | FAR     | Sim Agree  | Latency (ms)
--------------------------------------------------------------------------------------------------
Hybrid (Rules + Sim + RAG)     | 1.0000  | 1.0000  | 0.0000  | 0.0000  | 1.0000     | 3.04 ms
LLM Analysis (No RAG)          | 1.0000  | 1.0000  | 0.0000  | 0.0000  | 0.6667     | 1.69 ms
Rule & Simulation Only         | 1.0000  | 1.0000  | 0.0000  | 0.0000  | 0.6667     | 60.08 ms
==================================================================================================
```
- **Error Detection Precision**: `100.0%`
- **Error Detection Recall**: `100.0%`
- **False-Pass Rate (FPR)**: `0.0%` (Zero invalid proposals falsely passed)
- **False-Alarm Rate (FAR)**: `0.0%`
- **Simulation Agreement**: `100.0%` in Hybrid mode
- Results exported to: `evaluation_results.csv`

---

## 8. Security & Scientific Integrity Principles

1. **Inviolable Computational Precedence**:
   Computational simulation failure (e.g., probability mismatch or out-of-bounds gate) **CANNOT** be overridden by an LLM or literature argument. If simulation fails, the check status is strictly `FAIL`.
2. **Prompt Injection Defense**:
   Uploaded proposal text is strictly isolated in `<UNTRUSTED_PROPOSAL_TEXT>` tags and processed as passive data. System prompts instruct the model never to execute embedded directives.
3. **Traceable Citations (Zero Hallucination)**:
   Every citation contains the exact source filename, page number, chunk identifier, and similarity score. Passages are retrieved exclusively from indexed documents; citations are never fabricated.
4. **Physical NISQ Feasibility Disclaimer**:
   Passing an ideal statevector simulation does **not** certify physical experiment execution. All reports include mandatory disclaimers regarding $T_1/T_2^*$ decoherence, two-qubit gate fidelities, crosstalk, and pulse calibration overhead on real hardware.
