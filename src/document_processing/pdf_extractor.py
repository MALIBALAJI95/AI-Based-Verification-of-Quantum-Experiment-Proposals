"""
PDF Processing and Document Extraction Module.
Extracts text from PDF documents using PyMuPDF (fitz) or fallback parsers.
Identifies key scientific sections and flags missing or scanned content.
NEVER hallucinates or silently invents missing text.
"""

import io
import re
from typing import Dict, List, Optional, Any, Tuple

# Try PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Try PyPDF2 as secondary fallback
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class DocumentExtractionError(Exception):
    """Raised when document parsing fails critically."""
    pass


class ExtractedDocument:
    """Container for extracted document pages, text, and structural metadata."""
    def __init__(
        self,
        filename: str,
        full_text: str,
        pages: List[Dict[str, Any]],
        sections: Dict[str, str],
        math_expressions: List[str],
        warnings: List[str],
        is_scanned: bool = False
    ):
        self.filename = filename
        self.full_text = full_text
        self.pages = pages  # list of {"page_number": int, "text": str, "char_count": int}
        self.sections = sections
        self.math_expressions = math_expressions
        self.warnings = warnings
        self.is_scanned = is_scanned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "total_pages": len(self.pages),
            "char_count": len(self.full_text),
            "sections_identified": list(self.sections.keys()),
            "math_expression_count": len(self.math_expressions),
            "warnings": self.warnings,
            "is_scanned": self.is_scanned,
        }


class PDFExtractor:
    """
    Extracts text and structured sections from research proposals in PDF format.
    Ensures non-hallucinatory extraction with explicit warnings for missing data.
    """

    def __init__(self):
        pass

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded_proposal.pdf") -> ExtractedDocument:
        """Extract text and metadata from PDF bytes (e.g. from Streamlit file uploader)."""
        if not pdf_bytes:
            raise DocumentExtractionError("Uploaded file is empty (0 bytes).")

        pages_data: List[Dict[str, Any]] = []
        warnings: List[str] = []

        if HAS_PYMUPDF:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text") or ""
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": page_text,
                        "char_count": len(page_text.strip())
                    })
                doc.close()
            except Exception as e:
                warnings.append(f"PyMuPDF error: {e}. Trying fallback extractor.")

        if not pages_data and HAS_PYPDF2:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": page_text,
                        "char_count": len(page_text.strip())
                    })
            except Exception as e:
                warnings.append(f"PyPDF2 error: {e}")

        if not pages_data:
            # If neither library succeeded or neither is available
            raise DocumentExtractionError(
                "Unable to extract text from PDF: neither PyMuPDF nor PyPDF2 succeeded. "
                "Ensure PyMuPDF is installed or provide the proposal via direct text input."
            )

        # Assemble full text
        full_text = "\n\n".join([f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages_data])
        raw_combined = "\n\n".join([p["text"] for p in pages_data]).strip()

        # Check for scanned or empty pages
        total_chars = sum(p["char_count"] for p in pages_data)
        is_scanned = False
        if total_chars < 50:
            is_scanned = True
            warnings.append(
                "CRITICAL WARNING: The document contains fewer than 50 text characters. "
                "This document appears to be scanned or contains image-only pages. "
                "No information will be invented; please provide digital text or transcribe manually."
            )

        # Extract mathematical expressions (LaTeX, braket notation, Pauli gates)
        math_expressions = self._extract_math_expressions(raw_combined)

        # Identify standard sections
        sections = self._identify_sections(raw_combined)

        # Check missing sections and record warnings
        expected_sections = ["abstract", "objective", "methodology", "expected_results"]
        for sec in expected_sections:
            if sec not in sections or not sections[sec].strip():
                warnings.append(f"Notice: Section '{sec}' could not be unambiguously identified in the document structure.")

        return ExtractedDocument(
            filename=filename,
            full_text=full_text,
            pages=pages_data,
            sections=sections,
            math_expressions=math_expressions,
            warnings=warnings,
            is_scanned=is_scanned
        )

    def extract_from_file(self, file_path: str) -> ExtractedDocument:
        """Extract from a local file path."""
        with open(file_path, "rb") as f:
            content = f.read()
        return self.extract_from_bytes(content, filename=file_path.split("/")[-1].split("\\")[-1])

    def _extract_math_expressions(self, text: str) -> List[str]:
        """Detect Dirac notation, equations, and quantum operator expressions."""
        patterns = [
            r"\|[a-zA-Z0-9\+\-\/\s\?]+(?:\>|\\rangle)",  # Dirac ket |0>, |1>, |00>, |+>, |Phi+>, |psi>
            r"\$[^\$]+\$",                             # LaTeX inline math $...$
            r"\\[a-zA-Z]+\{[^}]*\}",                   # LaTeX commands \frac{}, \ket{}
            r"\b[XYZHST]\b\s*(\([^\)]+\))?\s*\b(gate|operation)\b", # Gate references
            r"\b\d+\s*qubits?\b",                      # System sizes
            r"\b\d+\.\d+\s*(\%|probability)\b"         # Probabilities
        ]
        found = []
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                item = m[0] if isinstance(m, tuple) else m
                if item and item not in found:
                    found.append(item)
        return found[:50]  # Cap list to top 50 expressions

    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Heuristic section segmenter looking for common scientific paper headers."""
        headers = {
            "title": r"^(.*?)(?=\n\s*(abstract|introduction|objective))",
            "abstract": r"(?is)\babstract\b[:\s]*(.*?)(?=\n\s*(introduction|objective|methodology|hypothesis|1\.|\n\n\n))",
            "objective": r"(?is)\b(objective|goal|purpose)\b[:\s]*(.*?)(?=\n\s*(hypothesis|methodology|methods|quantum system|circuit|experimental setup|2\.|\n\n\n))",
            "hypothesis": r"(?is)\b(hypothesis|theoretical basis|theoretical foundation)\b[:\s]*(.*?)(?=\n\s*(methodology|methods|quantum system|circuit|experimental setup|expected results|3\.|\n\n\n))",
            "quantum_system": r"(?is)\b(quantum system|target architecture|hardware platform|quantum processor|physical system|qubit implementation)\b[:\s]*(.*?)(?=\n\s*(methodology|methods|circuit|initial state|expected results|\n\n\n))",
            "methodology": r"(?is)\b(methodology|methods|quantum circuit|gate sequence|experimental setup|procedure)\b[:\s]*(.*?)(?=\n\s*(initial state|measurement|expected results|outcomes|results|discussion|4\.|\n\n\n))",
            "initial_state": r"(?is)\b(initial state|state preparation|input state|register initialization)\b[:\s]*(.*?)(?=\n\s*(gates|operations|circuit|measurement|expected results|\n\n\n))",
            "measurement_basis": r"(?is)\b(measurement basis|observables|readout basis|measurement)\b[:\s]*(.*?)(?=\n\s*(expected results|outcomes|results|assumptions|\n\n\n))",
            "expected_results": r"(?is)\b(expected results|expected outcomes|measurement outcomes|outcomes|results)\b[:\s]*(.*?)(?=\n\s*(assumptions|limitations|discussion|conclusion|references|5\.|\n\n\n))",
            "assumptions": r"(?is)\b(assumptions|constraints|hardware constraints)\b[:\s]*(.*?)(?=\n\s*(limitations|conclusion|references|\n\n\n))",
            "limitations": r"(?is)\b(limitations|threats to validity)\b[:\s]*(.*?)(?=\n\s*(conclusion|references|\n\n\n))",
            "references": r"(?is)\b(references|bibliography)\b[:\s]*(.*)$",
        }

        sections: Dict[str, str] = {}
        for sec_name, pattern in headers.items():
            match = re.search(pattern, text)
            if match:
                content = match.group(1).strip() if match.groups() else match.group(0).strip()
                if content:
                    sections[sec_name] = content[:3000]  # Store reasonable section preview

        # If title wasn't found by lookahead, take the first non-empty line
        if "title" not in sections:
            first_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if first_lines:
                sections["title"] = first_lines[0]

        return sections
