"""
DOCX Processing and Document Extraction Module.
Extracts text, paragraphs, tables, and page-level metadata from Microsoft Word (.docx) files
using pure-Python OpenXML parsing (zipfile + xml.etree.ElementTree).
Preserves page provenance, detects quantum math/notation, identifies sections,
and enforces non-hallucinatory extraction.
"""

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple

from .pdf_extractor import ExtractedDocument, DocumentExtractionError

# OpenXML Namespaces
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NAMESPACES = {"w": W_NS}


class DOCXExtractor:
    """
    Extracts text, sections, and page references from .docx proposal files.
    """

    def __init__(self, chars_per_page_fallback: int = 2500):
        self.chars_per_page_fallback = chars_per_page_fallback

    def extract_from_bytes(self, docx_bytes: bytes, filename: str = "uploaded_proposal.docx") -> ExtractedDocument:
        """Extract text and metadata from raw DOCX bytes."""
        if not docx_bytes:
            raise DocumentExtractionError("Uploaded DOCX file is empty (0 bytes).")

        warnings: List[str] = []

        try:
            with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
                if "word/document.xml" not in zf.namelist():
                    raise DocumentExtractionError("Invalid .docx file: missing 'word/document.xml'.")

                doc_xml_bytes = zf.read("word/document.xml")
                root = ET.fromstring(doc_xml_bytes)

        except zipfile.BadZipFile as e:
            raise DocumentExtractionError(f"Corrupted or invalid DOCX archive: {e}")
        except Exception as e:
            raise DocumentExtractionError(f"Failed to parse DOCX document XML: {e}")

        # Traverse document body elements (paragraphs and tables)
        pages_dict: Dict[int, List[str]] = {1: []}
        current_page = 1
        has_explicit_breaks = False

        body = root.find(f"{{{W_NS}}}body")
        if body is None:
            warnings.append("Document body element (<w:body>) is missing.")
            elements = []
        else:
            elements = list(body)

        for elem in elements:
            tag = elem.tag

            # Check for paragraph
            if tag == f"{{{W_NS}}}p":
                para_text, had_break = self._parse_paragraph(elem)
                if had_break:
                    current_page += 1
                    has_explicit_breaks = True
                    if current_page not in pages_dict:
                        pages_dict[current_page] = []
                if para_text.strip():
                    pages_dict[current_page].append(para_text.strip())

            # Check for table
            elif tag == f"{{{W_NS}}}tbl":
                table_lines = self._parse_table(elem)
                if table_lines:
                    pages_dict[current_page].extend(table_lines)

        # Assemble pages data
        pages_data: List[Dict[str, Any]] = []

        if has_explicit_breaks:
            for p_num in sorted(pages_dict.keys()):
                p_text = "\n\n".join(pages_dict[p_num]).strip()
                if p_text:
                    pages_data.append({
                        "page_number": p_num,
                        "text": p_text,
                        "char_count": len(p_text)
                    })
        else:
            # Fallback logical page segmentation if document has no hard page breaks
            all_text_combined = "\n\n".join(
                line for p_num in sorted(pages_dict.keys()) for line in pages_dict[p_num]
            ).strip()

            if len(all_text_combined) <= self.chars_per_page_fallback:
                pages_data.append({
                    "page_number": 1,
                    "text": all_text_combined,
                    "char_count": len(all_text_combined)
                })
            else:
                # Segment by paragraph into ~2500 char pages
                p_num = 1
                curr_chunk = []
                curr_len = 0
                all_paras = [line for p_num in sorted(pages_dict.keys()) for line in pages_dict[p_num]]

                for para in all_paras:
                    curr_chunk.append(para)
                    curr_len += len(para) + 2
                    if curr_len >= self.chars_per_page_fallback:
                        p_content = "\n\n".join(curr_chunk).strip()
                        pages_data.append({
                            "page_number": p_num,
                            "text": p_content,
                            "char_count": len(p_content)
                        })
                        p_num += 1
                        curr_chunk = []
                        curr_len = 0

                if curr_chunk:
                    p_content = "\n\n".join(curr_chunk).strip()
                    pages_data.append({
                        "page_number": p_num,
                        "text": p_content,
                        "char_count": len(p_content)
                    })

        # Check for empty document
        full_text = "\n\n".join([f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages_data])
        raw_combined = "\n\n".join([p["text"] for p in pages_data]).strip()

        total_chars = len(raw_combined)
        is_scanned = False
        if total_chars < 50:
            is_scanned = True
            warnings.append(
                "CRITICAL WARNING: The document contains fewer than 50 text characters. "
                "The file appears blank or contains only images. No information will be invented."
            )

        # Detect math expressions
        math_expressions = self._extract_math_expressions(raw_combined)

        # Identify sections
        sections = self._identify_sections(raw_combined)

        expected_sections = ["abstract", "objective", "methodology", "expected_results"]
        for sec in expected_sections:
            if sec not in sections or not sections[sec].strip():
                warnings.append(f"Notice: Section '{sec}' could not be unambiguously identified in DOCX structure.")

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

    def _parse_paragraph(self, p_elem: ET.Element) -> Tuple[str, bool]:
        """Extract text from paragraph runs and detect page breaks."""
        parts = []
        had_break = False

        for elem in p_elem.iter():
            tag = elem.tag

            # Page break tags: <w:br w:type="page"/> or <w:lastRenderedPageBreak/>
            if tag == f"{{{W_NS}}}br" and elem.attrib.get(f"{{{W_NS}}}type") == "page":
                had_break = True
            elif tag == f"{{{W_NS}}}lastRenderedPageBreak":
                had_break = True

            # Text tag
            elif tag == f"{{{W_NS}}}t" and elem.text:
                parts.append(elem.text)
            # Tab tag
            elif tag == f"{{{W_NS}}}tab":
                parts.append("\t")

        return "".join(parts), had_break

    def _parse_table(self, tbl_elem: ET.Element) -> List[str]:
        """Extract table rows as formatted lines."""
        lines = []
        for tr in tbl_elem.iter(f"{{{W_NS}}}tr"):
            row_cells = []
            for tc in tr.iter(f"{{{W_NS}}}tc"):
                cell_text = "".join(t.text for t in tc.iter(f"{{{W_NS}}}t") if t.text)
                row_cells.append(cell_text.strip())
            if any(row_cells):
                lines.append(" | ".join(row_cells))
        return lines

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
        return found[:50]

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
                    sections[sec_name] = content[:3000]

        if "title" not in sections:
            first_lines = [l.strip() for l in text.splitlines() if l.strip()]
            if first_lines:
                sections["title"] = first_lines[0]

        return sections
