"""
Unified Document Extractor for Research Proposals.
Supports both PDF and DOCX formats with unified ExtractedDocument representation,
traceable page provenance, mathematical expression detection, and section parsing.
"""

import os
from typing import Optional, Dict, Any, List

from .pdf_extractor import PDFExtractor, ExtractedDocument, DocumentExtractionError
from .docx_extractor import DOCXExtractor


class DocumentExtractor:
    """
    Unified entry point for extracting text and structured metadata from PDF and DOCX proposals.
    """

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DOCXExtractor()

    def extract_from_bytes(self, file_bytes: bytes, filename: str = "proposal.pdf") -> ExtractedDocument:
        """Extract text and metadata from document bytes, dispatching by extension or magic header."""
        if not file_bytes:
            raise DocumentExtractionError("Uploaded file is empty (0 bytes).")

        lower_name = filename.lower()

        # Check extension first
        if lower_name.endswith(".docx"):
            return self.docx_extractor.extract_from_bytes(file_bytes, filename=filename)
        elif lower_name.endswith(".pdf"):
            return self.pdf_extractor.extract_from_bytes(file_bytes, filename=filename)
        elif lower_name.endswith(".txt"):
            return self._extract_txt_bytes(file_bytes, filename=filename)

        # Sniff magic header
        if file_bytes.startswith(b"%PDF"):
            return self.pdf_extractor.extract_from_bytes(file_bytes, filename=filename)
        elif file_bytes.startswith(b"PK\x03\x04"):
            # ZIP header, likely DOCX
            try:
                return self.docx_extractor.extract_from_bytes(file_bytes, filename=filename)
            except Exception:
                pass

        raise DocumentExtractionError(
            f"Unsupported file format for '{filename}'. Only PDF (.pdf) and Word (.docx) files are supported."
        )

    def extract_from_file(self, file_path: str) -> ExtractedDocument:
        """Extract text and metadata from a local file path."""
        if not os.path.exists(file_path):
            raise DocumentExtractionError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            content = f.read()
        return self.extract_from_bytes(content, filename=os.path.basename(file_path))

    def _extract_txt_bytes(self, text_bytes: bytes, filename: str) -> ExtractedDocument:
        """Handle plain text files with simulated page chunking."""
        text = text_bytes.decode("utf-8", errors="ignore")
        # Split into ~2500 character logical pages
        page_size = 2500
        pages = []
        lines = text.splitlines(keepends=True)
        current_page_text = []
        current_len = 0
        p_num = 1

        for line in lines:
            current_page_text.append(line)
            current_len += len(line)
            if current_len >= page_size:
                p_text = "".join(current_page_text).strip()
                if p_text:
                    pages.append({
                        "page_number": p_num,
                        "text": p_text,
                        "char_count": len(p_text)
                    })
                    p_num += 1
                current_page_text = []
                current_len = 0

        if current_page_text:
            p_text = "".join(current_page_text).strip()
            if p_text:
                pages.append({
                    "page_number": p_num,
                    "text": p_text,
                    "char_count": len(p_text)
                })

        if not pages:
            pages.append({"page_number": 1, "text": text, "char_count": len(text)})

        full_text = "\n\n".join([f"--- Page {p['page_number']} ---\n{p['text']}" for p in pages])
        sections = self.docx_extractor._identify_sections(text)
        math_exprs = self.docx_extractor._extract_math_expressions(text)

        return ExtractedDocument(
            filename=filename,
            full_text=full_text,
            pages=pages,
            sections=sections,
            math_expressions=math_exprs,
            warnings=[],
            is_scanned=len(text.strip()) < 50
        )
