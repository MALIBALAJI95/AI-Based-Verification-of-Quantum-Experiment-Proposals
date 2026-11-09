"""
Traceable Document Chunker for RAG and Literature Knowledge Base.
Splits texts into semantic or windowed chunks preserving source filename, page number,
and deterministic chunk IDs for unalterable audit trails.
"""

import re
from typing import List, Dict, Any, Optional


class DocumentChunk:
    """Represents an atomic, traceable chunk of text in the knowledge base."""
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_filename: str,
        page_number: Optional[int] = None,
        section_name: Optional[str] = None,
        char_start: int = 0,
        char_end: int = 0
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_filename = source_filename
        self.page_number = page_number
        self.section_name = section_name
        self.char_start = char_start
        self.char_end = char_end

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "char_start": self.char_start,
            "char_end": self.char_end
        }


class TraceableChunker:
    """
    Chunks document pages into overlapping semantic windows.
    Ensures that every chunk preserves provenance metadata.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        full_text: str,
        source_filename: str,
        pages: Optional[List[Dict[str, Any]]] = None
    ) -> List[DocumentChunk]:
        """
        Split document into traceable chunks. If page data is available,
        chunks correspond to specific pages.
        """
        chunks: List[DocumentChunk] = []

        if pages:
            chunk_counter = 0
            for page in pages:
                page_num = page.get("page_number", 1)
                page_text = page.get("text", "").strip()
                if not page_text:
                    continue

                page_chunks = self._chunk_text(
                    page_text,
                    source_filename=source_filename,
                    page_number=page_num,
                    start_id=chunk_counter
                )
                chunk_counter += len(page_chunks)
                chunks.extend(page_chunks)
            return chunks

        # Fallback to full text chunking if page breakdown is absent
        return self._chunk_text(full_text, source_filename=source_filename, page_number=None, start_id=0)

    def _chunk_text(
        self,
        text: str,
        source_filename: str,
        page_number: Optional[int],
        start_id: int
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        if not text:
            return chunks

        step = max(50, self.chunk_size - self.chunk_overlap)
        idx = 0
        counter = start_id

        # Clean multiple newlines
        clean_text = re.sub(r"\n{3,}", "\n\n", text)
        text_len = len(clean_text)

        while idx < text_len:
            end_idx = min(idx + self.chunk_size, text_len)
            
            # Snap to sentence boundary if nearby
            if end_idx < text_len:
                period_pos = clean_text.rfind(". ", idx, end_idx)
                if period_pos != -1 and period_pos > (idx + self.chunk_size // 2):
                    end_idx = period_pos + 1

            chunk_content = clean_text[idx:end_idx].strip()
            if chunk_content:
                chunk_id = f"{source_filename}_p{page_number or 1}_c{counter:03d}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_content,
                        source_filename=source_filename,
                        page_number=page_number,
                        char_start=idx,
                        char_end=end_idx
                    )
                )
                counter += 1

            if end_idx >= text_len:
                break
            idx += step

        return chunks
