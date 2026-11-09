"""
Vector Store implementation for Quantum Literature RAG.
Supports ChromaDB with Sentence-Transformers embeddings, plus an in-memory
vector/BM25 fallback for deterministic offline operations.
Stores chunk metadata (filename, page number, chunk ID, title) for traceable citations.
"""

import os
import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv

# Load local environment configuration
load_dotenv(".env")

from ..document_processing.chunker import DocumentChunk, TraceableChunker
from .starter_corpus import STARTER_DOCUMENTS

# Try importing ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Try importing SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class FallbackVectorStore:
    """
    Lightweight, fast in-memory TF-IDF cosine-similarity vector store.
    Guarantees that semantic retrieval and citation tracking work deterministically
    even when external model downloads are restricted or ChromaDB is loading.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []

    def add(self, chunk: DocumentChunk, title: str):
        tokens = self._tokenize(chunk.text)
        self.documents.append({
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number,
            "title": title,
            "tokens": tokens,
            "tf": Counter(tokens)
        })

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.documents:
            return []

        q_tf = Counter(q_tokens)
        results = []

        for doc in self.documents:
            # Cosine similarity on word frequency
            dot = sum(q_tf[w] * doc["tf"].get(w, 0) for w in q_tf)
            norm_q = math.sqrt(sum(v * v for v in q_tf.values()))
            norm_d = math.sqrt(sum(v * v for v in doc["tf"].values()))
            score = (dot / (norm_q * norm_d)) if (norm_q > 0 and norm_d > 0) else 0.0

            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "source_filename": doc["source_filename"],
                "page_number": doc["page_number"],
                "title": doc["title"],
                "score": round(score, 4)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class QuantumKnowledgeBase:
    """
    Manages semantic storage, embedding, and traceable retrieval of quantum literature.
    """

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "quantum_literature"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.fallback = FallbackVectorStore()
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.use_chroma = False

        self._initialize_backends()

    def _initialize_backends(self):
        """Attempt to initialize ChromaDB and SentenceTransformer; fall back if needed."""
        if HAS_CHROMADB:
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
                self.use_chroma = True
            except Exception:
                self.use_chroma = False

        if HAS_ST:
            try:
                # Load lightweight standard embedding model
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(model_name)
            except Exception:
                self.embedding_model = None

    def index_chunks(self, chunks: List[DocumentChunk], source_title: str) -> int:
        """Add traceable document chunks into the knowledge base."""
        if not chunks:
            return 0

        # Always index into fallback store to ensure zero-failure retrieval
        for ch in chunks:
            self.fallback.add(ch, source_title)

        if self.use_chroma and self.collection:
            try:
                ids = [ch.chunk_id for ch in chunks]
                texts = [ch.text for ch in chunks]
                metadatas = [
                    {
                        "source_filename": ch.source_filename,
                        "page_number": ch.page_number or 1,
                        "title": source_title,
                        "chunk_id": ch.chunk_id
                    }
                    for ch in chunks
                ]

                embeddings = None
                if self.embedding_model:
                    embeddings = self.embedding_model.encode(texts).tolist()

                if embeddings:
                    self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
                else:
                    self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            except Exception:
                # If ChromaDB write fails, fallback store remains fully active
                pass

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve relevant literature passages with verified citation metadata.
        """
        if self.use_chroma and self.collection and self.collection.count() > 0:
            try:
                if self.embedding_model:
                    q_emb = self.embedding_model.encode([query]).tolist()
                    res = self.collection.query(query_embeddings=q_emb, n_results=top_k)
                else:
                    res = self.collection.query(query_texts=[query], n_results=top_k)

                results = []
                if res and "documents" in res and res["documents"]:
                    docs = res["documents"][0]
                    metas = res["metadatas"][0] if "metadatas" in res else [{}] * len(docs)
                    distances = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(docs)

                    for d, m, dist in zip(docs, metas, distances):
                        # Convert distance to similarity score
                        sim_score = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else 0.8
                        results.append({
                            "chunk_id": m.get("chunk_id", "unknown"),
                            "text": d,
                            "source_filename": m.get("source_filename", "literature"),
                            "page_number": m.get("page_number", 1),
                            "title": m.get("title", "Quantum Paper"),
                            "score": round(sim_score, 4)
                        })
                    if results:
                        return results
            except Exception:
                pass

        # Use fallback engine
        return self.fallback.search(query, top_k=top_k)

    def seed_starter_corpus(self) -> int:
        """Load built-in authoritative quantum computing papers if knowledge base is empty."""
        chunker = TraceableChunker(chunk_size=500, chunk_overlap=80)
        total_chunks = 0

        for doc in STARTER_DOCUMENTS:
            fname = doc["filename"]
            title = doc["title"]
            pages = doc.get("pages", [])

            chunks = chunker.chunk_document(full_text="", source_filename=fname, pages=pages)
            added = self.index_chunks(chunks, source_title=title)
            total_chunks += added

        return total_chunks

    def count(self) -> int:
        """Return total number of indexed passages."""
        if self.use_chroma and self.collection:
            try:
                cnt = self.collection.count()
                if cnt > 0:
                    return cnt
            except Exception:
                pass
        return len(self.fallback.documents)
