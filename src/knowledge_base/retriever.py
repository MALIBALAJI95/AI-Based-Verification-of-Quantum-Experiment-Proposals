"""
Traceable Quantum Literature Retriever.
Formulates targeted semantic queries across proposal hypotheses, gates, and expected distributions.
Converts raw chunks into verified LiteratureCitation models with transparent provenance.
"""

from typing import List, Dict, Any, Optional
from .vector_store import QuantumKnowledgeBase
from ..schemas.verification import LiteratureCitation
from ..schemas.proposal import ProposalSpecification


class QuantumRetriever:
    """
    Retrieves grounded evidence from the quantum computing knowledge base.
    Prevents hallucination by requiring every citation to originate from indexed text.
    """

    def __init__(self, kb: QuantumKnowledgeBase):
        self.kb = kb

    def retrieve_for_proposal(self, proposal: ProposalSpecification, top_k_per_query: int = 2) -> List[LiteratureCitation]:
        """
        Formulate targeted queries based on the proposal's title, hypothesis,
        gate sequence, and expected outcomes.
        """
        queries = []

        if proposal.hypothesis:
            queries.append(f"Hypothesis and physical principles: {proposal.hypothesis}")
        if proposal.objective:
            queries.append(f"Quantum experiment objective: {proposal.objective}")

        if proposal.circuit and proposal.circuit.gates:
            gate_names = [g.name for g in proposal.circuit.gates]
            queries.append(f"Quantum circuit gate sequence: {' '.join(gate_names)} entangling operations")

        if proposal.expected_outcomes:
            outcomes_desc = " ".join([f"|{o.basis_state}> prob {o.expected_probability}" for o in proposal.expected_outcomes])
            queries.append(f"Expected measurement outcomes: {outcomes_desc}")

        if not queries:
            if proposal.title:
                queries.append(proposal.title)
            else:
                queries.append("Quantum state preparation and measurement verification.")

        seen_chunks = set()
        citations: List[LiteratureCitation] = []
        cit_counter = 1

        for q in queries:
            results = self.kb.retrieve(q, top_k=top_k_per_query)
            for res in results:
                chunk_id = res["chunk_id"]
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)

                citation = LiteratureCitation(
                    citation_id=f"CIT-{cit_counter:02d}",
                    source_title=res.get("title", "Quantum Reference"),
                    filename=res.get("source_filename", "literature"),
                    page_number=res.get("page_number", 1),
                    chunk_id=chunk_id,
                    passage=res.get("text", "")[:400] + ("..." if len(res.get("text", "")) > 400 else ""),
                    similarity_score=res.get("score", 0.0),
                    relevance_comment=f"Retrieved for query context: '{q[:60]}...'"
                )
                citations.append(citation)
                cit_counter += 1

        return citations
