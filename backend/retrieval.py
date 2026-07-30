"""Multi-Stage / Multi-Chunk RAG Retrieval Pipeline module."""

import logging
from typing import Any

from backend.db import VectorStore
from backend.embeddings import LocalEmbedder

logger = logging.getLogger(__name__)


class MultiStageRetriever:
    """Multi-stage retrieval pipeline combining multi-query expansion, RRF, and neighbor window expansion."""

    def __init__(self, vector_store: VectorStore, embedder: LocalEmbedder) -> None:
        """Initialize retriever instance.

        Args:
            vector_store: VectorStore instance.
            embedder: LocalEmbedder instance.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def generate_query_variations(
        self, user_query: str, history: list[dict[str, str]] | None = None
    ) -> list[str]:
        """Generate search query variations for multi-query retrieval using conversation context.

        Args:
            user_query: Original user question.
            history: Optional list of prior conversation history turns.

        Returns:
            list[str]: List of unique search queries.
        """
        queries = [user_query.strip()]
        cleaned = user_query.strip()

        # Only construct conversationally expanded query if user query is a short follow-up question
        is_short_followup = (
            len(cleaned.split()) <= 4
            or any(w in cleaned.lower() for w in ["it", "this", "that", "he", "she", "they", "them", "which", "more", "details"])
        )

        if history and is_short_followup:
            recent_user_turns = [
                turn.get("content", "").strip()
                for turn in history
                if turn.get("role") == "user" and turn.get("content")
            ][-1:]

            if recent_user_turns:
                contextual_prompt = f"{' '.join(recent_user_turns)} {cleaned}".strip()
                if contextual_prompt != cleaned:
                    queries.append(contextual_prompt)

        return list(dict.fromkeys(queries))


    def reciprocal_rank_fusion(
        self, query_results: list[list[dict[str, Any]]], rrf_k: int = 60, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Combine multiple candidate chunk rank lists using Reciprocal Rank Fusion (RRF).

        Args:
            query_results: List of chunk candidate lists from different search queries.
            rrf_k: RRF constant.
            top_n: Number of top fused chunks to return.

        Returns:
            list[dict[str, Any]]: Deduplicated and RRF re-ranked chunk objects.
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for chunks in query_results:
            for rank, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                doc_id = meta.get("doc_id", meta.get("filename", "doc"))
                chunk_idx = meta.get("chunk_index", 0)
                chunk_key = f"{doc_id}-chunk-{chunk_idx}"

                if chunk_key not in chunk_map:
                    chunk_map[chunk_key] = chunk
                    scores[chunk_key] = 0.0

                scores[chunk_key] += 1.0 / (rrf_k + rank)

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_n]
        return [chunk_map[k] for k in sorted_keys]

    def expand_neighbor_context(
        self, ranked_chunks: list[dict[str, Any]], window_size: int = 1
    ) -> list[dict[str, Any]]:
        """Expand retrieved chunks with neighboring preceding and succeeding chunk context.

        Args:
            ranked_chunks: Top RRF-ranked chunk objects.
            window_size: Number of adjacent chunks to fetch (default: 1).

        Returns:
            list[dict[str, Any]]: Expanded contiguous chunk blocks.
        """
        if not ranked_chunks:
            return []

        expanded_results: list[dict[str, Any]] = []

        for chunk in ranked_chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("doc_id", meta.get("filename", ""))
            chunk_idx = meta.get("chunk_index", None)
            total_chunks = meta.get("total_chunks", 1)

            if not doc_id or chunk_idx is None:
                expanded_results.append(chunk)
                continue

            target_indices = [
                idx
                for idx in range(max(0, chunk_idx - window_size), min(total_chunks, chunk_idx + window_size + 1))
            ]

            neighbors = self.vector_store.get_chunks_by_indices(doc_id=doc_id, chunk_indices=target_indices)

            if neighbors:
                neighbors.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
                merged_text = "\n\n".join([c.get("text", "") for c in neighbors if c.get("text")])
                if merged_text:
                    expanded_chunk = dict(chunk)
                    expanded_chunk["match_text"] = chunk.get("text", "")
                    expanded_chunk["text"] = merged_text
                    expanded_results.append(expanded_chunk)
                else:
                    expanded_results.append(chunk)
            else:
                expanded_results.append(chunk)


        return expanded_results

    def retrieve(
        self, user_query: str, history: list[dict[str, str]] | None = None, top_k: int = 4
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Execute complete multi-stage retrieval pipeline and return results with reasoning steps.

        Args:
            user_query: User chat question.
            history: Optional list of prior conversation history turns.
            top_k: Number of top candidate results.

        Returns:
            tuple[list[dict[str, Any]], list[str]]: Final expanded chunk context list and reasoning trace steps.
        """
        reasoning_steps: list[str] = []

        if self.vector_store.collection.count() == 0:
            reasoning_steps.append("ChromaDB knowledge base is empty — 0 document chunks found")
            return [], reasoning_steps

        # Stage 1: Multi-query expansion
        query_variations = self.generate_query_variations(user_query=user_query, history=history)
        reasoning_steps.append(f"Generated {len(query_variations)} query variations for broad topic coverage")

        # Stage 2: Multi-vector search
        all_candidate_sets: list[list[dict[str, Any]]] = []
        for q in query_variations:
            candidates = self.vector_store.query_similar(query=q, embedder=self.embedder, top_k=top_k)
            if candidates:
                all_candidate_sets.append(candidates)

        if not all_candidate_sets:
            reasoning_steps.append("Vector search completed — 0 relevant document chunks passed similarity threshold (cosine distance <= 0.70)")
            return [], reasoning_steps


        # Stage 3: Reciprocal Rank Fusion
        fused_chunks = self.reciprocal_rank_fusion(all_candidate_sets, top_n=top_k)
        reasoning_steps.append(f"Applied Reciprocal Rank Fusion (RRF) across candidate sets")

        # Stage 4: Neighbor window context expansion
        final_chunks = self.expand_neighbor_context(fused_chunks, window_size=1)
        reasoning_steps.append(f"Stitched neighboring chunk windows into {len(final_chunks)} contiguous context blocks")

        return final_chunks, reasoning_steps
