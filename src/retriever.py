"""
Semantic Retriever with Optional Cross-Encoder Reranking

Retrieves relevant evidence chunks from the FAISS index using
semantic search, with optional cross-encoder reranking.
"""

import os
import pickle
import numpy as np
import faiss  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore[import-not-found]

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    EMBEDDINGS_DIR, EMBEDDING_MODEL,
    TOP_K, RETRIEVE_TOP_K, RERANK_ENABLED, RERANKER_MODEL,
    get_embeddings_dir,
)


class SemanticRetriever:
    """
    Retrieves relevant chunks from the FAISS index.
    Supports:
    - Basic semantic retrieval (embedding similarity)
    - Cross-encoder reranking for improved precision
    - Metadata filtering (by company, year)
    """

    def __init__(self, index_dir: str = None, year: int = None):
        """
        Initialize the retriever by loading the FAISS index and metadata.

        Args:
            index_dir: Directory containing the FAISS index files
            year: Fiscal year to load index for (uses year-specific dir)
        """
        if index_dir is None:
            if year is not None:
                # Try year-specific directory first
                year_dir = get_embeddings_dir(year)
                if os.path.exists(os.path.join(year_dir, "faiss_index.bin")):
                    index_dir = year_dir
                else:
                    index_dir = EMBEDDINGS_DIR
            else:
                index_dir = EMBEDDINGS_DIR

        self.index_dir = index_dir
        self.year = year
        self.index = None
        self.chunk_metadata = None
        self.embedding_model = None
        self.reranker_model = None

        self._load_index()

    def _load_index(self):
        """Load the FAISS index and chunk metadata."""
        # Load FAISS index
        index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

        # Load chunk metadata
        meta_path = os.path.join(self.index_dir, "chunk_metadata.pkl")
        with open(meta_path, "rb") as f:
            self.chunk_metadata = pickle.load(f)
        print(f"Loaded metadata for {len(self.chunk_metadata)} chunks")

    def _get_embedding_model(self):
        """Lazy-load the embedding model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print(f"Loaded embedding model: {EMBEDDING_MODEL}")
        return self.embedding_model

    def _get_reranker_model(self):
        """Lazy-load the cross-encoder reranker."""
        if self.reranker_model is None and RERANK_ENABLED:
            self.reranker_model = CrossEncoder(RERANKER_MODEL)
            print(f"Loaded reranker: {RERANKER_MODEL}")
        return self.reranker_model

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        For bge models, queries should be prefixed with "Represent this sentence: "
        """
        model = self._get_embedding_model()

        # BGE models recommend a query prefix
        if "bge" in EMBEDDING_MODEL.lower():
            query = "Represent this sentence: " + query

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        company_filter: str = None,
        year_filter: str = None,
        rerank: bool = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            top_k: Number of results to return (default from config)
            company_filter: Only return chunks from this company (ticker)
            year_filter: Only return chunks from this year
            rerank: Whether to use cross-encoder reranking (default from config)

        Returns:
            List of dicts with chunk text, metadata, and scores
        """
        if top_k is None:
            top_k = TOP_K
        if rerank is None:
            rerank = RERANK_ENABLED

        # Determine how many to initially retrieve
        initial_k = RETRIEVE_TOP_K if rerank else top_k

        # If we're filtering, retrieve more to account for filtered-out results
        if company_filter or year_filter:
            initial_k = min(initial_k * 3, self.index.ntotal)

        # Embed the query
        query_embedding = self.embed_query(query)

        # FAISS search
        scores, indices = self.index.search(query_embedding, initial_k)
        scores = scores[0]
        indices = indices[0]

        # Build results with metadata
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:  # FAISS returns -1 for not-found
                continue

            meta = self.chunk_metadata.get(int(idx), {})

            # Apply filters
            if company_filter and meta.get("company") != company_filter:
                continue
            if year_filter and meta.get("year") != year_filter:
                continue

            results.append({
                "chunk_id": meta.get("chunk_id", ""),
                "company": meta.get("company", ""),
                "company_name": meta.get("company_name", ""),
                "year": meta.get("year", ""),
                "section": meta.get("section", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "text": meta.get("text", ""),
                "faiss_score": float(score),
                "faiss_rank": len(results) + 1,
            })

        # Rerank if enabled
        if rerank and results:
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        return results

    def _rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """
        Rerank results using a cross-encoder model.
        Cross-encoders jointly encode query-document pairs for more accurate scoring.
        """
        reranker = self._get_reranker_model()
        if reranker is None:
            return results[:top_k]

        # Prepare query-document pairs
        pairs = [(query, r["text"]) for r in results]

        # Score with cross-encoder
        rerank_scores = reranker.predict(pairs)

        # Add rerank scores and sort
        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Update ranks
        for i, r in enumerate(results[:top_k]):
            r["final_rank"] = i + 1

        return results[:top_k]

    def retrieve_for_risk_category(
        self,
        risk_category: dict,
        company: str = None,
        top_k: int = None,
    ) -> list[dict]:
        """
        Retrieve evidence for a specific risk category using multiple query templates.
        Aggregates and deduplicates results across all queries.

        Args:
            risk_category: Risk category dict from config (with query_templates)
            company: Optional company ticker filter
            top_k: Number of final results to return

        Returns:
            Deduplicated and ranked list of evidence chunks
        """
        if top_k is None:
            top_k = TOP_K

        all_results = {}

        for query_template in risk_category["query_templates"]:
            results = self.retrieve(
                query=query_template,
                top_k=top_k,
                company_filter=company,
                rerank=RERANK_ENABLED,
            )
            for r in results:
                chunk_id = r["chunk_id"]
                if chunk_id not in all_results:
                    all_results[chunk_id] = r
                    all_results[chunk_id]["query_matches"] = 1
                else:
                    # Chunk appeared in multiple queries — boost its score
                    all_results[chunk_id]["query_matches"] += 1
                    # Keep the highest score
                    score_key = "rerank_score" if "rerank_score" in r else "faiss_score"
                    if r[score_key] > all_results[chunk_id].get(score_key, 0):
                        all_results[chunk_id][score_key] = r[score_key]

        # Sort by query_matches (more = better), then by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: (
                x.get("query_matches", 1),
                x.get("rerank_score", x.get("faiss_score", 0)),
            ),
            reverse=True,
        )

        return sorted_results[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int = None,
        company_filter: str = None,
    ) -> list[dict]:
        """
        Baseline keyword search using simple term matching.
        Used as a comparison baseline for evaluation.
        """
        if top_k is None:
            top_k = TOP_K

        query_terms = set(query.lower().split())
        results = []

        for idx, meta in self.chunk_metadata.items():
            if company_filter and meta.get("company") != company_filter:
                continue

            text_lower = meta.get("text", "").lower()
            # Count matching terms
            matches = sum(1 for term in query_terms if term in text_lower)

            if matches > 0:
                results.append({
                    "chunk_id": meta.get("chunk_id", ""),
                    "company": meta.get("company", ""),
                    "company_name": meta.get("company_name", ""),
                    "year": meta.get("year", ""),
                    "section": meta.get("section", ""),
                    "chunk_index": meta.get("chunk_index", -1),
                    "text": meta.get("text", ""),
                    "keyword_score": matches / len(query_terms),
                    "keyword_matches": matches,
                })

        results.sort(key=lambda x: x["keyword_score"], reverse=True)
        return results[:top_k]


def test_retrieval():
    """Quick test of the retrieval pipeline."""
    retriever = SemanticRetriever()

    test_queries = [
        "supply chain disruption risk",
        "cybersecurity data breach",
        "government regulation compliance",
        "competition and market share",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)
        for r in results:
            score_type = "rerank" if "rerank_score" in r else "faiss"
            score = r.get("rerank_score", r.get("faiss_score", 0))
            print(f"\n  [{r['company']}] Score={score:.4f} ({score_type})")
            print(f"  {r['text'][:200]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Semantic Retriever Test")
    print("=" * 60)
    test_retrieval()
