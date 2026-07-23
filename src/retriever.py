"""
Agents.retriever
================
Simple vector store manager using sentence-transformers for embeddings
and FAISS for the local vector index.

This module is optional: the rest of the project falls back to the
existing chunk-only processing when FAISS or sentence-transformers is
not installed. It is designed to be minimal and beginner-friendly.

Usage:
    vm = VectorStoreManager(model_name='all-MiniLM-L6-v2', index_path='data/faiss_index')
    vm.build_index([{"text": chunk_text, "meta": {"chunk_index": i}} for i, chunk_text in enumerate(chunks)])
    results = vm.query("recent grocery charges", k=5)

Note: FAISS can be tricky to install on some Windows setups. If you
prefer, use the "Hybrid" option to keep simple Python chunking only.
"""
from typing import Any, Dict, List, Optional, Tuple
import os

# Lazy imports — fail with clear message if dependencies missing
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception as e:
    # We'll not raise here — callers should catch import-time problems and
    # provide friendly messaging. Keep the module importable.
    np = None
    faiss = None
    SentenceTransformer = None


class DependencyError(RuntimeError):
    pass


class VectorStoreManager:
    """Manage an in-memory FAISS index and associated metadata.

    This keeps things simple and does not attempt to persist embeddings in
    a production-ready manner. It is intended for local use and debugging.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None) -> None:
        if SentenceTransformer is None or faiss is None or np is None:
            raise DependencyError(
                "Missing dependencies for VectorStoreManager. Install faiss-cpu, sentence-transformers, and numpy to enable RAG features."
            )

        self.model_name = model_name
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)
        self.index = None  # faiss index
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        # L2-normalize vectors for cosine-similarity using inner-product index
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_index(self, docs: List[Dict[str, Any]]) -> None:
        """Build an index from a list of docs where each doc is {"text":..., "meta":{...}}.

        This overwrites any existing in-memory index.
        """
        if not docs:
            # Create empty index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            return

        texts = [d.get("text", "") for d in docs]
        self.metadata = [d.get("meta", {}) for d in docs]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = self._normalize(embeddings)

        # Build FAISS index for inner-product (cosine search with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Optionally persist index and metadata
        if self.index_path:
            os.makedirs(self.index_path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            # Save metadata
            import json
            with open(os.path.join(self.index_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k docs similar to the query_text.

        Each returned item is a dict: {"score": float, "text": str, "meta": {...}}
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() before query().")

        if not query_text:
            return []

        q_emb = self.model.encode([query_text], convert_to_numpy=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        q_emb = self._normalize(q_emb)

        D, I = self.index.search(q_emb, k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({"score": float(score), "meta": meta})

        return results


# Backwards compatibility helper
def build_and_query(docs: List[Dict[str, Any]], query: str, model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    v = VectorStoreManager(model_name=model_name)
    v.build_index(docs)
    return v.query(query)
