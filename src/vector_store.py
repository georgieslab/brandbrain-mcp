"""ChromaDB vector store integration for BrandBrain.

Manages persistent indexing and semantic retrieval of brand guidelines,
color palettes, forbidden terms, and visual art direction.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

from src.bedrock_client import BedrockClient

load_dotenv()

logger = logging.getLogger("brandbrain.vector_store")

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = "brand_guidelines"


class LocalDeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """Lightweight deterministic local fallback embedding function.

    Generates normalized 256-dimensional pseudo-embeddings using token hashing.
    Used when AWS Bedrock Titan credentials are not available or offline.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @classmethod
    def name(cls) -> str:
        return "local_deterministic"

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "LocalDeterministicEmbeddingFunction":
        return cls(dim=config.get("dim", 256))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = text.lower().split()
            if not tokens:
                embeddings.append(vec)
                continue

            for token in tokens:
                token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = token_hash % self.dim
                sign = 1.0 if ((token_hash >> 8) % 2 == 0) else -1.0
                vec[idx] += sign

            # Normalize vector to unit length
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings


class BedrockTitanEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function that wraps AWS Bedrock Amazon Titan Text Embeddings V2

    with automatic fallback to local embedding if AWS credentials or network fail.
    """

    def __init__(
        self,
        bedrock_client: Optional[BedrockClient] = None,
        fallback_enabled: bool = True,
    ) -> None:
        self.bedrock_client = bedrock_client or BedrockClient()
        self.fallback_enabled = fallback_enabled
        self._fallback_function = LocalDeterministicEmbeddingFunction(dim=256)
        self._use_fallback = not self.bedrock_client.has_credentials()

    @classmethod
    def name(cls) -> str:
        return "bedrock_titan"

    def get_config(self) -> Dict[str, Any]:
        return {
            "fallback_enabled": self.fallback_enabled,
        }

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "BedrockTitanEmbeddingFunction":
        return cls(fallback_enabled=config.get("fallback_enabled", True))

    def __call__(self, input: Documents) -> Embeddings:
        if self._use_fallback:
            return self._fallback_function(input)

        embeddings: List[List[float]] = []
        try:
            for text in input:
                emb = self.bedrock_client.get_embedding(text, dimensions=1024)
                embeddings.append(emb)
            return embeddings
        except Exception as exc:
            if self.fallback_enabled:
                logger.warning(
                    "Bedrock Titan embedding failed (%s). Falling back to local embeddings.",
                    exc,
                )
                self._use_fallback = True
                return self._fallback_function(input)
            raise


class BrandVectorStore:
    """Persistent ChromaDB vector database manager for brand guidelines."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        bedrock_client: Optional[BedrockClient] = None,
        embedding_function: Optional[EmbeddingFunction[Documents]] = None,
    ) -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.bedrock_client = bedrock_client or BedrockClient()
        self.embedding_function = embedding_function or BedrockTitanEmbeddingFunction(
            bedrock_client=self.bedrock_client
        )

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Brand guidelines, voice rules, palettes, and visual specs"},
        )

    def add_guidelines(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert guideline documents into ChromaDB."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Indexed %d guidelines into collection '%s'.", len(ids), COLLECTION_NAME)

    def query_guidelines(
        self,
        client_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for brand guidelines matching the client and query.

        Args:
            client_id: Client identifier (e.g. 'solaris-fintech').
            query: Semantic search query string.
            top_k: Maximum number of relevant chunks to retrieve.

        Returns:
            List of dictionaries containing document, metadata, and distance.
        """
        count = self.collection.count()
        if count == 0:
            return []

        # Query with metadata filtering by client_id
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            where={"client_id": client_id},
        )

        output: List[Dict[str, Any]] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            # If filtered query returns empty, attempt a relaxed query or return empty
            return output

        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = (
            results["distances"][0] if results.get("distances") and results["distances"] else [0.0] * len(documents)
        )
        ids = results["ids"][0] if results.get("ids") else [f"doc_{i}" for i in range(len(documents))]

        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            output.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            })

        return output

    def get_all_client_guidelines(self, client_id: str) -> List[Dict[str, Any]]:
        """Retrieve all guideline documents for a given client."""
        results = self.collection.get(
            where={"client_id": client_id},
            include=["documents", "metadatas"],
        )

        output: List[Dict[str, Any]] = []
        if not results or not results.get("documents"):
            return output

        documents = results["documents"]
        metadatas = results["metadatas"] if results.get("metadatas") else [{}] * len(documents)
        ids = results["ids"] if results.get("ids") else [f"doc_{i}" for i in range(len(documents))]

        for doc_id, doc, meta in zip(ids, documents, metadatas):
            output.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
            })
        return output

    def format_guidelines_for_context(self, guidelines: List[Dict[str, Any]]) -> str:
        """Format retrieved guideline items into a structured context string for prompt injection."""
        if not guidelines:
            return "No relevant brand guidelines found in knowledge base."

        formatted_sections: List[str] = []
        for idx, item in enumerate(guidelines, start=1):
            meta = item.get("metadata", {})
            section_title = meta.get("section", f"Rule Section {idx}")
            category = meta.get("category", "General")
            doc = item.get("document", "").strip()

            formatted_sections.append(
                f"### [{category.upper()}] {section_title}\n{doc}"
            )

        return "\n\n".join(formatted_sections)

