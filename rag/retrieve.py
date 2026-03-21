"""Retrieve relevant chunks from the OEM vector store.

Supports query expansion (multiple reformulations of the same question)
and context-window retrieval (returning neighboring chunks for boundary-
spanning answers). Includes reranking with designation boosting for
bearing-specific queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.ingest import COLLECTION_NAME, DEFAULT_DB_DIR, EMBEDDING_MODEL


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""

    text: str
    source_file: str
    section_header: str
    page: int
    content_type: str
    chunk_index: int
    similarity: float
    is_context: bool = False


def get_retriever(
    db_dir: str | Path = DEFAULT_DB_DIR,
    model_name: str = EMBEDDING_MODEL,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """Load the ChromaDB collection and embedding model."""
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(model_name)
    return collection, model


def _extract_bearing_designation(query: str) -> str | None:
    """Extract a bearing model number from a query string."""
    match = re.search(r"\b(6[0-4]\d{2})\b", query)
    return match.group(1) if match else None


def _build_expanded_queries(query: str) -> list[str]:
    """Generate expanded query variants for better recall."""
    queries = [query]
    designation = _extract_bearing_designation(query)

    if designation:
        queries.append(f"SKF {designation} deep groove ball bearing specifications dimensions")
        queries.append(f"SKF {designation} dynamic load rating static load rating kN")

    query_lower = query.lower()
    if any(term in query_lower for term in ["failure", "damage", "spalling", "fatigue"]):
        queries.append("bearing failure modes progression stages spalling fatigue")
        queries.append("vibration signature bearing damage detection monitoring")

    if any(term in query_lower for term in ["life", "l10", "rating life"]):
        queries.append("bearing basic rating life calculation formula ISO 281 L10")
        queries.append("L10h operating hours dynamic load rating exponent")

    if any(term in query_lower for term in ["vibration", "severity", "zone", "10816", "threshold"]):
        queries.append("vibration severity zones monitoring condition bearing damage")

    return queries


def _raw_retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
) -> list[RetrievedChunk]:
    """Single-query retrieval against ChromaDB."""
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 - distance
        metadata = results["metadatas"][0][i]
        chunks.append(
            RetrievedChunk(
                text=results["documents"][0][i],
                source_file=metadata.get("source_file", ""),
                section_header=metadata.get("section_header", ""),
                page=metadata.get("page", 0),
                content_type=metadata.get("content_type", ""),
                chunk_index=metadata.get("chunk_index", 0),
                similarity=similarity,
            )
        )

    return chunks


def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 5,
    expand: bool = True,
) -> list[RetrievedChunk]:
    """
    Retrieve top-K chunks with optional query expansion and reranking.

    If expand=True, runs multiple query variants and merges results.
    Applies designation-based boosting for bearing-specific queries.
    """
    if expand:
        queries = _build_expanded_queries(query)
    else:
        queries = [query]

    # Retrieve from all queries
    seen_ids: dict[int, RetrievedChunk] = {}  # chunk_index -> best chunk
    for q in queries:
        hits = _raw_retrieve(q, collection, model, top_k=top_k * 2)
        for chunk in hits:
            idx = chunk.chunk_index
            if idx not in seen_ids or chunk.similarity > seen_ids[idx].similarity:
                seen_ids[idx] = chunk

    candidates = list(seen_ids.values())

    # Designation boosting: if query mentions a specific bearing, boost chunks
    # that contain that designation string
    designation = _extract_bearing_designation(query)
    if designation:
        for chunk in candidates:
            if designation in chunk.text:
                chunk.similarity *= 1.3
            # Extra boost for table chunks with the designation
            if designation in chunk.text and chunk.content_type == "table":
                chunk.similarity *= 1.1

    # Sort by similarity descending, return top-K
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates[:top_k]


def retrieve_with_context(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 3,
    context_window: int = 1,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks plus their neighbors from the same document.

    For each hit, also returns the chunks immediately before and after
    (by chunk_index within the same source file).
    """
    hits = retrieve(query, collection, model, top_k=top_k, expand=True)
    if context_window == 0:
        return hits

    # Gather neighbor chunk indices
    neighbor_indices = set()
    hit_indices = {c.chunk_index for c in hits}
    for chunk in hits:
        for offset in range(-context_window, context_window + 1):
            if offset == 0:
                continue
            neighbor_indices.add(chunk.chunk_index + offset)
    neighbor_indices -= hit_indices

    # Fetch neighbors from ChromaDB
    if neighbor_indices:
        neighbor_ids = [f"chunk_{idx:04d}" for idx in neighbor_indices if idx >= 0]
        try:
            neighbor_results = collection.get(
                ids=neighbor_ids,
                include=["documents", "metadatas"],
            )
            for i, doc_id in enumerate(neighbor_results["ids"]):
                metadata = neighbor_results["metadatas"][i]
                # Only include neighbors from the same source as a hit
                source = metadata.get("source_file", "")
                if any(h.source_file == source for h in hits):
                    hits.append(
                        RetrievedChunk(
                            text=neighbor_results["documents"][i],
                            source_file=source,
                            section_header=metadata.get("section_header", ""),
                            page=metadata.get("page", 0),
                            content_type=metadata.get("content_type", ""),
                            chunk_index=metadata.get("chunk_index", 0),
                            similarity=0.0,  # context chunks don't have a query similarity
                            is_context=True,
                        )
                    )
        except Exception:
            pass  # Neighbor IDs may not exist at boundaries

    return hits


def print_retrieval_results(query: str, chunks: list[RetrievedChunk]) -> None:
    """Pretty-print retrieval results."""
    print(f"\nQuery: {query}")
    print("=" * 70)
    for i, chunk in enumerate(chunks):
        ctx_label = " [CONTEXT]" if chunk.is_context else ""
        print(f"\n[{i+1}] Similarity: {chunk.similarity:.4f}{ctx_label}")
        print(f"    Source: {chunk.source_file} p.{chunk.page} | {chunk.content_type}")
        print(f"    Section: {chunk.section_header}")
        text_preview = chunk.text[:200].replace("\n", " ")
        print(f"    Text: {text_preview}...")
    print()
