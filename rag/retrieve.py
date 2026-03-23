"""Hybrid semantic + text retrieval for OEM bearing specifications.

Third stage of the RAG pipeline:
    pdf_extract -> ingest -> **retrieve** -> extract_params

Combines dense (embedding) search with designation-aware query expansion
and exact text matching to improve recall for bearing part numbers.
"""

from __future__ import annotations

import re

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Known bearing designations and their manufacturers
# ---------------------------------------------------------------------------

KNOWN_DESIGNATIONS: dict[str, str] = {
    "6205": "SKF", "6205-2RS": "SKF",
    "6204": "SKF", "6204-2RS": "SKF",
    "ZA-2115": "Rexnord", "ZA2115": "Rexnord",
    "UER204": "LDK",
}

DESIGNATION_PATTERN = re.compile(
    r"\b(6[12]\d{2}(?:-2RS)?|ZA-?2115|UER\d{3})\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_designation(query: str) -> str | None:
    """Extract a bearing designation from a query string.

    Returns the first matched designation in its canonical form, or None.
    """
    m = DESIGNATION_PATTERN.search(query)
    if m:
        return m.group(1)
    return None


_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def _get_collection(
    db_path: str = "data/processed/chromadb",
    collection_name: str = "oem_bearings",
) -> chromadb.Collection:
    """Open and return a ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(name=collection_name)


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    k: int = 5,
    expand: bool = True,
    db_path: str = "data/processed/chromadb",
    collection_name: str = "oem_bearings",
) -> list[dict]:
    """Hybrid retrieval: semantic search + designation-aware expansion.

    1. Run semantic search on the query.
    2. If the query contains a bearing designation (e.g. "6205", "ZA-2115"):
       - Run expanded semantic queries for specs and manufacturer context.
       - Scan all table chunks for the exact designation string.
       - Boost chunks containing the exact designation by 1.3x similarity.
    3. Merge, deduplicate by chunk_id, return top-K by best score.

    Each result dict has: chunk_id, text, metadata, score.
    """
    collection = _get_collection(db_path, collection_name)
    model = _get_embedding_model()

    # ---- 1. Primary semantic search ----
    n_results = k * 3  # fetch extra candidates for re-ranking
    query_embedding = model.encode(query).tolist()
    primary = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Build a results dict keyed by chunk_id -> best score
    candidates: dict[str, dict] = {}
    _merge_chroma_results(candidates, primary)

    # ---- 2. Designation-aware expansion ----
    designation = _extract_designation(query) if expand else None

    if designation:
        # Look up manufacturer
        # Normalise: try with and without dash for ZA variants
        manufacturer = (
            KNOWN_DESIGNATIONS.get(designation)
            or KNOWN_DESIGNATIONS.get(designation.upper())
            or ""
        )

        # Expanded semantic queries
        expanded_queries = [
            f"{designation} specifications dimensions load rating",
            f"{manufacturer} {designation} bearing".strip(),
        ]
        for eq in expanded_queries:
            eq_embedding = model.encode(eq).tolist()
            expanded = collection.query(
                query_embeddings=[eq_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            _merge_chroma_results(candidates, expanded)

        # Exact text scan: search ALL chunks for the designation string
        # (not just tables — OCR'd text may be classified differently)
        try:
            text_results = collection.get(
                include=["documents", "metadatas"],
            )
            if text_results and text_results["ids"]:
                desig_upper = designation.upper()
                base_desig = re.sub(r"-2RS$", "", desig_upper, flags=re.IGNORECASE)
                # Also try numeric core (e.g., "2115" from "ZA-2115")
                numeric_core = re.sub(r"^[A-Z]{1,3}-?", "", desig_upper)

                # Build search variants including OCR fuzzy matches
                search_terms = {desig_upper, base_desig}
                if numeric_core and len(numeric_core) >= 3:
                    search_terms.add(numeric_core)
                # OCR misreadings for common prefixes
                if desig_upper.startswith("UER"):
                    suffix = desig_upper[3:]
                    for v in ["UER", "VER", "UFR", "UBR"]:
                        search_terms.add(v + suffix)

                for i, doc in enumerate(text_results["documents"]):
                    doc_upper = doc.upper()
                    if any(term in doc_upper for term in search_terms):
                        cid = text_results["ids"][i]
                        if cid not in candidates:
                            candidates[cid] = {
                                "chunk_id": cid,
                                "text": doc,
                                "metadata": text_results["metadatas"][i],
                                "score": 0.85,
                            }
                        else:
                            candidates[cid]["score"] = max(candidates[cid]["score"], 0.85)
        except Exception:
            pass

        # Boost chunks containing the exact designation
        desig_upper = designation.upper()
        base_desig = re.sub(r"-2RS$", "", desig_upper, flags=re.IGNORECASE)
        for cid, entry in candidates.items():
            text_upper = entry["text"].upper()
            if desig_upper in text_upper or base_desig in text_upper:
                entry["score"] = min(entry["score"] * 1.3, 1.0)

        # Extra boost for chunks that contain the designation AND
        # numerical values likely to be load ratings (numbers > 1000
        # with comma formatting, or kN values). This prioritizes
        # spec/data pages over dimensional/housing pages.
        for cid, entry in candidates.items():
            text = entry["text"]
            has_comma_thousands = bool(re.search(r"\d{1,3},\d{3}", text))
            has_kn_numbers = bool(re.search(r"\d+\.?\d*\s*(?:kN|KN|kn)", text))
            has_large_numbers = bool(re.search(r"\b\d{4,6}\b", text))
            if has_comma_thousands or has_kn_numbers:
                entry["score"] += 0.1  # strong signal: load ratings
            elif has_large_numbers:
                entry["score"] += 0.05  # moderate signal: numbers

    # ---- 3. Sort by score descending and return top-K ----
    ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:k]


def _merge_chroma_results(
    candidates: dict[str, dict],
    chroma_result: dict,
) -> None:
    """Merge ChromaDB query results into the candidates dict.

    Keeps the highest score (similarity = 1 - distance) for each chunk_id.
    """
    if not chroma_result or not chroma_result.get("ids"):
        return

    ids = chroma_result["ids"][0]
    docs = chroma_result["documents"][0]
    metas = chroma_result["metadatas"][0]
    dists = chroma_result["distances"][0]

    for i, cid in enumerate(ids):
        score = 1.0 - dists[i]  # cosine distance -> similarity
        if cid not in candidates or score > candidates[cid]["score"]:
            candidates[cid] = {
                "chunk_id": cid,
                "text": docs[i],
                "metadata": metas[i],
                "score": score,
            }
