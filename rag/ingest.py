"""Structure-aware chunking and ChromaDB ingestion for OEM bearing PDFs.

Second stage of the RAG pipeline:
    pdf_extract -> **ingest** -> retrieve -> extract_params

Handles three content types:
- Prose (theory, failure descriptions): split on section headers, 200-400 words
- Tables (bearing specs): preserve header rows, ~15-20 rows per chunk
- Mixed/short (appendices, glossaries): page-boundary chunking
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag.pdf_extract import extract_all_pdfs

# ---------------------------------------------------------------------------
# Manufacturer detection
# ---------------------------------------------------------------------------

_KNOWN_MANUFACTURERS = ["SKF", "Rexnord", "LDK"]


def detect_manufacturer(filename: str, first_page_text: str) -> str:
    """Detect bearing manufacturer from filename or first-page text.

    Checks filename first, then first-page text for known manufacturer names.
    Returns 'Unknown' if no match found.
    """
    fn_upper = filename.upper()
    text_upper = first_page_text.upper()
    for mfr in _KNOWN_MANUFACTURERS:
        if mfr.upper() in fn_upper or mfr.upper() in text_upper:
            return mfr
    return "Unknown"


# ---------------------------------------------------------------------------
# Content-type classification
# ---------------------------------------------------------------------------

def classify_content_type(text: str) -> str:
    """Classify a page/block as 'table', 'prose', or 'mixed'.

    Table heuristic: >40% of characters are digits, and multiple lines
    match the pattern of a designation followed by numbers.
    """
    if not text.strip():
        return "mixed"

    # Count digit fraction
    total_chars = len(re.sub(r"\s", "", text))
    if total_chars == 0:
        return "mixed"
    digit_chars = sum(1 for c in text if c.isdigit())
    digit_frac = digit_chars / total_chars

    # Count lines matching "designation + numbers" pattern
    # e.g. "6205 25 52 15 14.8 7.8 0.335"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    table_line_pattern = re.compile(
        r"^[\w\-/]+\s+[\d.]+(?:\s+[\d.]+){2,}"
    )
    table_lines = sum(1 for l in lines if table_line_pattern.match(l))
    table_line_frac = table_lines / max(len(lines), 1)

    if digit_frac > 0.40 and table_line_frac > 0.30:
        return "table"
    if digit_frac > 0.50:
        return "table"
    if table_line_frac > 0.50:
        return "table"

    return "prose"


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\s+\S"  # numbered: "1.2 Title" or "3 Title"
    r"|[A-Z][A-Z\s]{4,}$"  # ALL CAPS lines (at least 5 chars)
    r")",
    re.MULTILINE,
)


def _detect_section_header(text: str) -> str:
    """Extract the most likely section header from a chunk of text."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _SECTION_HEADER_RE.match(line):
            return line[:120]
    return ""


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------

def chunk_prose(text: str, target_words: int = 300) -> list[str]:
    """Split prose text into chunks of approximately *target_words* words.

    Strategy:
    1. Split on section headers (numbered, ALL CAPS, or short lines before
       paragraphs).
    2. Merge sections under 100 words with the next section.
    3. If a section exceeds target_words * 1.5, split on paragraph breaks.
    4. Add 1-2 sentence overlap between consecutive chunks.
    """
    if not text.strip():
        return []

    # Split on section headers
    sections = _split_on_headers(text)

    # Merge small sections
    sections = _merge_small_sections(sections, min_words=100)

    # Split oversized sections
    max_words = int(target_words * 1.5)
    split_sections: list[str] = []
    for sec in sections:
        wc = len(sec.split())
        if wc > max_words:
            split_sections.extend(_split_by_paragraphs(sec, target_words))
        else:
            split_sections.append(sec)

    # Merge any remaining tiny chunks
    split_sections = _merge_small_sections(split_sections, min_words=100)

    # Add overlap
    chunks = _add_overlap(split_sections)
    return chunks


def _split_on_headers(text: str) -> list[str]:
    """Split text at lines matching section header patterns."""
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and _SECTION_HEADER_RE.match(stripped) and current:
            sections.append("".join(current).strip())
            current = []
        current.append(line)

    if current:
        sections.append("".join(current).strip())

    return [s for s in sections if s.strip()]


def _merge_small_sections(sections: list[str], min_words: int) -> list[str]:
    """Merge sections with fewer than *min_words* words into the next section."""
    if not sections:
        return []
    merged: list[str] = []
    buf = sections[0]
    for sec in sections[1:]:
        if len(buf.split()) < min_words:
            buf = buf + "\n\n" + sec
        else:
            merged.append(buf)
            buf = sec
    merged.append(buf)
    return merged


def _split_by_paragraphs(text: str, target_words: int) -> list[str]:
    """Split a long text block into chunks at paragraph boundaries.

    Falls back to word-boundary splitting if there are no paragraph breaks.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If there's only one "paragraph" (no breaks), split on word boundaries
    if len(paragraphs) <= 1:
        return _split_by_words(text, target_words)

    chunks: list[str] = []
    current: list[str] = []
    current_wc = 0

    for para in paragraphs:
        pwc = len(para.split())
        if current_wc + pwc > target_words and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_wc = pwc
        else:
            current.append(para)
            current_wc += pwc

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_by_words(text: str, target_words: int) -> list[str]:
    """Split text into chunks of *target_words* at word boundaries."""
    words = text.split()
    if len(words) <= target_words:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), target_words):
        chunk = " ".join(words[i : i + target_words])
        chunks.append(chunk)
    return chunks


def _last_sentences(text: str, n: int = 2) -> str:
    """Extract the last *n* sentences from text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    overlap = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(overlap)


def _add_overlap(sections: list[str]) -> list[str]:
    """Prepend 1-2 sentence overlap from previous chunk."""
    if len(sections) <= 1:
        return sections
    result = [sections[0]]
    for i in range(1, len(sections)):
        overlap = _last_sentences(sections[i - 1], 2)
        result.append(overlap + "\n\n" + sections[i])
    return result


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------

def chunk_table(text: str, rows_per_chunk: int = 18) -> list[str]:
    """Chunk tabular text into logical blocks.

    Two strategies:
    1. Block-aware: if the table has clear block separators (blank lines
       or designation-like lines), split on those boundaries so that each
       bearing's data stays together.
    2. Row-count fallback: split every *rows_per_chunk* rows, prepending
       the header to each chunk.
    """
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return []

    # Strategy 1: detect logical blocks separated by blank lines
    # This handles catalogs like Rexnord where each bearing size is
    # separated by blank lines and blocks are 30-50 lines long.
    blocks = _split_into_logical_blocks(lines)
    if blocks and len(blocks) >= 2 and all(len(b) <= 800 for b in blocks):
        return blocks

    # Strategy 2: simple row-count chunking with header preservation
    header = non_empty[0]
    data_rows = non_empty[1:]

    if not data_rows:
        return [header]

    chunks: list[str] = []
    for start in range(0, len(data_rows), rows_per_chunk):
        batch = data_rows[start : start + rows_per_chunk]
        chunk = header + "\n" + "\n".join(batch)
        chunks.append(chunk)

    return chunks


def _split_into_logical_blocks(lines: list[str]) -> list[str]:
    """Split table lines into logical blocks at blank-line boundaries.

    Groups consecutive non-empty lines together. Merges very small
    blocks (< 5 lines) with the next block. Returns chunk strings
    or empty list if the text doesn't have clear block structure.
    """
    raw_blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if not line.strip():
            if current:
                raw_blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        raw_blocks.append(current)

    if len(raw_blocks) < 2:
        return []

    # Merge very small blocks with the next block
    merged: list[list[str]] = []
    for block in raw_blocks:
        if merged and len(merged[-1]) < 5:
            merged[-1].extend(block)
        else:
            merged.append(list(block))

    # Convert to strings, skip trivially small blocks
    result = []
    for block in merged:
        text = "\n".join(block)
        if len(text.strip()) >= 20:
            result.append(text)

    return result


# ---------------------------------------------------------------------------
# Page-level chunking dispatcher
# ---------------------------------------------------------------------------

def chunk_page(
    text: str,
    content_type: str,
    section_header: str,
) -> list[str]:
    """Chunk a single page based on its content type.

    Returns a list of chunk strings. The *section_header* is informational
    metadata only and is NOT prepended to the chunk text here (it is stored
    in metadata instead).
    """
    if content_type == "table":
        return chunk_table(text)
    elif content_type == "prose":
        return chunk_prose(text)
    else:  # mixed
        # Chunk at ~300 words, merge short pages handled upstream
        words = text.split()
        if len(words) <= 400:
            return [text] if text.strip() else []
        return chunk_prose(text, target_words=300)


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_oem_pdfs(
    oem_dir: str | Path = "data/oem",
    db_path: str | Path = "data/processed/chromadb",
    collection_name: str = "oem_bearings",
) -> dict:
    """Extract, chunk, embed, and store OEM bearing PDFs in ChromaDB.

    Parameters
    ----------
    oem_dir : directory containing OEM PDF files
    db_path : path for the persistent ChromaDB store
    collection_name : name of the ChromaDB collection

    Returns
    -------
    Dict with ingestion statistics: total_chunks, total_pages, per_file, etc.
    """
    oem_dir = Path(oem_dir)
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract text from all PDFs
    all_docs = extract_all_pdfs(oem_dir)

    # 2. Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 3. Create/open ChromaDB
    client = chromadb.PersistentClient(path=str(db_path))
    # Delete existing collection if present so re-runs are idempotent
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Chunk and embed
    all_chunks_meta: list[dict] = []
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []

    total_pages = 0
    per_file_stats: dict[str, dict] = {}

    for filename, pages in all_docs.items():
        first_page_text = pages[0]["text"] if pages else ""
        manufacturer = detect_manufacturer(filename, first_page_text)
        file_chunk_count = 0
        total_pages += len(pages)

        # Track current section header across pages
        current_section = ""

        for page_info in pages:
            page_num = page_info["page"]
            text = page_info["text"]

            # Detect section header
            detected = _detect_section_header(text)
            if detected:
                current_section = detected

            # Classify content type
            ctype = classify_content_type(text)

            # Chunk
            chunks = chunk_page(text, ctype, current_section)

            for ci, chunk_text in enumerate(chunks):
                chunk_id = f"{filename}__p{page_num}__c{ci}"
                meta = {
                    "source": filename,
                    "page": page_num,
                    "section_header": current_section,
                    "content_type": ctype,
                    "manufacturer": manufacturer,
                }

                all_ids.append(chunk_id)
                all_texts.append(chunk_text)
                all_metadatas.append(meta)
                all_chunks_meta.append({
                    "id": chunk_id,
                    **meta,
                    "word_count": len(chunk_text.split()),
                })
                file_chunk_count += 1

        per_file_stats[filename] = {
            "pages": len(pages),
            "chunks": file_chunk_count,
            "manufacturer": manufacturer,
        }

    # 5. Embed in batches
    batch_size = 64
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)

    # 6. Upsert into ChromaDB
    # ChromaDB has a batch limit; upsert in chunks of 5000
    for i in range(0, len(all_ids), 5000):
        collection.upsert(
            ids=all_ids[i : i + 5000],
            embeddings=all_embeddings[i : i + 5000],
            documents=all_texts[i : i + 5000],
            metadatas=all_metadatas[i : i + 5000],
        )

    # 7. Save chunk inventory CSV
    inventory_path = Path("analysis/chunk_inventory.csv")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    if all_chunks_meta:
        fieldnames = list(all_chunks_meta[0].keys())
        with open(inventory_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_chunks_meta)

    # 8. Print summary
    stats = {
        "total_chunks": len(all_ids),
        "total_pages": total_pages,
        "per_file": per_file_stats,
        "db_path": str(db_path),
        "collection": collection_name,
    }

    print(f"\n{'='*60}")
    print("RAG Ingestion Summary")
    print(f"{'='*60}")
    print(f"  Total PDFs:   {len(all_docs)}")
    print(f"  Total pages:  {total_pages}")
    print(f"  Total chunks: {len(all_ids)}")
    print(f"  ChromaDB:     {db_path / collection_name}")
    print(f"  Inventory:    {inventory_path}")
    for fname, fstats in per_file_stats.items():
        print(f"  {fname}: {fstats['pages']} pages, {fstats['chunks']} chunks "
              f"({fstats['manufacturer']})")
    print(f"{'='*60}\n")

    return stats
