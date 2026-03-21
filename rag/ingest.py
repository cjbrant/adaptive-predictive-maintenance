"""Ingest OEM PDF documents into ChromaDB vector store.

Extracts text from real SKF PDFs, chunks intelligently around document
structure (prose sections, product tables, mixed content), embeds with
sentence-transformers, and stores in a persistent ChromaDB collection.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from rag.pdf_extract import extract_all_pdfs, is_table_page, find_table_header


DEFAULT_OEM_DIR = Path("data/oem")
DEFAULT_DB_DIR = Path("data/vectorstore")
COLLECTION_NAME = "oem_bearing_specs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Target chunk sizes in words
MIN_CHUNK_WORDS = 60
TARGET_CHUNK_WORDS = 250
MAX_CHUNK_WORDS = 500
TABLE_ROWS_PER_CHUNK = 18


def _detect_section_header(line: str, prev_line: str, next_line: str) -> bool:
    """Heuristic to detect section headers in SKF documents."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    # Numbered sections: "1 Bearing life", "2.3 Lubrication"
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
        return True
    # Short title-case or uppercase lines preceded by blank
    if (
        len(stripped) < 80
        and (not prev_line.strip())
        and not re.match(r"^\d+[\s,.]", stripped)  # not a table row
        and (stripped.istitle() or stripped.isupper())
    ):
        return True
    return False


def _word_count(text: str) -> int:
    return len(text.split())


def _chunk_prose(
    text: str,
    source: str,
    page: int,
    base_header: str,
) -> list[dict]:
    """Chunk prose content by section headers and paragraph boundaries."""
    lines = text.split("\n")
    chunks = []
    current_section = base_header
    current_buf: list[str] = []

    def flush_buffer():
        nonlocal current_buf
        content = "\n".join(current_buf).strip()
        if _word_count(content) >= MIN_CHUNK_WORDS:
            chunks.append(
                {
                    "text": content,
                    "source_file": source,
                    "page": page,
                    "section_header": current_section,
                    "content_type": "prose",
                }
            )
        elif chunks and _word_count(content) > 0:
            # Merge short trailing fragment into previous chunk
            chunks[-1]["text"] += "\n\n" + content
        current_buf = []

    for i, line in enumerate(lines):
        prev = lines[i - 1] if i > 0 else ""
        nxt = lines[i + 1] if i < len(lines) - 1 else ""

        if _detect_section_header(line, prev, nxt):
            # Flush current buffer before starting new section
            if _word_count("\n".join(current_buf)) >= MIN_CHUNK_WORDS:
                flush_buffer()
            current_section = line.strip()
            current_buf.append(line)
            continue

        current_buf.append(line)

        # Check if buffer exceeds target size at a paragraph boundary
        buf_text = "\n".join(current_buf)
        if _word_count(buf_text) >= TARGET_CHUNK_WORDS:
            # Look for a paragraph break (blank line) near the end
            if not line.strip():
                flush_buffer()
            elif _word_count(buf_text) >= MAX_CHUNK_WORDS:
                flush_buffer()

    # Flush remainder
    flush_buffer()
    return chunks


def _is_table_data_line(line: str) -> bool:
    """Check if a line is a product table data row (tab-delimited numbers + designation)."""
    stripped = line.strip()
    if not stripped or "\t" not in stripped:
        return False
    parts = [p.strip() for p in stripped.split("\t") if p.strip()]
    if len(parts) < 3:
        return False
    # Count numeric-looking parts (digits, commas, spaces within numbers)
    numeric_count = sum(1 for p in parts if re.match(r"^[\d\s,.]+$", p))
    return numeric_count >= 3


def _chunk_table(
    text: str,
    source: str,
    page: int,
    section_header: str,
) -> list[dict]:
    """Chunk product table content, keeping header with each chunk."""
    lines = text.split("\n")

    # Find the table header
    header = find_table_header(lines)

    # Separate header/metadata lines from data rows.
    # In reconstructed tables, data rows are tab-delimited with many numeric values.
    data_rows: list[str] = []
    preamble_lines: list[str] = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_table_data_line(line):
            data_rows.append(stripped)
            in_data = True
        elif in_data:
            # Non-data line after data started (e.g., footnote, page footer)
            # Keep it as a separator in the data
            if re.match(r"^\*\s", stripped) or len(stripped) < 30:
                data_rows.append(stripped)
            else:
                data_rows.append(stripped)
        else:
            preamble_lines.append(stripped)

    # Build context prefix from preamble
    preamble_text = " | ".join(preamble_lines[:5]) if preamble_lines else ""
    context_prefix = f"{section_header}\n{preamble_text}".strip()
    if header:
        context_prefix += f"\nColumns: {header}"

    # Chunk data rows
    chunks = []
    for start in range(0, len(data_rows), TABLE_ROWS_PER_CHUNK):
        end = min(start + TABLE_ROWS_PER_CHUNK, len(data_rows))
        # Include overlap: repeat last 2 rows of previous chunk
        overlap_start = max(0, start - 2) if start > 0 else 0
        chunk_rows = data_rows[overlap_start:end]
        chunk_text = context_prefix + "\n" + "\n".join(chunk_rows)

        chunks.append(
            {
                "text": chunk_text,
                "source_file": source,
                "page": page,
                "section_header": section_header,
                "content_type": "table",
            }
        )

    # If no data rows were found, treat the whole page as a single chunk
    if not chunks and _word_count(text) >= MIN_CHUNK_WORDS:
        chunks.append(
            {
                "text": text,
                "source_file": source,
                "page": page,
                "section_header": section_header,
                "content_type": "table",
            }
        )

    return chunks


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Chunk extracted PDF pages into retrieval-ready segments.

    Dispatches to prose or table chunking based on page content.
    Merges short pages with neighbors from the same document.
    """
    all_chunks = []

    # Group pages by source document for merging
    by_source: dict[str, list[dict]] = {}
    for page in pages:
        by_source.setdefault(page["source"], []).append(page)

    for source, doc_pages in by_source.items():
        pending_short = ""
        pending_header = ""
        pending_page = 0

        for pg in doc_pages:
            text = pg["text"]
            page_num = pg["page"]

            # Detect content type
            if pg["is_table"]:
                # Flush any pending prose
                if _word_count(pending_short) >= MIN_CHUNK_WORDS:
                    all_chunks.append(
                        {
                            "text": pending_short.strip(),
                            "source_file": source,
                            "page": pending_page,
                            "section_header": pending_header,
                            "content_type": "prose",
                        }
                    )
                    pending_short = ""

                # Determine section header from text
                header = _extract_page_section_header(text)
                chunks = _chunk_table(text, source, page_num, header or "Product table")
                all_chunks.extend(chunks)
            else:
                header = _extract_page_section_header(text)
                wc = _word_count(text)

                if wc < MIN_CHUNK_WORDS:
                    # Accumulate short pages
                    pending_short += "\n\n" + text
                    if not pending_header:
                        pending_header = header or ""
                    if not pending_page:
                        pending_page = page_num
                    if _word_count(pending_short) >= TARGET_CHUNK_WORDS:
                        all_chunks.extend(
                            _chunk_prose(
                                pending_short.strip(),
                                source,
                                pending_page,
                                pending_header,
                            )
                        )
                        pending_short = ""
                        pending_header = ""
                        pending_page = 0
                else:
                    # Flush pending short content first
                    if pending_short:
                        combined = pending_short + "\n\n" + text
                        all_chunks.extend(
                            _chunk_prose(
                                combined.strip(),
                                source,
                                pending_page or page_num,
                                pending_header or header or "",
                            )
                        )
                        pending_short = ""
                        pending_header = ""
                        pending_page = 0
                    else:
                        all_chunks.extend(
                            _chunk_prose(text, source, page_num, header or "")
                        )

        # Flush final pending
        if _word_count(pending_short) >= MIN_CHUNK_WORDS // 2:
            all_chunks.append(
                {
                    "text": pending_short.strip(),
                    "source_file": source,
                    "page": pending_page,
                    "section_header": pending_header,
                    "content_type": "mixed",
                }
            )

    # Add prose overlap: for consecutive prose chunks from the same source,
    # prepend the last sentence of the previous chunk
    for i in range(1, len(all_chunks)):
        prev = all_chunks[i - 1]
        curr = all_chunks[i]
        if (
            prev["source_file"] == curr["source_file"]
            and prev["content_type"] == "prose"
            and curr["content_type"] == "prose"
        ):
            # Extract last sentence of previous chunk
            sentences = re.split(r"(?<=[.!?])\s+", prev["text"])
            if len(sentences) >= 2:
                overlap = sentences[-1]
                curr["text"] = f"[...] {overlap}\n\n{curr['text']}"

    # Assign sequential chunk IDs
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    return all_chunks


def _extract_page_section_header(text: str) -> str | None:
    """Try to extract a section header from the first few lines of a page."""
    lines = text.split("\n")
    for i, line in enumerate(lines[:5]):
        stripped = line.strip()
        if not stripped:
            continue
        prev = lines[i - 1].strip() if i > 0 else ""
        nxt = lines[i + 1].strip() if i < len(lines) - 1 else ""
        if _detect_section_header(line, prev, nxt):
            return stripped
    # Fallback: use first non-empty short line
    for line in lines[:3]:
        stripped = line.strip()
        if stripped and len(stripped) < 80 and not re.match(r"^\d+$", stripped):
            return stripped
    return None


def ingest_oem_documents(
    oem_dir: str | Path = DEFAULT_OEM_DIR,
    db_dir: str | Path = DEFAULT_DB_DIR,
    model_name: str = EMBEDDING_MODEL,
) -> tuple[chromadb.Collection, int, list[dict]]:
    """
    Ingest all PDF files from the OEM directory into ChromaDB.

    Returns (collection, n_chunks, chunk_inventory).
    """
    oem_dir = Path(oem_dir)
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Extract text from all PDFs
    print("--- PDF Text Extraction ---")
    pages = extract_all_pdfs(oem_dir)
    if not pages:
        raise FileNotFoundError(f"No PDF content extracted from {oem_dir}")
    print(f"Total pages with text: {len(pages)}")

    # Chunk
    print("\n--- Chunking ---")
    all_chunks = chunk_pages(pages)
    print(f"Total chunks: {len(all_chunks)}")

    # Stats
    type_counts = {}
    word_counts = []
    for c in all_chunks:
        ct = c["content_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
        word_counts.append(_word_count(c["text"]))

    for ct, count in sorted(type_counts.items()):
        print(f"  {ct}: {count} chunks")
    print(f"  Average chunk size: {sum(word_counts) / len(word_counts):.0f} words")
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}")

    # Load embedding model
    print(f"\n--- Embedding ({model_name}) ---")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Initialize ChromaDB
    print("\n--- Storing in ChromaDB ---")
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i:04d}" for i in range(len(all_chunks))]
    metadatas = [
        {
            "source_file": c["source_file"],
            "page": c["page"],
            "section_header": c.get("section_header", ""),
            "content_type": c["content_type"],
            "chunk_index": c["chunk_index"],
        }
        for c in all_chunks
    ]

    # ChromaDB has a batch size limit; add in batches
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"Ingested {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'")

    # Build chunk inventory
    inventory = []
    for i, c in enumerate(all_chunks):
        inventory.append(
            {
                "chunk_id": ids[i],
                "source": c["source_file"],
                "page": c["page"],
                "section_header": c.get("section_header", ""),
                "content_type": c["content_type"],
                "word_count": _word_count(c["text"]),
                "first_80_chars": c["text"][:80].replace("\n", " "),
            }
        )

    return collection, len(all_chunks), inventory


def save_chunk_inventory(
    inventory: list[dict],
    output_path: str | Path = "analysis/chunk_inventory.csv",
) -> None:
    """Save chunk inventory to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(inventory)
    df.to_csv(output_path, index=False)
    print(f"Saved chunk inventory ({len(df)} rows) to {output_path}")


if __name__ == "__main__":
    collection, n, inventory = ingest_oem_documents()
    save_chunk_inventory(inventory)
    print(f"\nDone. Collection has {collection.count()} documents.")
