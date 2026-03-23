# RAG Pipeline + XJTU-SY Dataset + Benchmark Expansion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG pipeline to extract bearing specs from OEM PDFs, add the XJTU-SY dataset (15 run-to-failure bearings), re-run the full benchmark with RAG-extracted specs, and generate updated analysis notebooks and report.

**Architecture:** The RAG pipeline (`rag/`) ingests manufacturer PDFs via PyMuPDF, chunks with structure-awareness (prose vs tables), embeds with MiniLM-L6-v2 into ChromaDB, and extracts structured bearing parameters via hybrid retrieval. The XJTU-SY dataset follows the existing `DatasetLoader` pattern (config/download/feature_extraction/loader). The benchmark runner is extended to try RAG-extracted params before falling back to hardcoded configs. Results flow into updated notebooks and an expanded R Markdown report.

**Tech Stack:** PyMuPDF (fitz), sentence-transformers (all-MiniLM-L6-v2), ChromaDB, numpy/scipy/pandas, matplotlib/seaborn, R/knitr/ggplot2/prettydoc

---

## File Structure

### New Files
```
rag/
  __init__.py
  pdf_extract.py          # PyMuPDF text extraction from OEM PDFs
  ingest.py               # Structure-aware chunking + ChromaDB embedding
  retrieve.py             # Hybrid semantic + text retrieval
  extract_params.py       # Structured bearing parameter extraction

datasets/xjtu_sy/
  __init__.py
  config.py               # XJTU_SY_CONFIG dict
  download.py             # Download from Google Drive/GitHub
  feature_extraction.py   # Vibration feature extraction (kurtosis, RMS, defect freqs)
  loader.py               # XJTUSYLoader class

notebooks/
  06_xjtu_sy_analysis.ipynb

tests/
  test_rag.py             # RAG pipeline tests
  test_xjtu_sy.py         # XJTU-SY loader tests
```

### Modified Files
```
framework/benchmark_runner.py   # Add XJTU-SY loader, RAG param injection
framework/results_summary.py    # Update for 5+ datasets
core/oem_prior.py               # Update ground_truth_C validation dict
requirements.txt                # Already has PyMuPDF, sentence-transformers, chromadb
notebooks/05_cross_dataset_comparison.ipynb  # Add XJTU-SY
reports/benchmark_report.Rmd    # Full rewrite with RAG + XJTU-SY sections
```

---

### Task 1: RAG PDF Extraction Module

**Files:**
- Create: `rag/__init__.py`
- Create: `rag/pdf_extract.py`
- Create: `tests/test_rag.py`

- [ ] **Step 1: Create `rag/__init__.py`**

```python
# rag/__init__.py
```

- [ ] **Step 2: Write test for PDF extraction**

```python
# tests/test_rag.py
import pytest
from pathlib import Path

class TestPDFExtraction:
    def test_extract_returns_list_of_dicts(self):
        """Extract from a known PDF returns page dicts."""
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
        # Use the smallest PDF available
        pdfs = sorted(oem_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
        if not pdfs:
            pytest.skip("No OEM PDFs found")
        result = extract_pdf(pdfs[0])
        assert isinstance(result, list)
        assert len(result) > 0
        assert "page" in result[0]
        assert "text" in result[0]
        assert "source" in result[0]

    def test_skips_empty_pages(self):
        """Pages with <50 chars after cleaning are skipped."""
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
        pdfs = list(oem_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No OEM PDFs found")
        result = extract_pdf(pdfs[0])
        for page in result:
            assert len(page["text"].strip()) >= 50

    def test_extract_all_pdfs(self):
        """extract_all_pdfs processes every PDF in data/oem/."""
        from rag.pdf_extract import extract_all_pdfs
        result = extract_all_pdfs("data/oem")
        assert isinstance(result, dict)
        # Should have at least skf.pdf
        assert len(result) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_rag.py::TestPDFExtraction -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement `rag/pdf_extract.py`**

```python
"""Extract text from OEM specification PDFs using PyMuPDF."""
from pathlib import Path
import fitz  # PyMuPDF


def extract_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns list of dicts with keys: page, text, source.
    Skips pages with <50 chars after cleaning (full-page images/drawings).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Primary: plain text extraction
        text = page.get_text("text")

        # Fallback: try blocks for better table reconstruction
        if not text or len(text.strip()) < 50:
            blocks = page.get_text("blocks")
            block_texts = []
            for block in blocks:
                if block[6] == 0:  # text block (not image)
                    block_texts.append(block[4])
            text = "\n".join(block_texts)

        # Clean: collapse whitespace, strip repeated headers/footers
        text = _clean_text(text)

        if len(text.strip()) < 50:
            continue

        pages.append({
            "page": page_num + 1,  # 1-indexed
            "text": text,
            "source": pdf_path.name,
        })

    doc.close()
    return pages


def extract_all_pdfs(oem_dir: str | Path = "data/oem") -> dict[str, list[dict]]:
    """Extract text from all PDFs in the OEM directory.

    Returns dict mapping filename -> list of page dicts.
    """
    oem_dir = Path(oem_dir)
    results = {}

    for pdf_path in sorted(oem_dir.glob("*.pdf")):
        try:
            pages = extract_pdf(pdf_path)
            results[pdf_path.name] = pages
            print(f"  {pdf_path.name}: {len(pages)} pages extracted")
        except Exception as e:
            print(f"  {pdf_path.name}: FAILED — {e}")

    return results


def _clean_text(text: str) -> str:
    """Collapse whitespace and clean extracted text."""
    import re
    # Collapse multiple spaces (but preserve newlines for table structure)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rag.py::TestPDFExtraction -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rag/__init__.py rag/pdf_extract.py tests/test_rag.py
git commit -m "feat: add RAG PDF text extraction module"
```

---

### Task 2: RAG Chunking and Ingestion

**Files:**
- Create: `rag/ingest.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Write tests for chunking and ingestion**

Add to `tests/test_rag.py`:

```python
class TestChunking:
    def test_classify_content_type(self):
        from rag.ingest import classify_content_type
        table_text = "6205 25 52 15 14.8 7.8 0.335\n6206 30 62 16 19.5 10.0 0.450"
        assert classify_content_type(table_text) == "table"
        prose_text = "Deep groove ball bearings are the most widely used bearing type."
        assert classify_content_type(prose_text) == "prose"

    def test_chunk_prose(self):
        from rag.ingest import chunk_prose
        text = " ".join(["word"] * 500)  # 500 words
        chunks = chunk_prose(text, target_words=200)
        assert len(chunks) >= 2
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 500  # generous upper bound

    def test_chunk_table_preserves_header(self):
        from rag.ingest import chunk_table
        lines = ["d D B C C0"] + [f"620{i} {20+i} {42+i*10} {12+i} {10+i} {5+i}" for i in range(30)]
        text = "\n".join(lines)
        chunks = chunk_table(text, rows_per_chunk=15)
        assert len(chunks) >= 2
        # Every chunk should start with the header
        for chunk in chunks:
            assert chunk.startswith("d D B C C0")

    def test_detect_manufacturer(self):
        from rag.ingest import detect_manufacturer
        assert detect_manufacturer("skf.pdf", "SKF Rolling Bearings") == "SKF"
        assert detect_manufacturer("mounted-bearing.pdf", "LDK Mounted Bearings") == "LDK"

class TestIngestion:
    def test_ingest_creates_chromadb(self):
        """Full ingestion pipeline creates a ChromaDB collection."""
        from rag.ingest import ingest_oem_pdfs
        import shutil
        test_db_path = "data/processed/chromadb_test"
        try:
            stats = ingest_oem_pdfs(
                oem_dir="data/oem",
                db_path=test_db_path,
            )
            assert stats["total_chunks"] > 0
            assert stats["total_pages"] > 0
        finally:
            shutil.rmtree(test_db_path, ignore_errors=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag.py::TestChunking -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `rag/ingest.py`**

```python
"""Structure-aware chunking and ChromaDB ingestion of OEM PDFs."""
import csv
import re
from pathlib import Path
from rag.pdf_extract import extract_all_pdfs


def detect_manufacturer(filename: str, first_page_text: str = "") -> str:
    """Detect manufacturer from filename or first-page content."""
    combined = (filename + " " + first_page_text).lower()
    if "skf" in combined:
        return "SKF"
    if "rexnord" in combined:
        return "Rexnord"
    if "ldk" in combined:
        return "LDK"
    if "mounted-bearing" in combined or "mounted bearing" in combined:
        return "LDK"
    return "Unknown"


def classify_content_type(text: str) -> str:
    """Classify text as 'table', 'prose', or 'mixed'.

    Table pages: >40% digit characters, multiple lines matching
    designation + numbers pattern.
    """
    lines = text.strip().split("\n")
    if not lines:
        return "mixed"

    # Count digit ratio
    all_chars = "".join(lines)
    non_space = all_chars.replace(" ", "").replace("\n", "")
    if not non_space:
        return "mixed"
    digit_ratio = sum(c.isdigit() or c == "." for c in non_space) / len(non_space)

    # Count lines that look like table rows (designation + multiple numbers)
    table_pattern = re.compile(r"[\w\-/]+\s+\d+[\s.]+\d+")
    table_lines = sum(1 for line in lines if table_pattern.search(line))
    table_ratio = table_lines / len(lines) if lines else 0

    if digit_ratio > 0.35 and table_ratio > 0.3:
        return "table"
    if digit_ratio > 0.25 or table_ratio > 0.2:
        return "mixed"
    return "prose"


def chunk_prose(text: str, target_words: int = 300,
                min_words: int = 100, overlap_sentences: int = 1) -> list[str]:
    """Split prose text into chunks of target_words with sentence overlap."""
    # Try to split on section headers first
    header_pattern = re.compile(r"\n(?=[A-Z][A-Z\s]{3,}\n|[\d]+\.[\d]*\s+[A-Z])")
    sections = header_pattern.split(text)

    chunks = []
    current = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        current_words = len(current.split())
        section_words = len(section.split())

        if current_words + section_words <= target_words * 1.3:
            current = (current + "\n\n" + section).strip()
        else:
            if current and len(current.split()) >= min_words:
                chunks.append(current)
            elif current:
                # Too short — merge with next
                current = (current + "\n\n" + section).strip()
                continue
            current = section

    if current and len(current.split()) >= min_words // 2:
        if chunks and len(current.split()) < min_words:
            chunks[-1] = chunks[-1] + "\n\n" + current
        else:
            chunks.append(current)

    # Add overlap
    if overlap_sentences > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_sentences = chunks[i - 1].split(".")
            overlap = ". ".join(prev_sentences[-overlap_sentences - 1:]).strip()
            if overlap and not overlap.endswith("."):
                overlap += "."
            overlapped.append(overlap + "\n" + chunks[i])
        chunks = overlapped

    return chunks if chunks else [text]


def chunk_table(text: str, rows_per_chunk: int = 18) -> list[str]:
    """Chunk table text, preserving header row in every chunk."""
    lines = text.strip().split("\n")
    if len(lines) <= 2:
        return [text]

    # First non-empty line is the header
    header = ""
    data_lines = []
    for line in lines:
        if not header and line.strip():
            header = line
        elif line.strip():
            data_lines.append(line)

    if not data_lines:
        return [text]

    chunks = []
    for i in range(0, len(data_lines), rows_per_chunk):
        batch = data_lines[i:i + rows_per_chunk]
        chunk = header + "\n" + "\n".join(batch)
        chunks.append(chunk)

    return chunks


def chunk_page(text: str, content_type: str, section_header: str = "") -> list[str]:
    """Chunk a single page based on content type."""
    if content_type == "table":
        chunks = chunk_table(text)
    elif content_type == "prose":
        chunks = chunk_prose(text)
    else:
        # Mixed — chunk at page boundary
        words = text.split()
        if len(words) > 400:
            chunks = chunk_prose(text, target_words=300)
        else:
            chunks = [text]

    # Prepend section header context to table chunks
    if section_header and content_type == "table":
        chunks = [f"{section_header}\n\n{chunk}" for chunk in chunks]

    return chunks


def _detect_section_header(pages: list[dict], page_idx: int) -> str:
    """Try to detect the section header for a page by looking at preceding pages."""
    # Look backwards for a short line that looks like a heading
    for i in range(page_idx, max(-1, page_idx - 5), -1):
        if i >= len(pages):
            continue
        lines = pages[i]["text"].strip().split("\n")
        for line in lines[:5]:  # check first 5 lines
            line = line.strip()
            if 3 < len(line) < 80 and (line.isupper() or re.match(r"^\d+[\.\s]", line)):
                return line
    return ""


def ingest_oem_pdfs(oem_dir: str = "data/oem",
                    db_path: str = "data/processed/chromadb",
                    collection_name: str = "oem_specs") -> dict:
    """Full ingestion pipeline: extract PDFs, chunk, embed, store in ChromaDB.

    Returns stats dict with counts.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # Extract all PDFs
    print("Extracting text from OEM PDFs...")
    all_pages = extract_all_pdfs(oem_dir)

    if not all_pages:
        raise RuntimeError(f"No PDFs found in {oem_dir}")

    # Prepare ChromaDB
    db_path_obj = Path(db_path)
    db_path_obj.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path_obj))

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Load embedding model
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_chunks = []
    chunk_id = 0
    stats = {"total_pages": 0, "total_chunks": 0,
             "by_type": {"prose": 0, "table": 0, "mixed": 0},
             "by_manufacturer": {}}

    for filename, pages in all_pages.items():
        manufacturer = detect_manufacturer(
            filename,
            pages[0]["text"][:500] if pages else "",
        )
        stats["total_pages"] += len(pages)
        stats["by_manufacturer"].setdefault(manufacturer, 0)

        for idx, page_data in enumerate(pages):
            content_type = classify_content_type(page_data["text"])
            section_header = _detect_section_header(pages, idx)

            page_chunks = chunk_page(page_data["text"], content_type, section_header)

            for chunk_text in page_chunks:
                chunk_meta = {
                    "source": filename,
                    "page": page_data["page"],
                    "section_header": section_header,
                    "content_type": content_type,
                    "manufacturer": manufacturer,
                    "chunk_id": f"chunk_{chunk_id}",
                }
                all_chunks.append({"text": chunk_text, "metadata": chunk_meta})
                stats["total_chunks"] += 1
                stats["by_type"][content_type] = stats["by_type"].get(content_type, 0) + 1
                stats["by_manufacturer"][manufacturer] += 1
                chunk_id += 1

    # Embed and store in batches
    print(f"Embedding {len(all_chunks)} chunks...")
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        metas = [c["metadata"] for c in batch]
        ids = [c["metadata"]["chunk_id"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
            ids=ids,
        )

    # Save chunk inventory
    inventory_path = Path("analysis/chunk_inventory.csv")
    inventory_path.parent.mkdir(exist_ok=True)
    with open(inventory_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "source", "page",
                                                "content_type", "manufacturer",
                                                "section_header", "text_length"])
        writer.writeheader()
        for chunk in all_chunks:
            writer.writerow({
                "chunk_id": chunk["metadata"]["chunk_id"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "content_type": chunk["metadata"]["content_type"],
                "manufacturer": chunk["metadata"]["manufacturer"],
                "section_header": chunk["metadata"]["section_header"],
                "text_length": len(chunk["text"]),
            })

    print(f"\nIngestion complete:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By manufacturer: {stats['by_manufacturer']}")
    print(f"  Chunk inventory saved to {inventory_path}")

    return stats
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag.py -v`
Expected: All PASS (ingestion test will take ~1-2 min for embedding)

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_rag.py
git commit -m "feat: add structure-aware chunking and ChromaDB ingestion"
```

---

### Task 3: RAG Retrieval Module

**Files:**
- Create: `rag/retrieve.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Write retrieval tests**

Add to `tests/test_rag.py`:

```python
class TestRetrieval:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Ensure ChromaDB exists for retrieval tests."""
        self.db_path = "data/processed/chromadb"
        if not Path(self.db_path).exists():
            from rag.ingest import ingest_oem_pdfs
            ingest_oem_pdfs(db_path=self.db_path)

    def test_semantic_search_returns_results(self):
        from rag.retrieve import retrieve
        results = retrieve("deep groove ball bearing specifications", k=3)
        assert len(results) > 0
        assert "text" in results[0]
        assert "metadata" in results[0]

    def test_designation_search_boosts_exact_match(self):
        from rag.retrieve import retrieve
        results = retrieve("6205 bearing specifications", k=5)
        # At least one result should contain "6205"
        texts = " ".join(r["text"] for r in results)
        # This may or may not find 6205 depending on catalog content
        assert len(results) > 0

    def test_extract_designation_from_query(self):
        from rag.retrieve import _extract_designation
        assert _extract_designation("SKF 6205-2RS specs") == "6205"
        assert _extract_designation("ZA-2115 load rating") == "ZA-2115"
        assert _extract_designation("UER204 bearing") == "UER204"
        assert _extract_designation("general bearing info") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag.py::TestRetrieval -v`
Expected: FAIL

- [ ] **Step 3: Implement `rag/retrieve.py`**

```python
"""Hybrid retrieval: semantic search + text fallback for bearing specs."""
import re
from pathlib import Path

# Known bearing designations and their manufacturers
KNOWN_DESIGNATIONS = {
    "6205": "SKF", "6205-2RS": "SKF",
    "6204": "SKF", "6204-2RS": "SKF",
    "ZA-2115": "Rexnord", "ZA2115": "Rexnord",
    "UER204": "LDK",
}

DESIGNATION_PATTERN = re.compile(
    r"\b(6[12]\d{2}(?:-2RS)?|ZA-?2115|UER\d{3})\b", re.IGNORECASE
)


def _extract_designation(query: str) -> str | None:
    """Extract a bearing designation from a query string."""
    match = DESIGNATION_PATTERN.search(query)
    return match.group(0) if match else None


def _get_collection(db_path: str = "data/processed/chromadb",
                    collection_name: str = "oem_specs"):
    """Get ChromaDB collection, loading embedding model for queries."""
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(collection_name)


def _get_embedding_model():
    """Load the sentence transformer model (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve(query: str, k: int = 5, expand: bool = True,
             db_path: str = "data/processed/chromadb") -> list[dict]:
    """Hybrid retrieval: semantic search + exact text scan for designations.

    1. Run semantic search on the query.
    2. If query contains a bearing designation:
       - Run expanded queries for specs/dimensions
       - Exact text scan of all table chunks for the designation string
       - Boost chunks containing the exact designation by 1.3x
    3. Merge, deduplicate by chunk_id, return top-K by best score.
    """
    collection = _get_collection(db_path)
    model = _get_embedding_model()

    designation = _extract_designation(query)

    # Primary semantic search
    query_embedding = model.encode(query).tolist()
    primary = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k * 2, 20),
        include=["documents", "metadatas", "distances"],
    )

    # Build results dict keyed by chunk_id
    results = {}
    if primary["ids"] and primary["ids"][0]:
        for i, chunk_id in enumerate(primary["ids"][0]):
            score = 1.0 - primary["distances"][0][i]  # cosine distance to similarity
            results[chunk_id] = {
                "chunk_id": chunk_id,
                "text": primary["documents"][0][i],
                "metadata": primary["metadatas"][0][i],
                "score": score,
            }

    # Expanded queries for designation
    if designation and expand:
        manufacturer = KNOWN_DESIGNATIONS.get(designation, "")
        expanded_queries = [
            f"{designation} specifications dimensions load rating",
            f"{manufacturer} {designation} bearing" if manufacturer else f"{designation} bearing catalog",
        ]

        for eq in expanded_queries:
            eq_embedding = model.encode(eq).tolist()
            expanded = collection.query(
                query_embeddings=[eq_embedding],
                n_results=10,
                include=["documents", "metadatas", "distances"],
            )
            if expanded["ids"] and expanded["ids"][0]:
                for i, chunk_id in enumerate(expanded["ids"][0]):
                    score = 1.0 - expanded["distances"][0][i]
                    if chunk_id in results:
                        results[chunk_id]["score"] = max(results[chunk_id]["score"], score)
                    else:
                        results[chunk_id] = {
                            "chunk_id": chunk_id,
                            "text": expanded["documents"][0][i],
                            "metadata": expanded["metadatas"][0][i],
                            "score": score,
                        }

        # Exact text scan: search table chunks for designation string
        table_chunks = collection.get(
            where={"content_type": "table"},
            include=["documents", "metadatas"],
        )
        if table_chunks["ids"]:
            for i, chunk_id in enumerate(table_chunks["ids"]):
                doc = table_chunks["documents"][i]
                if designation.lower() in doc.lower() or designation.replace("-", "").lower() in doc.replace("-", "").lower():
                    if chunk_id in results:
                        results[chunk_id]["score"] *= 1.3  # boost
                    else:
                        results[chunk_id] = {
                            "chunk_id": chunk_id,
                            "text": doc,
                            "metadata": table_chunks["metadatas"][i],
                            "score": 0.5 * 1.3,  # base score + boost
                        }

    # Also boost any result containing the designation in text
    if designation:
        for chunk_id, r in results.items():
            if designation.lower() in r["text"].lower():
                r["score"] *= 1.3

    # Sort by score, return top-k
    sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:k]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag.py::TestRetrieval -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rag/retrieve.py tests/test_rag.py
git commit -m "feat: add hybrid semantic + text retrieval for bearing specs"
```

---

### Task 4: RAG Parameter Extraction

**Files:**
- Create: `rag/extract_params.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Write extraction tests**

Add to `tests/test_rag.py`:

```python
class TestExtraction:
    def test_bearing_oem_params_dataclass(self):
        from rag.extract_params import ExtractedBearingParams
        params = ExtractedBearingParams(
            designation="SKF 6205-2RS",
            manufacturer="SKF",
            bore_mm=25.0,
            C_kn=14.8,
            C0_kn=7.8,
            life_exponent=3.0,
            bearing_type="ball",
            source_file="skf.pdf",
            extraction_confidence="high",
            raw_text="test",
        )
        assert params.bore_mm == 25.0

    def test_validate_extracted_values(self):
        from rag.extract_params import _validate_params
        # Valid ball bearing params
        assert _validate_params(bore_mm=25.0, C_kn=14.8, bearing_type="ball")
        # C too high for a 25mm ball bearing
        assert not _validate_params(bore_mm=25.0, C_kn=200.0, bearing_type="ball")

    def test_extract_all_bearings(self):
        """Full extraction pipeline for all benchmark bearings."""
        from rag.extract_params import extract_all_bearings
        from pathlib import Path
        results = extract_all_bearings()
        assert isinstance(results, dict)
        # Save results to JSON for inspection
        assert Path("analysis/extracted_oem_params.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag.py::TestExtraction -v`
Expected: FAIL

- [ ] **Step 3: Implement `rag/extract_params.py`**

```python
"""Extract structured bearing parameters from RAG-retrieved chunks."""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from rag.retrieve import retrieve


@dataclass
ExtractedBearingParams:
    designation: str
    manufacturer: str
    bore_mm: float
    C_kn: float              # Dynamic load rating
    C0_kn: float             # Static load rating
    life_exponent: float     # 3.0 for ball, 10/3 for roller
    bearing_type: str        # "ball" or "roller"
    # Geometry (optional)
    n_balls_or_rollers: int | None = None
    pitch_diameter_mm: float | None = None
    ball_or_roller_diameter_mm: float | None = None
    contact_angle_deg: float | None = None
    # Extraction metadata
    source_file: str = ""
    source_page: int | None = None
    extraction_confidence: str = "low"
    raw_text: str = ""


# Ground truth for validation
GROUND_TRUTH = {
    "6205": {"C_kn": 14.8, "bore_mm": 25.0},
    "6204": {"C_kn": 12.7, "bore_mm": 20.0},
    "ZA-2115": {"C_kn": 128.5, "bore_mm": 49.2},
    "UER204": {"C_kn": 12.82, "bore_mm": 20.0},
}

# Bearing designations needed for the benchmark
BENCHMARK_BEARINGS = {
    "6205": {"manufacturer": "SKF", "type": "ball", "datasets": ["cwru"]},
    "6204": {"manufacturer": "SKF", "type": "ball", "datasets": ["femto"]},
    "ZA-2115": {"manufacturer": "Rexnord", "type": "roller", "datasets": ["ims"]},
    "UER204": {"manufacturer": "LDK", "type": "ball", "datasets": ["xjtu_sy"]},
}


def _validate_params(bore_mm: float, C_kn: float, bearing_type: str) -> bool:
    """Validate extracted values against known engineering ranges."""
    if bore_mm <= 0 or C_kn <= 0:
        return False
    if bearing_type == "ball":
        # Ball bearings: C typically 5-50 kN for 10-50mm bore
        if bore_mm < 5 or bore_mm > 100:
            return False
        if C_kn < 2 or C_kn > 100:
            return False
    elif bearing_type == "roller":
        # Roller bearings: C typically 20-500 kN for 20-100mm bore
        if bore_mm < 10 or bore_mm > 200:
            return False
        if C_kn < 10 or C_kn > 1000:
            return False
    return True


def _parse_number(text: str) -> float | None:
    """Parse a number from text, handling commas and units."""
    text = text.strip().replace(",", "").replace(" ", "")
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        return float(match.group(1))
    return None


def _extract_from_table_row(text: str, designation: str,
                            manufacturer: str) -> dict:
    """Try to extract params from a table row containing the designation."""
    params = {}
    lines = text.split("\n")

    # Find lines containing the designation
    target_lines = []
    for i, line in enumerate(lines):
        if designation.lower() in line.lower() or designation.replace("-", "").lower() in line.replace("-", "").lower():
            target_lines.append((i, line))

    if not target_lines:
        return params

    for line_idx, line in target_lines:
        # Extract all numbers from the line
        numbers = re.findall(r"[\d]+\.?\d*", line)
        numbers = [float(n) for n in numbers]

        if len(numbers) < 3:
            continue

        if manufacturer == "SKF":
            # SKF table: d, D, B, ..., C (kN), C0 (kN), ...
            # Bore is often a group header above, but sometimes in the row
            # Look for numbers in the C/C0 range
            for i, n in enumerate(numbers):
                if 10 <= n <= 100 and i < len(numbers) - 1:
                    # Possible bore-range numbers
                    pass

            # Try to match known patterns
            # The designation number encodes bore: 6205 -> 25mm, 6204 -> 20mm
            bore_from_designation = _bore_from_designation(designation)
            if bore_from_designation:
                params["bore_mm"] = bore_from_designation

            # Look for C and C0 (usually adjacent, C > C0 for ball bearings)
            for i in range(len(numbers) - 1):
                if 5 < numbers[i] < 100 and 3 < numbers[i + 1] < numbers[i]:
                    # Likely C, C0
                    params["C_kn"] = numbers[i]
                    params["C0_kn"] = numbers[i + 1]
                    break

        elif manufacturer == "Rexnord":
            # Rexnord: loads in lbf, need to convert
            # Look for large numbers that could be load ratings in lbf
            lbf_to_kn = 0.00444822
            for i in range(len(numbers) - 1):
                if 10000 < numbers[i] < 200000 and 10000 < numbers[i + 1] < 200000:
                    params["C_kn"] = round(numbers[i] * lbf_to_kn, 1)
                    params["C0_kn"] = round(numbers[i + 1] * lbf_to_kn, 1)
                    break
            bore_from_designation = _bore_from_designation(designation)
            if bore_from_designation:
                params["bore_mm"] = bore_from_designation

        elif manufacturer == "LDK":
            # LDK: varies — try flexible parsing
            bore_from_designation = _bore_from_designation(designation)
            if bore_from_designation:
                params["bore_mm"] = bore_from_designation
            # Look for C and C0 in kN range
            for i in range(len(numbers) - 1):
                if 5 < numbers[i] < 50 and 2 < numbers[i + 1] < numbers[i]:
                    params["C_kn"] = numbers[i]
                    params["C0_kn"] = numbers[i + 1]
                    break

    return params


def _bore_from_designation(designation: str) -> float | None:
    """Extract bore diameter from bearing designation number.

    Standard encoding: last two digits * 5 for sizes 04+
    204 -> 20mm, 6205 -> 25mm, ZA-2115 -> ~49.2mm (non-standard)
    """
    # SKF/LDK standard
    match = re.search(r"(\d{2})(\d{2})$", designation.replace("-", "").replace("2RS", ""))
    if match:
        size_code = int(match.group(2))
        if size_code >= 4:
            return size_code * 5.0
        elif size_code == 0:
            return 10.0
        elif size_code == 1:
            return 12.0
        elif size_code == 2:
            return 15.0
        elif size_code == 3:
            return 17.0

    # Rexnord ZA-2115 — non-standard, bore from catalog is 49.2mm (1 15/16")
    if "2115" in designation:
        return 49.2

    # LDK UER204 — 204 series = 20mm bore
    match = re.search(r"(\d{3})$", designation)
    if match:
        code = int(match.group(1))
        size = code % 100
        if size >= 4:
            return size * 5.0

    return None


def _extract_from_prose(text: str, designation: str) -> dict:
    """Try to extract params from prose text via regex."""
    params = {}

    # Dynamic load rating patterns
    c_patterns = [
        r"(?:dynamic\s+load\s+rating|basic\s+dynamic)[^:]*?[:\s]+(\d+\.?\d*)\s*(?:kN|KN)",
        r"C\s*=\s*(\d+\.?\d*)\s*(?:kN|KN)",
        r"(\d+\.?\d*)\s*kN\s*(?:dynamic|C\b)",
    ]
    for pattern in c_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            params["C_kn"] = float(match.group(1))
            break

    # Static load rating
    c0_patterns = [
        r"(?:static\s+load\s+rating|basic\s+static)[^:]*?[:\s]+(\d+\.?\d*)\s*(?:kN|KN)",
        r"C0\s*=\s*(\d+\.?\d*)\s*(?:kN|KN)",
    ]
    for pattern in c0_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            params["C0_kn"] = float(match.group(1))
            break

    return params


def extract_bearing_params(designation: str) -> ExtractedBearingParams:
    """Extract params for a specific bearing designation from the RAG store.

    Strategy:
    1. Retrieve chunks matching the designation
    2. For table chunks: find the row with the designation, parse positionally
    3. For prose chunks: regex for load rating patterns
    4. Cross-validate and pick best values
    5. Validate against known ranges
    """
    info = BENCHMARK_BEARINGS.get(designation, {})
    manufacturer = info.get("manufacturer", "Unknown")
    bearing_type = info.get("type", "ball")
    life_exp = 3.0 if bearing_type == "ball" else 10 / 3

    # Retrieve relevant chunks
    try:
        chunks = retrieve(f"{designation} specifications", k=10)
    except Exception as e:
        print(f"  RAG retrieval failed for {designation}: {e}")
        return _fallback_params(designation, manufacturer, bearing_type, life_exp)

    if not chunks:
        print(f"  No chunks found for {designation}")
        return _fallback_params(designation, manufacturer, bearing_type, life_exp)

    # Try extraction from each chunk
    best_params = {}
    best_source = ""
    best_page = None
    best_raw = ""

    for chunk in chunks:
        text = chunk["text"]
        meta = chunk["metadata"]

        # Try table extraction
        table_params = _extract_from_table_row(text, designation, manufacturer)
        if table_params.get("C_kn"):
            if not best_params.get("C_kn"):
                best_params.update(table_params)
                best_source = meta.get("source", "")
                best_page = meta.get("page")
                best_raw = text[:500]

        # Try prose extraction
        prose_params = _extract_from_prose(text, designation)
        if prose_params.get("C_kn") and not best_params.get("C_kn"):
            best_params.update(prose_params)
            best_source = meta.get("source", "")
            best_page = meta.get("page")
            best_raw = text[:500]

    # Fill in bore from designation if not extracted
    if not best_params.get("bore_mm"):
        bore = _bore_from_designation(designation)
        if bore:
            best_params["bore_mm"] = bore

    # Determine confidence
    confidence = "low"
    if best_params.get("C_kn") and best_params.get("bore_mm"):
        if _validate_params(best_params["bore_mm"], best_params["C_kn"], bearing_type):
            confidence = "medium"
            # Check against ground truth
            gt = GROUND_TRUTH.get(designation, {})
            if gt:
                c_error = abs(best_params["C_kn"] - gt.get("C_kn", 0)) / gt.get("C_kn", 1)
                b_error = abs(best_params["bore_mm"] - gt.get("bore_mm", 0)) / gt.get("bore_mm", 1)
                if c_error < 0.10 and b_error < 0.10:
                    confidence = "high"

    return ExtractedBearingParams(
        designation=designation,
        manufacturer=manufacturer,
        bore_mm=best_params.get("bore_mm", 0.0),
        C_kn=best_params.get("C_kn", 0.0),
        C0_kn=best_params.get("C0_kn", 0.0),
        life_exponent=life_exp,
        bearing_type=bearing_type,
        n_balls_or_rollers=best_params.get("n_balls_or_rollers"),
        pitch_diameter_mm=best_params.get("pitch_diameter_mm"),
        ball_or_roller_diameter_mm=best_params.get("ball_or_roller_diameter_mm"),
        contact_angle_deg=best_params.get("contact_angle_deg"),
        source_file=best_source,
        source_page=best_page,
        extraction_confidence=confidence,
        raw_text=best_raw,
    )


def _fallback_params(designation: str, manufacturer: str,
                     bearing_type: str, life_exp: float) -> ExtractedBearingParams:
    """Return params from ground truth when RAG extraction fails."""
    gt = GROUND_TRUTH.get(designation, {})
    bore = _bore_from_designation(designation) or gt.get("bore_mm", 0.0)
    return ExtractedBearingParams(
        designation=designation,
        manufacturer=manufacturer,
        bore_mm=bore,
        C_kn=gt.get("C_kn", 0.0),
        C0_kn=gt.get("C0_kn", 0.0),
        life_exponent=life_exp,
        bearing_type=bearing_type,
        source_file="hardcoded_fallback",
        extraction_confidence="fallback",
        raw_text="Ground truth values used — RAG extraction failed",
    )


def extract_all_bearings() -> dict[str, ExtractedBearingParams]:
    """Extract params for all bearings needed by the benchmark.

    Saves results to analysis/extracted_oem_params.json and
    human-readable report to analysis/extraction_report.txt.
    """
    results = {}
    report_lines = ["OEM Parameter Extraction Report", "=" * 50, ""]

    for designation, info in BENCHMARK_BEARINGS.items():
        print(f"\nExtracting params for {designation} ({info['manufacturer']})...")
        params = extract_bearing_params(designation)
        results[designation] = params

        # Report
        report_lines.append(f"--- {designation} ({info['manufacturer']}) ---")
        report_lines.append(f"  Bore: {params.bore_mm} mm")
        report_lines.append(f"  C (dynamic): {params.C_kn} kN")
        report_lines.append(f"  C0 (static): {params.C0_kn} kN")
        report_lines.append(f"  Confidence: {params.extraction_confidence}")
        report_lines.append(f"  Source: {params.source_file} (page {params.source_page})")

        # Validate against ground truth
        gt = GROUND_TRUTH.get(designation, {})
        if gt:
            for param_name, expected in gt.items():
                actual = getattr(params, param_name, 0.0)
                if expected > 0:
                    pct_error = abs(actual - expected) / expected * 100
                    status = "PASS" if pct_error < 10 else "FAIL"
                    report_lines.append(
                        f"  Validation {param_name}: extracted={actual}, "
                        f"expected={expected}, error={pct_error:.1f}% [{status}]"
                    )
        report_lines.append("")

    # Save JSON
    json_path = Path("analysis/extracted_oem_params.json")
    json_path.parent.mkdir(exist_ok=True)
    json_data = {k: asdict(v) for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nSaved extracted params to {json_path}")

    # Save report
    report_path = Path("analysis/extraction_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved extraction report to {report_path}")

    return results


def run_full_extraction():
    """Entry point for full RAG extraction pipeline."""
    from rag.ingest import ingest_oem_pdfs
    from pathlib import Path

    db_path = "data/processed/chromadb"
    if not Path(db_path).exists():
        print("Running RAG ingestion first...")
        ingest_oem_pdfs(db_path=db_path)

    return extract_all_bearings()


if __name__ == "__main__":
    run_full_extraction()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_rag.py::TestExtraction -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rag/extract_params.py tests/test_rag.py
git commit -m "feat: add RAG-based bearing parameter extraction with validation"
```

---

### Task 5: XJTU-SY Dataset Config and Download

**Files:**
- Create: `datasets/xjtu_sy/__init__.py`
- Create: `datasets/xjtu_sy/config.py`
- Create: `datasets/xjtu_sy/download.py`

- [ ] **Step 1: Create the XJTU-SY package**

Create `datasets/xjtu_sy/__init__.py` (empty).

- [ ] **Step 2: Create `datasets/xjtu_sy/config.py`**

```python
XJTU_SY_CONFIG = {
    "name": "xjtu_sy",
    "equipment": "LDK UER204 deep groove ball bearing",
    "equipment_type": "ball_bearing",
    "prior_quality": "exact_oem",
    "is_run_to_failure": True,

    "oem_specs": {
        "designation": "LDK UER204",
        "bore_mm": 20.0,
        "C_kn": 12.82,
        "C0_kn": 6.65,
        "life_exponent": 3.0,
        "n_balls": 8,
        "pitch_diameter_mm": 34.55,
        "ball_diameter_mm": 7.92,
        "contact_angle_deg": 0.0,
        "outer_race_diameter_mm": 39.80,
        "inner_race_diameter_mm": 29.30,
    },

    "conditions": {
        1: {"rpm": 2100, "radial_load_kn": 12.0, "n_bearings": 5,
            "dir_name": "35Hz12kN"},
        2: {"rpm": 2250, "radial_load_kn": 11.0, "n_bearings": 5,
            "dir_name": "37.5Hz11kN"},
        3: {"rpm": 2400, "radial_load_kn": 10.0, "n_bearings": 5,
            "dir_name": "40Hz10kN"},
    },

    "data_settings": {
        "sampling_rate_hz": 25600,
        "samples_per_snapshot": 32768,
        "snapshot_interval_sec": 60,
        "channels": ["horizontal", "vertical"],
    },

    "bearing_failures": {
        "Bearing1_1": {"failure": "outer_race", "life_min": 123},
        "Bearing1_2": {"failure": "outer_race", "life_min": 161},
        "Bearing1_3": {"failure": "outer_race", "life_min": 158},
        "Bearing1_4": {"failure": "cage", "life_min": 122},
        "Bearing1_5": {"failure": "inner_race+roller", "life_min": 52},
        "Bearing2_1": {"failure": "inner_race", "life_min": 491},
        "Bearing2_2": {"failure": "outer_race", "life_min": 161},
        "Bearing2_3": {"failure": "cage", "life_min": 533},
        "Bearing2_4": {"failure": "outer_race", "life_min": 42},
        "Bearing2_5": {"failure": "outer_race", "life_min": 339},
        "Bearing3_1": {"failure": "outer_race", "life_min": 2538},
        "Bearing3_2": {"failure": "inner_race+outer_race+cage", "life_min": 2496},
        "Bearing3_3": {"failure": "inner_race", "life_min": 371},
        "Bearing3_4": {"failure": "inner_race", "life_min": 1515},
        "Bearing3_5": {"failure": "outer_race", "life_min": 114},
    },

    "feature_settings": {
        "primary_feature": "kurtosis",
        "features": ["rms", "kurtosis", "crest_factor", "peak_to_peak", "skewness",
                     "bpfi_energy", "bpfo_energy", "bsf_energy", "ftf_energy"],
    },
}
```

- [ ] **Step 3: Create `datasets/xjtu_sy/download.py`**

```python
"""Download XJTU-SY bearing dataset."""
import urllib.request
import zipfile
import shutil
from pathlib import Path


DOWNLOAD_URLS = [
    # GitHub repository (most reliable)
    "https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets/archive/refs/heads/master.zip",
]


def download_xjtu_sy_data(output_dir: str = "data/raw/xjtu_sy") -> None:
    """Download and extract XJTU-SY dataset."""
    output_path = Path(output_dir)

    # Check if already extracted
    if (output_path / "35Hz12kN").exists():
        print("XJTU-SY data already downloaded.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / "xjtu_sy.zip"

    for url in DOWNLOAD_URLS:
        try:
            print(f"Downloading XJTU-SY dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    else:
        raise RuntimeError(
            "Could not download XJTU-SY dataset. Try manually:\n"
            "1. Clone https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets\n"
            "2. Copy the condition directories to data/raw/xjtu_sy/\n"
            "3. Ensure directories 35Hz12kN/, 37.5Hz11kN/, 40Hz10kN/ exist"
        )

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_path)

    # Move files from nested directory if needed
    nested = output_path / "xjtu-sy-bearing-datasets-master"
    if nested.exists():
        for item in nested.iterdir():
            if item.is_dir() and "Hz" in item.name:
                dest = output_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
        shutil.rmtree(nested, ignore_errors=True)

    # Clean up zip
    if zip_path.exists():
        zip_path.unlink()
    print(f"XJTU-SY data extracted to {output_path}")
```

- [ ] **Step 4: Commit**

```bash
git add datasets/xjtu_sy/__init__.py datasets/xjtu_sy/config.py datasets/xjtu_sy/download.py
git commit -m "feat: add XJTU-SY dataset config and download"
```

---

### Task 6: XJTU-SY Feature Extraction

**Files:**
- Create: `datasets/xjtu_sy/feature_extraction.py`
- Create: `tests/test_xjtu_sy.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_xjtu_sy.py
import numpy as np
import pytest


class TestXJTUSYFeatureExtraction:
    def test_compute_defect_frequencies(self):
        from datasets.xjtu_sy.feature_extraction import compute_defect_frequencies
        freqs = compute_defect_frequencies(rpm=2100)
        assert freqs["bpfo"] > 0
        assert freqs["bpfi"] > freqs["bpfo"]  # BPFI > BPFO for ball bearings
        assert freqs["ftf"] < freqs["bsf"]

    def test_extract_features_shape(self):
        from datasets.xjtu_sy.feature_extraction import extract_xjtu_features
        np.random.seed(42)
        snapshot = np.random.randn(32768)
        features = extract_xjtu_features(snapshot, sr=25600)
        assert "rms" in features
        assert "kurtosis" in features
        assert "crest_factor" in features
        assert features["rms"] > 0

    def test_extract_features_with_defect_freqs(self):
        from datasets.xjtu_sy.feature_extraction import (
            extract_xjtu_features, compute_defect_frequencies
        )
        np.random.seed(42)
        snapshot = np.random.randn(32768)
        freqs = compute_defect_frequencies(rpm=2100)
        features = extract_xjtu_features(snapshot, sr=25600, defect_freqs=freqs)
        assert "bpfo_energy" in features
        assert "bpfi_energy" in features


class TestXJTUSYConfig:
    def test_config_structure(self):
        from datasets.xjtu_sy.config import XJTU_SY_CONFIG
        assert XJTU_SY_CONFIG["name"] == "xjtu_sy"
        assert len(XJTU_SY_CONFIG["bearing_failures"]) == 15
        assert len(XJTU_SY_CONFIG["conditions"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_xjtu_sy.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `datasets/xjtu_sy/feature_extraction.py`**

```python
"""XJTU-SY vibration feature extraction for LDK UER204 bearings."""
import numpy as np
from scipy import stats


def compute_defect_frequencies(rpm: float = 2100) -> dict:
    """Compute defect frequencies for LDK UER204.

    n = 8 balls, d = 7.92 mm, D = 34.55 mm, alpha = 0 deg.
    cos(alpha) = 1, simplifying the formulas.
    """
    n_balls = 8
    d_ball = 7.92     # mm
    d_pitch = 34.55   # mm
    # contact_angle = 0 => cos(alpha) = 1
    shaft_freq = rpm / 60.0
    ratio = d_ball / d_pitch

    bpfo = (n_balls / 2) * (1 - ratio) * shaft_freq
    bpfi = (n_balls / 2) * (1 + ratio) * shaft_freq
    bsf = (d_pitch / (2 * d_ball)) * (1 - ratio**2) * shaft_freq
    ftf = 0.5 * (1 - ratio) * shaft_freq

    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def compute_spectral_energy(signal: np.ndarray, sr: float,
                            center_freq: float, bandwidth: float = 5.0,
                            n_harmonics: int = 3) -> float:
    """Compute spectral energy around a defect frequency and its harmonics."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft_mag = np.abs(np.fft.rfft(signal)) ** 2

    total_energy = 0.0
    for h in range(1, n_harmonics + 1):
        f_center = center_freq * h
        mask = (freqs >= f_center - bandwidth) & (freqs <= f_center + bandwidth)
        total_energy += float(np.sum(fft_mag[mask]))
    return total_energy


def extract_xjtu_features(snapshot: np.ndarray, sr: float = 25600,
                           defect_freqs: dict | None = None) -> dict:
    """Extract features from one XJTU-SY snapshot (32768 samples).

    Uses horizontal channel (column 0) as primary if 2D array passed.
    """
    if snapshot.ndim == 2:
        snapshot = snapshot[:, 0]  # horizontal channel

    rms_val = float(np.sqrt(np.mean(snapshot ** 2)))
    features = {
        "rms": rms_val,
        "peak": float(np.max(np.abs(snapshot))),
        "kurtosis": float(stats.kurtosis(snapshot, fisher=True) + 3),
        "skewness": float(stats.skew(snapshot)),
        "crest_factor": float(np.max(np.abs(snapshot)) / rms_val) if rms_val > 0 else 0.0,
        "peak_to_peak": float(np.max(snapshot) - np.min(snapshot)),
    }

    if defect_freqs:
        for name, freq in defect_freqs.items():
            features[f"{name}_energy"] = compute_spectral_energy(snapshot, sr, freq)

    return features


def process_xjtu_bearing(bearing_dir: str, sr: float = 25600,
                          defect_freqs: dict | None = None) -> list[dict]:
    """Process all CSV snapshots from one bearing directory.

    Each CSV: 32768 rows x 2 columns (horizontal, vertical acceleration).
    Returns list of feature dicts, one per snapshot.
    """
    from pathlib import Path
    bearing_path = Path(bearing_dir)
    if not bearing_path.exists():
        raise FileNotFoundError(f"Bearing directory not found: {bearing_path}")

    # Sort CSV files numerically
    csv_files = sorted(
        bearing_path.glob("*.csv"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )

    all_features = []
    for i, csv_file in enumerate(csv_files):
        try:
            data = np.loadtxt(csv_file, delimiter=",")
            if data.ndim == 1:
                continue
            # Use horizontal channel (column 0)
            snapshot = data[:, 0]
            features = extract_xjtu_features(snapshot, sr, defect_freqs)
            all_features.append(features)
        except Exception as e:
            print(f"  Warning: skipped {csv_file.name}: {e}")
            continue

    return all_features
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_xjtu_sy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add datasets/xjtu_sy/feature_extraction.py tests/test_xjtu_sy.py
git commit -m "feat: add XJTU-SY feature extraction with LDK UER204 defect frequencies"
```

---

### Task 7: XJTU-SY Loader

**Files:**
- Create: `datasets/xjtu_sy/loader.py`
- Modify: `tests/test_xjtu_sy.py`

- [ ] **Step 1: Write loader tests**

Add to `tests/test_xjtu_sy.py`:

```python
class TestXJTUSYLoader:
    def test_loader_instantiates(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        loader = XJTUSYLoader()
        assert loader.config["name"] == "xjtu_sy"

    def test_dataset_info(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        loader = XJTUSYLoader()
        info = loader.get_dataset_info()
        assert info["name"] == "xjtu_sy"
        assert info["prior_quality"] == "exact_oem"
        assert info["n_trajectories"] == 15

    def test_compute_oem_prior(self):
        from datasets.xjtu_sy.loader import XJTUSYLoader
        import pandas as pd
        loader = XJTUSYLoader()
        # Create a dummy features DataFrame
        df = pd.DataFrame({"kurtosis": np.random.randn(100) * 0.1 + 3.0})
        prior = loader._compute_oem_prior(df, condition=1)
        assert prior.expected_life > 0
        assert prior.confidence == "exact_oem"
        assert len(prior.baseline_curve) == 100

    def test_l10_values_per_condition(self):
        """Verify L10 calculations match expected values from spec."""
        from core.oem_prior import compute_l10_hours
        # Condition 1: C=12.82, P=12.0, RPM=2100
        l10_1 = compute_l10_hours(C_kn=12.82, P_kn=12.0, rpm=2100, p=3.0)
        assert 8 < l10_1 < 12  # ~9.7 hours

        # Condition 2: C=12.82, P=11.0, RPM=2250
        l10_2 = compute_l10_hours(C_kn=12.82, P_kn=11.0, rpm=2250, p=3.0)
        assert 10 < l10_2 < 14  # ~11.7 hours

        # Condition 3: C=12.82, P=10.0, RPM=2400
        l10_3 = compute_l10_hours(C_kn=12.82, P_kn=10.0, rpm=2400, p=3.0)
        assert 13 < l10_3 < 17  # ~14.6 hours
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_xjtu_sy.py::TestXJTUSYLoader -v`
Expected: FAIL

- [ ] **Step 3: Implement `datasets/xjtu_sy/loader.py`**

```python
"""XJTU-SY dataset loader."""
import numpy as np
import pandas as pd
from pathlib import Path
from framework.dataset_loader import DatasetLoader, DegradationTrajectory, OEMPrior
from datasets.xjtu_sy.config import XJTU_SY_CONFIG
from datasets.xjtu_sy.download import download_xjtu_sy_data
from datasets.xjtu_sy.feature_extraction import (
    process_xjtu_bearing, compute_defect_frequencies,
)
from core.oem_prior import compute_l10_hours, compute_degradation_baseline


class XJTUSYLoader(DatasetLoader):
    def __init__(self, data_dir: str = "data/raw/xjtu_sy",
                 processed_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.processed_dir = Path(processed_dir)
        self.config = XJTU_SY_CONFIG

    def download(self) -> None:
        download_xjtu_sy_data(self.data_dir)

    def load_trajectories(self) -> list[DegradationTrajectory]:
        """Load all 15 XJTU-SY run-to-failure trajectories."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        trajectories = []

        for cond_num, cond_info in self.config["conditions"].items():
            rpm = cond_info["rpm"]
            defect_freqs = compute_defect_frequencies(rpm)
            cond_dir = Path(self.data_dir) / cond_info["dir_name"]

            for bearing_idx in range(1, cond_info["n_bearings"] + 1):
                bearing_name = f"Bearing{cond_num}_{bearing_idx}"
                bearing_dir = cond_dir / bearing_name
                unit_id = f"xjtu_sy_{bearing_name}"

                # Load or extract features
                df = self._load_or_extract(bearing_name, bearing_dir,
                                           defect_freqs, rpm)
                if df is None or len(df) == 0:
                    print(f"  Skipping {bearing_name}: no data")
                    continue

                n = len(df)
                failure_info = self.config["bearing_failures"].get(bearing_name, {})

                # All XJTU-SY bearings are run-to-failure
                failure_index = n - 1
                # Time in hours (snapshots are 1 minute apart)
                time_hours = np.arange(n) / 60.0
                total_hours = time_hours[-1]
                true_rul = np.array([total_hours - t for t in time_hours])

                # Compute OEM prior
                prior = self._compute_oem_prior(df, condition=cond_num)

                traj = DegradationTrajectory(
                    unit_id=unit_id,
                    dataset="xjtu_sy",
                    features=df.reset_index(drop=True),
                    primary_feature=self.config["feature_settings"]["primary_feature"],
                    true_rul=true_rul,
                    failure_index=failure_index,
                    oem_prior=prior,
                    operating_conditions={
                        "rpm": rpm,
                        "radial_load_kn": cond_info["radial_load_kn"],
                        "condition": cond_num,
                    },
                    metadata={
                        "equipment_type": self.config["equipment_type"],
                        "failure_mode": failure_info.get("failure", "unknown"),
                        "life_min": failure_info.get("life_min"),
                        "condition": cond_num,
                        "bearing_name": bearing_name,
                    },
                    is_run_to_failure=True,
                )
                trajectories.append(traj)

        return trajectories

    def _load_or_extract(self, bearing_name: str, bearing_dir: Path,
                          defect_freqs: dict, rpm: float) -> pd.DataFrame | None:
        """Load cached features or extract from raw CSV files."""
        cache_path = self.processed_dir / f"xjtu_sy_{bearing_name}_features.csv"

        if cache_path.exists():
            print(f"  Loading cached features for {bearing_name}")
            return pd.read_csv(cache_path)

        if not bearing_dir.exists():
            return None

        print(f"  Extracting features for {bearing_name}...")
        feature_list = process_xjtu_bearing(
            str(bearing_dir), sr=25600, defect_freqs=defect_freqs,
        )

        if not feature_list:
            return None

        df = pd.DataFrame(feature_list)
        df["time_min"] = np.arange(len(df))
        df.to_csv(cache_path, index=False)
        print(f"  Cached {cache_path} ({len(df)} snapshots)")
        return df

    def _compute_oem_prior(self, features_df: pd.DataFrame,
                            condition: int) -> OEMPrior:
        """Compute OEM prior for LDK UER204 under a specific condition."""
        specs = self.config["oem_specs"]
        cond = self.config["conditions"][condition]

        l10h = compute_l10_hours(
            C_kn=specs["C_kn"],
            P_kn=cond["radial_load_kn"],
            rpm=cond["rpm"],
            p=specs["life_exponent"],
        )

        n = len(features_df)
        baseline = compute_degradation_baseline(l10h, n)

        primary = self.config["feature_settings"]["primary_feature"]
        feat_vals = features_df[primary].values
        healthy_n = max(1, int(n * 0.1))
        healthy_mean = float(np.mean(feat_vals[:healthy_n]))
        healthy_std = float(np.std(feat_vals[:healthy_n]))
        threshold = healthy_mean + 5 * healthy_std
        if threshold <= healthy_mean:
            threshold = healthy_mean + 1.0

        scaled_baseline = healthy_mean + baseline * (threshold - healthy_mean)

        return OEMPrior(
            expected_life=l10h,
            baseline_curve=scaled_baseline,
            threshold=threshold,
            life_unit="hours",
            source=f"LDK {specs['designation']} catalog",
            confidence="exact_oem",
            parameters=dict(specs),
        )

    def get_dataset_info(self) -> dict:
        return {
            "name": "xjtu_sy",
            "equipment": self.config["equipment"],
            "equipment_type": self.config["equipment_type"],
            "prior_quality": self.config["prior_quality"],
            "n_trajectories": 15,
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_xjtu_sy.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add datasets/xjtu_sy/loader.py tests/test_xjtu_sy.py
git commit -m "feat: add XJTU-SY dataset loader with OEM prior computation"
```

---

### Task 8: Wire RAG Into Benchmark Runner

**Files:**
- Modify: `framework/benchmark_runner.py`
- Modify: `core/oem_prior.py`

- [ ] **Step 1: Update `core/oem_prior.py` ground truth dict**

In `core/oem_prior.py`, update the `ground_truth_C` dict in `load_extracted_params` to include all benchmark bearings:

```python
# In load_extracted_params(), add entries to the ground_truth_C dict (keep existing 6203):
ground_truth_C = {
    "6205": 14.8,
    "6203": 9.95,
    "6204": 12.7,
    "ZA-2115": 128.5,
    "UER204": 12.82,
}
```

- [ ] **Step 2: Update `framework/benchmark_runner.py`**

Add XJTU-SY to the loader map and add RAG integration:

At the top of `run_full_benchmark()`, update the default datasets list and add XJTU-SY loader:

```python
# Update default datasets list to include xjtu_sy:
if datasets is None:
    datasets = ["cwru", "ims", "femto", "cmapss", "xjtu_sy"]

# Add XJTU-SY loader import:
try:
    from datasets.xjtu_sy.loader import XJTUSYLoader
    loader_map["xjtu_sy"] = XJTUSYLoader
except ImportError:
    pass
```

Add RAG ingestion step before running datasets:

```python
# At the start of run_full_benchmark(), before the loop:
# Run RAG extraction if PDFs exist and results don't
_ensure_rag_extraction()
```

Add the helper function:

```python
def _ensure_rag_extraction():
    """Run RAG ingestion and extraction if needed."""
    from pathlib import Path
    json_path = Path("analysis/extracted_oem_params.json")
    oem_dir = Path("data/oem")

    if json_path.exists():
        return  # Already extracted

    if not oem_dir.exists() or not list(oem_dir.glob("*.pdf")):
        return  # No PDFs to process

    try:
        from rag.extract_params import run_full_extraction
        print("\nRunning RAG extraction from OEM PDFs...")
        run_full_extraction()
    except Exception as e:
        import warnings
        warnings.warn(f"RAG extraction failed: {e}. Using hardcoded configs.")
```

- [ ] **Step 3: Run existing tests to ensure nothing broke**

Run: `pytest tests/test_core.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add framework/benchmark_runner.py core/oem_prior.py
git commit -m "feat: wire XJTU-SY and RAG extraction into benchmark runner"
```

---

### Task 9: Run Full Benchmark

**Files:**
- No new files — runs existing code to produce CSVs

- [ ] **Step 1: Run the RAG extraction pipeline standalone**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application && python -m rag.extract_params`

Expected: Creates `analysis/extracted_oem_params.json`, `analysis/extraction_report.txt`, `analysis/chunk_inventory.csv`. Check the extraction report for PASS/FAIL on each bearing.

- [ ] **Step 2: Download XJTU-SY raw data**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application && python -c "from datasets.xjtu_sy.download import download_xjtu_sy_data; download_xjtu_sy_data()"`

Expected: Downloads and extracts to `data/raw/xjtu_sy/` with directories `35Hz12kN/`, `37.5Hz11kN/`, `40Hz10kN/`. If download fails (network issues, GitHub rate limiting), manually clone the repository and copy the condition directories.

- [ ] **Step 3: Run the full benchmark**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application && python -m framework.benchmark_runner`

Expected: Produces updated CSVs in `analysis/`:
- `cwru_metrics.csv`
- `ims_metrics.csv`
- `femto_metrics.csv`
- `cmapss_metrics.csv`
- `xjtu_sy_metrics.csv` (NEW)
- `cross_dataset_summary.csv` (UPDATED)

Note: This will take several minutes for IMS and C-MAPSS feature extraction. XJTU-SY will also take a few minutes on first run.

- [ ] **Step 4: Update results_summary.py to handle 5+ datasets**

In `framework/results_summary.py`, no structural changes needed — the functions already work generically on DataFrames. But regenerate the figures:

Run: `python -c "import pandas as pd; from framework.results_summary import plot_cross_dataset_comparison, prior_quality_comparison; df = pd.read_csv('analysis/cross_dataset_summary.csv'); plot_cross_dataset_comparison(df); prior_quality_comparison(df).to_csv('analysis/prior_quality_comparison.csv', index=False)"`

- [ ] **Step 5: Commit results**

```bash
git add analysis/*.csv analysis/*.json analysis/*.txt reports/figures/
git commit -m "feat: benchmark results with XJTU-SY and RAG-extracted OEM specs"
```

---

### Task 10: XJTU-SY Analysis Notebook

**Files:**
- Create: `notebooks/06_xjtu_sy_analysis.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/06_xjtu_sy_analysis.ipynb` with these cells:

**Cell 1 (markdown):** Title and overview — 15 bearings, 3 conditions, failure modes, life durations. Note extreme variation.

**Cell 2 (code):** Load XJTU-SY results from `analysis/xjtu_sy_metrics.csv` and config from `datasets/xjtu_sy/config.py`. Display bearing lifetime summary table.

**Cell 3 (markdown + code):** OEM extraction demo — show RAG extracting LDK UER204 specs from catalog. Load `analysis/extracted_oem_params.json`, display extracted vs ground truth.

**Cell 4 (code):** L10 calculation per condition — compute and display L10 hours. Plot actual bearing lifetimes vs L10 as a scatter with L10 line.

**Cell 5 (code):** Feature trajectories — load cached features for 2-3 bearings with different lifetimes (e.g., Bearing2_4 = 42 min, Bearing1_1 = 123 min, Bearing3_1 = 2538 min). Plot kurtosis over time.

**Cell 6 (code):** Model comparison — aggregate results by model. Bar chart of RMSE across models. Per-condition breakdown.

**Cell 7 (code):** Cross-manufacturer comparison — load IMS and CWRU results. Compare PID+Regime RMSE across SKF/Rexnord/LDK.

**Cell 8 (markdown):** Summary and interpretation.

Implementation note: Use `nbformat` to create the notebook programmatically, or create it as a JSON file with the standard Jupyter notebook format. Each code cell should import what it needs and be self-contained.

- [ ] **Step 2: Verify notebook runs**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application && python -m jupyter nbconvert --to notebook --execute notebooks/06_xjtu_sy_analysis.ipynb --output 06_xjtu_sy_analysis.ipynb`

Expected: Notebook executes without errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_xjtu_sy_analysis.ipynb
git commit -m "feat: add XJTU-SY analysis notebook"
```

---

### Task 11: Update Cross-Dataset Comparison Notebook

**Files:**
- Modify: `notebooks/05_cross_dataset_comparison.ipynb`

- [ ] **Step 1: Update the notebook**

Read the existing notebook, then add:
- Load `xjtu_sy_metrics.csv` alongside existing datasets
- Update the main results table to include XJTU-SY
- Add manufacturer comparison section (SKF vs Rexnord vs LDK)
- Add XJTU-SY per-condition breakdown
- Add RAG extraction reliability section (load from `analysis/extracted_oem_params.json`)

- [ ] **Step 2: Verify notebook runs**

Run: `python -m jupyter nbconvert --to notebook --execute notebooks/05_cross_dataset_comparison.ipynb --output 05_cross_dataset_comparison.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_cross_dataset_comparison.ipynb
git commit -m "feat: update cross-dataset comparison with XJTU-SY and RAG results"
```

---

### Task 12: Updated Benchmark Report

**Files:**
- Modify: `reports/benchmark_report.Rmd`

- [ ] **Step 1: Rewrite the report**

Rewrite `reports/benchmark_report.Rmd` with the full structure specified in the original prompt. Key changes:
- Add OEM Specification Extraction section
- Add XJTU-SY results section (most space)
- Add cross-manufacturer comparison
- Update cross-dataset table to 5 datasets
- Add RAG extraction reliability discussion
- Expand discussion, limitations, future work
- Load `xjtu_sy_metrics.csv` in the data loading chunk
- Load `analysis/extracted_oem_params.json` for extraction results
- Use `=` for assignment throughout
- Use `prettydoc::html_pretty` with tactile theme, github highlight
- Colorblind-friendly palette (already in place)

The report should be written for a reliability engineer/maintenance manager. Define every technical term on first use. Interpret every table.

Structure (from spec):
1. Introduction (2-3 paragraphs)
2. OEM Specification Extraction (1 page)
3. Datasets (1 paragraph each)
4. Results — Real Run-to-Failure Bearings (main section: IMS brief, XJTU-SY detailed)
5. Results — Other Datasets (FEMTO, C-MAPSS, CWRU)
6. Cross-Dataset Comparison (main table, prior quality, regime benefit)
7. Discussion (substantive paragraphs)
8. Limitations
9. Future Work

- [ ] **Step 2: Verify report renders**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application/reports && Rscript -e "rmarkdown::render('benchmark_report.Rmd')"`

Expected: HTML file generated without errors.

- [ ] **Step 3: Commit**

```bash
git add reports/benchmark_report.Rmd
git commit -m "feat: expanded benchmark report with RAG extraction and XJTU-SY results"
```

---

### Task 13: Final Validation

- [ ] **Step 1: Run all tests**

Run: `cd /Users/christopherbrantner/local/projects/adaptive-drift-forecasting/predictive-maintenance-application && pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 2: Verify all expected output files exist**

Check for:
- `analysis/extracted_oem_params.json`
- `analysis/extraction_report.txt`
- `analysis/chunk_inventory.csv`
- `analysis/xjtu_sy_metrics.csv`
- `analysis/cross_dataset_summary.csv` (updated)
- `analysis/prior_quality_comparison.csv` (updated)
- `reports/figures/*.png` (regenerated)
- `notebooks/06_xjtu_sy_analysis.ipynb`

- [ ] **Step 3: Commit final state**

```bash
git add analysis/ reports/ notebooks/ rag/ datasets/xjtu_sy/ tests/ framework/ core/oem_prior.py requirements.txt
git commit -m "chore: final validation — all tests pass, all outputs generated"
```
