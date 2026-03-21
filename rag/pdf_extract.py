"""Extract text from PDF documents, preserving page structure.

Uses PyMuPDF (fitz) to handle the SKF catalogue's dense product tables
and the failure analysis guide's mixed prose/image content.

The SKF catalogue tables are the hardest part. PyMuPDF extracts table cells
as individual lines (one value per line), not as tab-delimited rows. This
module reconstructs table rows from the block-level positioning data.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


def _clean_page_text(text: str) -> str:
    """Clean extracted text: collapse whitespace, remove repeated headers/footers."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d{1,3}$", stripped):
            continue
        if stripped in ("SKF", "®", "SKF Group", "www.skf.com"):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_bearing_designation(text: str) -> bool:
    """Check if a string looks like a bearing designation."""
    text = text.strip().lstrip("* ")
    return bool(re.match(r"^\d{3,6}[A-Za-z0-9 \-/]*$", text))


def _is_numeric_value(text: str) -> bool:
    """Check if a string is a numeric value (possibly with European comma decimals)."""
    text = text.strip()
    return bool(re.match(r"^[\d\s,.\-–]+$", text)) and any(c.isdigit() for c in text)


def _reconstruct_table_from_blocks(page: fitz.Page) -> list[str] | None:
    """
    Reconstruct product table rows from positioned text blocks.

    PyMuPDF's get_text("blocks") returns (x0, y0, x1, y1, text, block_idx, type).
    By grouping text spans by their y-coordinate, we can reconstruct rows.
    Returns a list of row strings, or None if the page isn't tabular.
    """
    text_dict = page.get_text("dict")
    if not text_dict.get("blocks"):
        return None

    # Collect all text spans with their positions
    spans = []
    for block in text_dict["blocks"]:
        if block.get("type") != 0:  # text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                bbox = span["bbox"]  # (x0, y0, x1, y1)
                y_center = (bbox[1] + bbox[3]) / 2.0
                x_left = bbox[0]
                spans.append({"text": text, "y": y_center, "x": x_left})

    if not spans:
        return None

    # Group spans into rows by y-coordinate (within ~4pt tolerance)
    spans.sort(key=lambda s: (s["y"], s["x"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    current_y = spans[0]["y"]

    for span in spans:
        if abs(span["y"] - current_y) > 4.0:
            if current_row:
                rows.append(current_row)
            current_row = [span]
            current_y = span["y"]
        else:
            current_row.append(span)

    if current_row:
        rows.append(current_row)

    # Convert rows to strings, tab-separated
    row_strings = []
    designation_count = 0
    numeric_row_count = 0

    for row in rows:
        row.sort(key=lambda s: s["x"])
        texts = [s["text"] for s in row]
        joined = "\t".join(texts)
        row_strings.append(joined)

        # Count rows that look like table data
        full_text = " ".join(texts)
        if _is_bearing_designation(texts[0] if texts else ""):
            designation_count += 1
        if sum(1 for t in texts if _is_numeric_value(t)) >= 3:
            numeric_row_count += 1

    # A page is a product table if it has many designation rows or numeric rows
    if designation_count >= 3 or numeric_row_count >= 8:
        return row_strings

    return None


def is_table_page(page_text: str) -> bool:
    """
    Heuristic to detect product table pages from plain text.

    Checks for bearing designation patterns and high numeric density.
    This is a fallback — prefer _reconstruct_table_from_blocks when
    the page object is available.
    """
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    if not lines:
        return False

    # Count lines that look like bearing designations
    designation_lines = sum(1 for l in lines if _is_bearing_designation(l))

    # Count purely numeric lines (table cell values)
    numeric_lines = sum(1 for l in lines if _is_numeric_value(l))

    # Column header detection
    header_terms = {"Principal", "Designation", "dimensions", "dynamic", "static",
                    "load", "speed", "Mass", "Fatigue"}
    has_header = any(
        sum(1 for t in header_terms if t in l) >= 2
        for l in lines
    )

    # Heuristic: table pages have many designations OR many numeric lines with headers
    return (designation_lines >= 5
            or (has_header and designation_lines >= 2)
            or (has_header and numeric_lines >= 20))


def find_table_header(lines: list[str]) -> str | None:
    """Find column header from reconstructed table rows or raw text lines."""
    header_keywords = {"Principal", "Basic", "Fatigue", "Speed", "Mass",
                       "Designation", "dimensions", "dynamic", "static",
                       "load", "limit", "speed", "Reference", "Limiting"}
    unit_keywords = {"mm", "kN", "r/min", "kg"}

    header_parts = []
    for line in lines[:15]:  # Headers are at the top
        words = set(line.replace("\t", " ").split())
        if len(words & header_keywords) >= 2:
            header_parts.append(line.strip())
        elif len(words & unit_keywords) >= 2:
            header_parts.append(line.strip())
        elif re.search(r"\bd\b.*\bD\b.*\bB\b", line):
            header_parts.append(line.strip())

    if header_parts:
        return " | ".join(header_parts)
    return None


def extract_pdf_text(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF, preserving page structure.

    For catalogue product table pages, attempts block-level reconstruction
    to produce tab-delimited rows. Falls back to plain text extraction
    for prose pages.

    Returns a list of dicts, one per page:
      [{"page": 1, "text": "...", "source": "filename.pdf",
        "is_table": bool, "char_count": int}, ...]
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    source_name = pdf_path.name
    pages = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Try block-based table reconstruction first
        table_rows = _reconstruct_table_from_blocks(page)

        if table_rows is not None:
            text = "\n".join(table_rows)
            pages.append(
                {
                    "page": page_idx + 1,
                    "text": text,
                    "source": source_name,
                    "is_table": True,
                    "char_count": len(text),
                }
            )
        else:
            raw_text = page.get_text("text")
            cleaned = _clean_page_text(raw_text)

            if len(cleaned) < 50:
                continue

            # Check if it's a table page via text heuristic
            table_flag = is_table_page(cleaned)

            pages.append(
                {
                    "page": page_idx + 1,
                    "text": cleaned,
                    "source": source_name,
                    "is_table": table_flag,
                    "char_count": len(cleaned),
                }
            )

    doc.close()
    return pages


def extract_all_pdfs(oem_dir: str | Path = "data/oem") -> list[dict]:
    """Extract text from all PDFs in the OEM directory."""
    oem_dir = Path(oem_dir)
    all_pages = []

    for pdf_file in sorted(oem_dir.glob("*.pdf")):
        print(f"Extracting {pdf_file.name}...")
        pages = extract_pdf_text(pdf_file)
        table_count = sum(1 for p in pages if p["is_table"])
        all_pages.extend(pages)
        print(f"  {len(pages)} pages ({table_count} table, {len(pages)-table_count} prose)")

    return all_pages
