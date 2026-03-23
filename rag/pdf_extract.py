"""PDF text extraction for the RAG pipeline.

Extracts text from OEM bearing specification PDFs using PyMuPDF (fitz).
This is the first stage of the RAG pipeline:
    pdf_extract -> ingest -> retrieve -> extract_params
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip repeated headers/footers.

    Parameters
    ----------
    text : raw extracted text from a PDF page

    Returns
    -------
    Cleaned text with collapsed whitespace and trimmed edges.
    """
    # Collapse runs of whitespace (spaces, tabs) into single spaces
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines into double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Final strip
    text = text.strip()
    return text


def _strip_repeated_headers_footers(pages_text: list[str]) -> list[str]:
    """Remove lines that appear identically on most pages (headers/footers).

    Checks the first and last 2 lines of each page. If a line appears on
    more than half the pages, it is considered a repeated header or footer
    and is removed from all pages.

    Parameters
    ----------
    pages_text : list of raw text strings, one per page

    Returns
    -------
    List of text strings with repeated header/footer lines removed.
    """
    if len(pages_text) < 4:
        return pages_text

    # Collect candidate header/footer lines
    from collections import Counter
    candidates: Counter[str] = Counter()

    for text in pages_text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # Check first 2 and last 2 lines
        edge_lines = set()
        for line in lines[:2]:
            edge_lines.add(line)
        for line in lines[-2:]:
            edge_lines.add(line)
        for line in edge_lines:
            candidates[line] += 1

    threshold = len(pages_text) * 0.5
    repeated = {line for line, count in candidates.items()
                if count >= threshold and len(line) < 200}

    if not repeated:
        return pages_text

    cleaned = []
    for text in pages_text:
        lines = text.splitlines()
        filtered = [l for l in lines if l.strip() not in repeated]
        cleaned.append("\n".join(filtered))
    return cleaned


def _ocr_fallback(pdf_path: Path, n_pages: int) -> list[str]:
    """OCR fallback for image-only PDFs using pytesseract.

    Renders each page to a high-res pixmap and runs Tesseract OCR.
    Returns list of OCR'd text strings, one per page.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        print(f"  OCR skipped for {pdf_path.name}: pytesseract or Pillow not installed")
        return [""] * n_pages

    print(f"  Running OCR on {pdf_path.name} ({n_pages} pages)...")
    doc = fitz.open(str(pdf_path))
    ocr_pages: list[str] = []

    for page_num in range(len(doc)):
        try:
            # Render page at 300 DPI for OCR quality
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = doc[page_num].get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Pre-process for better OCR: convert to grayscale, increase contrast
            img = img.convert("L")  # grayscale
            try:
                from PIL import ImageFilter, ImageOps
                img = ImageOps.autocontrast(img, cutoff=1)
                img = img.filter(ImageFilter.SHARPEN)
            except ImportError:
                pass

            text = pytesseract.image_to_string(
                img,
                config="--psm 6",  # assume uniform block of text
            )
            ocr_pages.append(text)
            if (page_num + 1) % 10 == 0:
                print(f"    OCR page {page_num + 1}/{len(doc)}")
        except Exception as e:
            print(f"    OCR failed on page {page_num + 1}: {e}")
            ocr_pages.append("")

    doc.close()
    print(f"  OCR complete: {sum(1 for t in ocr_pages if len(t.strip()) >= 50)} pages with text")
    return ocr_pages


def extract_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract text from a single PDF file.

    For each page, extracts text using PyMuPDF's text mode. Pages with
    fewer than 50 characters after cleaning are skipped (likely full-page
    images or drawings).

    Parameters
    ----------
    pdf_path : path to the PDF file

    Returns
    -------
    List of dicts with keys: "page" (1-indexed), "text", "source" (filename).
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    doc = fitz.open(str(pdf_path))
    raw_pages: list[str] = []

    for page in doc:
        text = page.get_text("text")

        # Also try block extraction for better table reconstruction
        blocks = page.get_text("blocks")
        if blocks:
            block_text = "\n".join(
                b[4] for b in blocks if b[6] == 0  # type 0 = text blocks
            )
            # Use block text if it captures more content
            if len(block_text.strip()) > len(text.strip()):
                text = block_text

        raw_pages.append(text)

    # If no pages yielded text, try OCR as fallback
    text_pages_count = sum(1 for t in raw_pages if len(t.strip()) >= 50)
    if text_pages_count == 0 and len(raw_pages) > 0:
        raw_pages = _ocr_fallback(pdf_path, len(raw_pages))

    doc.close()

    # Strip repeated headers/footers across pages
    cleaned_pages = _strip_repeated_headers_footers(raw_pages)

    results: list[dict] = []
    for i, text in enumerate(cleaned_pages):
        cleaned = _clean_text(text)
        # Skip pages with < 50 chars (full-page images/drawings)
        if len(cleaned) < 50:
            continue
        results.append({
            "page": i + 1,  # 1-indexed
            "text": cleaned,
            "source": filename,
        })

    return results


def extract_all_pdfs(oem_dir: str | Path) -> dict[str, list[dict]]:
    """Extract text from all PDFs in a directory.

    Parameters
    ----------
    oem_dir : directory containing OEM PDF files

    Returns
    -------
    Dict mapping filename to list of page dicts (same format as extract_pdf).
    """
    oem_dir = Path(oem_dir)
    results: dict[str, list[dict]] = {}

    for pdf_path in sorted(oem_dir.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        if pages:
            results[pdf_path.name] = pages

    return results
