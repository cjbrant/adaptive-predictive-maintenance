"""Tests for the RAG ingestion and retrieval pipeline on real PDFs."""

import pytest
from pathlib import Path

from rag.pdf_extract import extract_pdf_text, is_table_page, _reconstruct_table_from_blocks
from rag.ingest import (
    ingest_oem_documents,
    COLLECTION_NAME,
    _is_table_data_line,
    chunk_pages,
)
from rag.extract_params import (
    _extract_floats,
    _parse_table_row_for_designation,
    extract_bearing_params,
    extract_vibration_thresholds,
)


@pytest.fixture(scope="module")
def oem_dir():
    path = Path("data/oem")
    if not path.exists() or not list(path.glob("*.pdf")):
        pytest.skip("OEM PDFs not found in data/oem/")
    return path


@pytest.fixture(scope="module")
def vectorstore(oem_dir):
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        collection, n_chunks, inventory = ingest_oem_documents(oem_dir, db_dir=tmpdir)
        from sentence_transformers import SentenceTransformer
        from rag.ingest import EMBEDDING_MODEL

        model = SentenceTransformer(EMBEDDING_MODEL)
        yield collection, model, n_chunks, inventory


class TestPDFExtraction:
    def test_catalogue_pages(self, oem_dir):
        pages = extract_pdf_text(oem_dir / "skf_general_catalogue.pdf")
        assert len(pages) > 100  # Should extract many pages
        # Should have some table pages
        table_pages = [p for p in pages if p["is_table"]]
        assert len(table_pages) > 20

    def test_table_contains_6205(self, oem_dir):
        """The 6205 bearing data should be in a table page."""
        pages = extract_pdf_text(oem_dir / "skf_general_catalogue.pdf")
        table_pages = [p for p in pages if p["is_table"]]
        found = any("6205" in p["text"] for p in table_pages)
        assert found, "6205 not found in any table page"

    def test_failure_pdf_prose(self, oem_dir):
        pages = extract_pdf_text(oem_dir / "sfk_bearing_damage_and_failure_analysis.pdf")
        assert len(pages) > 50
        # Should contain failure-related content
        all_text = " ".join(p["text"] for p in pages)
        assert "spalling" in all_text.lower()


class TestTableDataLine:
    def test_tab_delimited_numbers(self):
        assert _is_table_data_line("52\t15\t14,8\t7,8\t0,335\t28 000\t18 000\t0,13\t*\t6205")

    def test_prose_not_table(self):
        assert not _is_table_data_line("The bearing is damaged as soon as spalling occurs.")

    def test_header_not_table(self):
        assert not _is_table_data_line("Single row deep groove ball bearings")


class TestFloatExtraction:
    def test_tab_delimited(self):
        nums = _extract_floats("52\t15\t14,8\t7,8\t0,335")
        assert nums == pytest.approx([52.0, 15.0, 14.8, 7.8, 0.335])

    def test_european_decimals(self):
        nums = _extract_floats("9,95\t4,75\t0,2")
        assert nums == pytest.approx([9.95, 4.75, 0.2])

    def test_space_thousands(self):
        nums = _extract_floats("28 000\t18 000")
        assert nums == pytest.approx([28000.0, 18000.0])

    def test_prose_text(self):
        nums = _extract_floats("The dynamic load rating (C): 14.8 kN")
        assert 14.8 in nums


class TestTableRowParsing:
    def test_6205_row(self):
        text = ("d\t25\tmm\n"
                "52\t15\t14,8\t7,8\t0,335\t28 000\t18 000\t0,13\t*\t6205")
        result = _parse_table_row_for_designation(text, "6205")
        assert result is not None
        assert result["bore_mm"] == pytest.approx(25.0)
        assert result["outside_diameter_mm"] == pytest.approx(52.0)
        assert result["width_mm"] == pytest.approx(15.0)
        assert result["dynamic_load_rating_kn"] == pytest.approx(14.8)
        assert result["static_load_rating_kn"] == pytest.approx(7.8)

    def test_6203_row(self):
        text = ("d\t12\tmm\n"
                "17\t40\t12\t9,95\t4,75\t0,2\t38 000\t24 000\t0,065\t*\t6203")
        result = _parse_table_row_for_designation(text, "6203")
        assert result is not None
        assert result["bore_mm"] == pytest.approx(17.0)
        assert result["dynamic_load_rating_kn"] == pytest.approx(9.95)

    def test_missing_designation(self):
        text = "52\t15\t14,8\t7,8\t*\t6205"
        result = _parse_table_row_for_designation(text, "9999")
        assert result is None


class TestIngestion:
    def test_chunk_count(self, vectorstore):
        collection, model, n_chunks, inventory = vectorstore
        assert n_chunks > 200  # Real PDFs produce many chunks
        assert collection.count() == n_chunks

    def test_content_types(self, vectorstore):
        _, _, _, inventory = vectorstore
        types = set(c["content_type"] for c in inventory)
        assert "prose" in types
        assert "table" in types

    def test_all_sources_present(self, vectorstore):
        _, _, _, inventory = vectorstore
        sources = set(c["source"] for c in inventory)
        assert any("general_catalogue" in s for s in sources)
        assert any("failure" in s.lower() for s in sources)


class TestExtraction:
    def test_6205_params(self, vectorstore):
        collection, model, _, _ = vectorstore
        params = extract_bearing_params(collection, model, "6205", verbose=False)

        assert params.dynamic_load_rating_kn == pytest.approx(14.8, abs=0.1)
        assert params.static_load_rating_kn == pytest.approx(7.8, abs=0.1)
        assert params.bore_mm == pytest.approx(25.0, abs=0.5)
        assert params.outside_diameter_mm == pytest.approx(52.0, abs=0.5)
        assert params.life_exponent == pytest.approx(3.0)

    def test_6203_params(self, vectorstore):
        collection, model, _, _ = vectorstore
        params = extract_bearing_params(collection, model, "6203", verbose=False)

        assert params.dynamic_load_rating_kn == pytest.approx(9.95, abs=0.1)
        assert params.bore_mm == pytest.approx(17.0, abs=0.5)

    def test_vibration_thresholds(self, vectorstore):
        collection, model, _, _ = vectorstore
        thresholds = extract_vibration_thresholds(collection, model, verbose=False)

        # These are ISO 10816 standard values (may or may not be found in PDFs)
        assert thresholds.good_upper == pytest.approx(0.71, abs=0.1)
        assert thresholds.alarm_upper == pytest.approx(4.5, abs=0.5)
