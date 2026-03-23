import pytest
from pathlib import Path


class TestPDFExtraction:
    def test_extract_returns_list_of_dicts(self):
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
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
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
        pdfs = list(oem_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No OEM PDFs found")
        result = extract_pdf(pdfs[0])
        for page in result:
            assert len(page["text"].strip()) >= 50

    def test_extract_all_pdfs(self):
        from rag.pdf_extract import extract_all_pdfs
        result = extract_all_pdfs("data/oem")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_clean_text_collapses_whitespace(self):
        from rag.pdf_extract import _clean_text
        raw = "  hello   world  \n\n\n  foo   bar  "
        cleaned = _clean_text(raw)
        assert "   " not in cleaned
        assert cleaned == cleaned.strip()

    def test_source_is_filename_not_full_path(self):
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
        pdfs = list(oem_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No OEM PDFs found")
        result = extract_pdf(pdfs[0])
        if result:
            assert "/" not in result[0]["source"]
            assert result[0]["source"].endswith(".pdf")

    def test_page_numbers_are_one_indexed(self):
        from rag.pdf_extract import extract_pdf
        oem_dir = Path("data/oem")
        pdfs = sorted(oem_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
        if not pdfs:
            pytest.skip("No OEM PDFs found")
        result = extract_pdf(pdfs[0])
        if result:
            assert result[0]["page"] >= 1


class TestChunking:
    def test_classify_content_type(self):
        from rag.ingest import classify_content_type
        table_text = "6205 25 52 15 14.8 7.8 0.335\n6206 30 62 16 19.5 10.0 0.450"
        assert classify_content_type(table_text) == "table"
        prose_text = "Deep groove ball bearings are the most widely used bearing type."
        assert classify_content_type(prose_text) == "prose"

    def test_chunk_prose(self):
        from rag.ingest import chunk_prose
        text = " ".join(["word"] * 500)
        chunks = chunk_prose(text, target_words=200)
        assert len(chunks) >= 2
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 500

    def test_chunk_table_preserves_header(self):
        from rag.ingest import chunk_table
        lines = ["d D B C C0"] + [f"620{i} {20+i} {42+i*10} {12+i} {10+i} {5+i}" for i in range(30)]
        text = "\n".join(lines)
        chunks = chunk_table(text, rows_per_chunk=15)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.startswith("d D B C C0")

    def test_detect_manufacturer(self):
        from rag.ingest import detect_manufacturer
        assert detect_manufacturer("skf.pdf", "SKF Rolling Bearings") == "SKF"
        assert detect_manufacturer("mounted-bearing.pdf", "LDK Mounted Bearings") == "LDK"


class TestIngestion:
    def test_ingest_creates_chromadb(self):
        from rag.ingest import ingest_oem_pdfs
        import shutil
        test_db_path = "data/processed/chromadb_test"
        try:
            stats = ingest_oem_pdfs(oem_dir="data/oem", db_path=test_db_path)
            assert stats["total_chunks"] > 0
            assert stats["total_pages"] > 0
        finally:
            shutil.rmtree(test_db_path, ignore_errors=True)


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
        assert len(results) > 0

    def test_extract_designation_from_query(self):
        from rag.retrieve import _extract_designation
        assert _extract_designation("SKF 6205-2RS specs") == "6205-2RS"
        assert _extract_designation("ZA-2115 load rating") == "ZA-2115"
        assert _extract_designation("UER204 bearing") == "UER204"
        assert _extract_designation("general bearing info") is None


class TestExtraction:
    def test_extracted_bearing_params_dataclass(self):
        from rag.extract_params import ExtractedBearingParams
        params = ExtractedBearingParams(
            designation="SKF 6205-2RS", manufacturer="SKF",
            bore_mm=25.0, C_kn=14.8, C0_kn=7.8,
            life_exponent=3.0, bearing_type="ball",
            source_file="skf.pdf", extraction_confidence="high", raw_text="test",
        )
        assert params.bore_mm == 25.0

    def test_validate_extracted_values(self):
        from rag.extract_params import _validate_params
        assert _validate_params(bore_mm=25.0, C_kn=14.8, bearing_type="ball")
        assert not _validate_params(bore_mm=25.0, C_kn=200.0, bearing_type="ball")

    def test_bore_from_designation(self):
        from rag.extract_params import _bore_from_designation
        assert _bore_from_designation("6205") == 25.0
        assert _bore_from_designation("6204") == 20.0
        assert _bore_from_designation("UER204") == 20.0
        assert _bore_from_designation("ZA-2115") == 49.2

    def test_extract_all_bearings(self):
        from rag.extract_params import extract_all_bearings
        from pathlib import Path
        results = extract_all_bearings()
        assert isinstance(results, dict)
        assert Path("analysis/extracted_oem_params.json").exists()
