from dataclasses import asdict

import pytest

from core.document_manager import DocumentManager, delete_document
from core.errors import LibraryBusyError, PDFError
from core.keyword_retriever import KeywordRetriever, SelectedEvidenceRetriever
from core.library_lock import LibraryLock
from core.text_chunker import TextChunker, keyword_chunker
from core.types import PageText, VisualPageResult
from core.visual_cache import CachedPageVision


def test_text_ingest_search_restart_and_delete_need_no_models(settings, pdf_bytes, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Keyword-only processing must not initialize models or vectors.")

    for target in (
        "core.vector_store.VectorStore.__init__", "core.embeddings.Embeddings",
        "core.vision.VisionAnalyzer.__init__", "core.llm.OllamaClient.__init__",
        "socket.socket.connect",
    ):
        monkeypatch.setattr(target, forbidden)
    data = pdf_bytes("E104 means inlet blockage.", "Recommended inspection interval is 12 months.")
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(data, "manual.pdf")
    assert result.chunk_count == 2
    assert manager.store is None
    assert manager.ingest(data, "renamed.pdf").duplicate
    manager.close()
    manager = DocumentManager(settings, None, keyword_chunker())
    retriever = KeywordRetriever(settings.data_dir)
    hits = retriever.retrieve("What does E104 mean?")
    assert hits[0].chunk.page_number == 1
    assert hits[0].exact_match
    assert hits[0].semantic_score is None
    assert retriever.retrieve("What is the warranty duration?") == []
    assert retriever.retrieve("what is the") == []
    assert retriever.retrieve('"inspection interval"')[0].chunk.page_number == 2
    previous = manager.catalog.latest_ready_generations()
    manager.reindex(result.document_id)
    assert manager.catalog.latest_ready_generations() != previous
    assert retriever.retrieve("E104")
    assert not (settings.data_dir / "chroma").exists()
    delete_document(settings.data_dir, result.document_id)
    assert retriever.retrieve("E104") == []
    assert not manager.catalog.document_path(result.document_id).exists()
    manager.close()


def test_latest_complete_profile_and_filters_ignore_partial_chunks(settings, fake_embedder, pdf_bytes):
    data = pdf_bytes("E104 inlet blockage.")
    hybrid = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    first = hybrid.ingest(data, "manual.pdf")
    manager = DocumentManager(settings, None, keyword_chunker())
    manager.ingest(data, "manual.pdf")
    second = manager.ingest(pdf_bytes("E104 is also documented here."), "update.pdf")
    latest = manager.catalog.latest_ready_generations([first.document_id])
    assert latest == manager.catalog.ready_generations(manager.profile, [first.document_id])
    with manager.catalog.connect() as db:
        db.execute(
            "INSERT INTO generations(id,document_id,profile,state) VALUES (?,?,?,'staging')",
            ("unfinished", first.document_id, "other-profile"),
        )
        db.execute("""
            INSERT INTO chunks SELECT 'unfinished',chunk_id,document_id,filename,
            page_number,page_end,'partialsecret',page_label,source_type,source_model
            FROM chunks WHERE generation=?
        """, (latest[0],))
    retriever = KeywordRetriever(settings.data_dir)
    assert len(retriever.retrieve("E104")) == 2
    assert retriever.retrieve("partialsecret") == []
    assert {hit.chunk.document_id for hit in retriever.retrieve("E104", [second.document_id])} == {
        second.document_id
    }
    assert retriever.retrieve("E104", []) == []
    chosen = retriever.retrieve("E104")[:1]
    evidence = SelectedEvidenceRetriever(chosen)
    assert evidence.retrieve("a different query") == chosen
    assert evidence.retrieve("E104", []) == []
    with LibraryLock(settings.data_dir):
        with pytest.raises(LibraryBusyError):
            retriever.retrieve("E104")
    hybrid.close()
    manager.close()


def test_saved_visual_cache_invalidates_search_and_replaces_duplicate_visual_chunks(settings, pdf_bytes):
    class Analyzer:
        model_identity = "vision@test"
        text = "Visual interpretation: a turquoise arrow."

        def analyze_page(self, *_args):
            return VisualPageResult(self.text)

    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("Selectable text."), "diagram.pdf")
    retriever = KeywordRetriever(settings.data_dir)
    assert retriever.retrieve("turquoise") == []
    analyzer = Analyzer()
    cache = CachedPageVision(manager.catalog, result.document_id, analyzer, "test")
    cache.analyze_page(b"rendered-image", "diagram.pdf", 1)
    hits = retriever.retrieve("turquoise")
    assert len(hits) == 1
    assert hits[0].chunk.source_type == "visual"
    assert hits[0].chunk.source_model == "vision@test"
    generation = manager.catalog.latest_ready_generations()[0]
    with manager.catalog.connect() as db:
        db.execute("""
            INSERT INTO chunks(generation,chunk_id,document_id,filename,page_number,page_end,
            text,page_label,source_type,source_model) VALUES (:generation,:chunk_id,:document_id,
            :filename,:page_number,:page_end,:text,:page_label,:source_type,:source_model)
        """, asdict(hits[0].chunk) | {"generation": generation})
    assert len(KeywordRetriever(settings.data_dir).retrieve("turquoise")) == 1
    analyzer.text = "Visual interpretation: a magenta arrow."
    cache.analyze_page(b"new-rendering", "diagram.pdf", 1)
    assert retriever.retrieve("turquoise") == []
    assert len(retriever.retrieve("magenta")) == 1
    delete_document(settings.data_dir, result.document_id)
    assert retriever.retrieve("magenta") == []
    assert manager.catalog.saved_visual_pages() == []
    manager.close()


def test_keyword_passages_are_byte_bounded_and_keep_unicode():
    text = ("A longer paragraph with words. " * 200) + ("\u754c" * 1800)
    chunks = list(keyword_chunker().chunk_pages([PageText("id", "text.pdf", 2, text)]))
    assert len(chunks) > 3
    assert all(len(chunk.text.encode("utf-8")) <= 2400 for chunk in chunks)
    assert all(chunk.page_number == chunk.page_end == 2 for chunk in chunks)
    assert chunks[-1].text.endswith("\u754c" * 100)
    assert chunks == list(keyword_chunker().chunk_pages([PageText("id", "text.pdf", 2, text)]))


@pytest.mark.parametrize("query", ["", "  ", "x" * 4097])
def test_invalid_search_is_explicit(settings, query):
    with pytest.raises(ValueError):
        KeywordRetriever(settings.data_dir).retrieve(query)


def test_keyword_ingestion_rejects_automatic_vision(settings):
    with pytest.raises(ValueError, match="automatic visual"):
        DocumentManager(settings.model_copy(update={"vision_enabled": True}), None, keyword_chunker())


def test_scanned_document_is_retained_for_explicit_page_analysis(settings, pdf_bytes):
    manager = DocumentManager(settings, None, keyword_chunker())
    with pytest.raises(PDFError, match="explicitly analyze"):
        manager.ingest(pdf_bytes(""), "scan.pdf")
    document = manager.catalog.list_documents()[0]
    assert manager.catalog.document_path(document["id"]).exists()
    assert document["page_count"] == 1
    assert KeywordRetriever(settings.data_dir).retrieve("anything") == []
    manager.close()
