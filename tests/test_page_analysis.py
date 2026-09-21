import pytest

from core.document_manager import DocumentManager, delete_document
from core.errors import IndexError, LibraryBusyError, PDFError
from core.keyword_retriever import KeywordRetriever
from core.library_lock import LibraryLock
from core.page_analysis import analyze_saved_page
from core.text_chunker import keyword_chunker
from core.types import VisualPageResult


class Analyzer:
    model_identity = "test-vision@digest"

    def __init__(self):
        self.calls = []
        self.closed = False

    def analyze_page(self, image, filename, page):
        assert image.startswith(b"\x89PNG")
        self.calls.append((filename, page))
        return VisualPageResult("Visual interpretation: a turquoise diagram.", ("Check labels.",))

    def close(self):
        self.closed = True


def test_only_requested_page_is_analyzed_cached_and_searchable(settings, pdf_bytes):
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("Page one.", "Page two.", "Page three."), "manual.pdf")
    generations = manager.catalog.latest_ready_generations()
    retriever = KeywordRetriever(settings.data_dir)
    assert retriever.retrieve("turquoise") == []
    first = Analyzer()
    analysis = analyze_saved_page(settings, result.document_id, 2, analyzer_factory=lambda: first)
    assert first.calls == [("manual.pdf", 2)]
    assert first.closed
    second = Analyzer()
    assert analyze_saved_page(
        settings, result.document_id, 2, analyzer_factory=lambda: second
    ) == analysis
    assert second.calls == []
    assert second.closed
    assert analysis.warnings == ("Check labels.",)
    assert manager.catalog.latest_ready_generations() == generations
    hits = retriever.retrieve("turquoise")
    assert len(hits) == 1
    assert hits[0].chunk.page_number == 2
    assert hits[0].chunk.source_type == "visual"
    assert not (settings.data_dir / "chroma").exists()
    manager.close()


@pytest.mark.parametrize("page", [0, -1, 3, True, 1.5])
def test_invalid_pages_do_not_start_ai(settings, pdf_bytes, page):
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("One.", "Two."), "manual.pdf")

    def forbidden():
        pytest.fail("Invalid page must be rejected before creating the analyzer.")

    with pytest.raises(PDFError, match="valid physical"):
        analyze_saved_page(settings, result.document_id, page, analyzer_factory=forbidden)
    manager.close()


def test_changed_deleted_and_busy_documents_do_not_start_ai(settings, pdf_bytes):
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("Original."), "manual.pdf")

    def forbidden():
        pytest.fail("Invalid document or busy library must not invoke AI.")

    with LibraryLock(settings.data_dir):
        with pytest.raises(LibraryBusyError):
            analyze_saved_page(settings, result.document_id, 1, analyzer_factory=forbidden)
    manager.catalog.document_path(result.document_id).write_bytes(pdf_bytes("Changed."))
    with pytest.raises(PDFError, match="changed"):
        analyze_saved_page(settings, result.document_id, 1, analyzer_factory=forbidden)
    delete_document(settings.data_dir, result.document_id)
    with pytest.raises(IndexError, match="no longer exists"):
        analyze_saved_page(settings, result.document_id, 1, analyzer_factory=forbidden)
    manager.close()


def test_failed_analysis_is_not_cached_and_closes_client(settings, pdf_bytes):
    class Failure(Analyzer):
        def analyze_page(self, *_args):
            raise ValueError("Vision request failed.")

    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("Text."), "manual.pdf")
    analyzer = Failure()
    with pytest.raises(ValueError, match="Vision request failed"):
        analyze_saved_page(settings, result.document_id, 1, analyzer_factory=lambda: analyzer)
    assert analyzer.closed
    assert manager.catalog.saved_visual_pages() == []
    assert KeywordRetriever(settings.data_dir).retrieve("Text")
    manager.close()
