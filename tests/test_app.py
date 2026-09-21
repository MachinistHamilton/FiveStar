import json

import pymupdf
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from config.settings import PROJECT_ROOT
from core.document_manager import DocumentManager
from core.errors import GenerationError
from core.text_chunker import TextChunker
from core.types import VisualPageResult


@pytest.fixture(autouse=True)
def deterministic_processing_mode(monkeypatch):
    monkeypatch.setenv("VISION_ENABLED", "false")


def test_empty_app_starts_without_loading_models(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_CACHE", str(tmp_path / "missing-models"))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("Ask your documents" == title.value for title in app.title)
    assert any("Choose PDFs" in info.value for info in app.info)
    assert not list((tmp_path / "missing-models").glob("*"))
    clear = next(button for button in app.button if button.label == "Clear conversation")
    clear.click().run()
    assert not app.exception


def test_ui_surfaces_ollama_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    def unavailable(_self):
        raise GenerationError("Ollama is not running. Start ollama serve.")

    monkeypatch.setattr("core.llm.OllamaClient.check_available", unavailable)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    button = next(button for button in app.button if button.label == "Check Ollama connection")
    button.click().run()
    assert not app.exception
    assert any("ollama serve" in error.value for error in app.error)


def test_chat_sources_viewer_and_export(settings, fake_embedder, pdf_bytes, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("RERANK_ENABLED", "false")
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    manager.ingest(
        pdf_bytes("The recommended inspection interval under normal conditions is 12 months."),
        "manual.pdf",
    )
    monkeypatch.setattr("core.embeddings.Embeddings", lambda _settings: fake_embedder)
    monkeypatch.setattr("core.llm.OllamaClient.check_available", lambda _self: None)

    calls = []

    def respond(_self, messages, **_kwargs):
        calls.append(messages)
        if "Check each claim" in messages[0]["content"]:
            return json.dumps({"support": [{"claim_id": 1, "supported": True}]})
        return json.dumps({
            "sufficient": True,
            "claims": [{"text": "The inspection interval is 12 months.", "sources": [1]}],
        })

    monkeypatch.setattr("core.llm.OllamaClient.chat", respond)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    app.chat_input[0].set_value("What is the normal inspection interval?").run()
    assert not app.exception
    assert calls == []
    assert any("12 months" in text.value for text in app.text)
    next(button for button in app.button if button.label == "Explain these results (AI)").click().run()
    assert len(calls) == 2
    assert not app.exception
    assert any("12 months" in text.value for text in app.markdown)
    assert any("physical PDF page 1" in text.value for text in app.markdown)
    source_button = next(button for button in app.button if button.label == "Inspect PDF page")
    source_button.click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert any(number.label == "Physical PDF page" for number in app.number_input)
    assert any("Text stored for search" == title.value for title in app.subheader)
    assert any("12 months" in text.value for text in app.text)
    assert app.get("image")
    assert len(app.get("download_button")) == 2
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    manager.store.close()


def test_extraction_coverage_preview_and_notes_without_models(
    settings, fake_embedder, pdf_bytes, monkeypatch
):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(
        pdf_bytes("This selectable text was indexed even though the page has an image.", ""),
        "mixed.pdf",
    )
    document = manager.catalog.get_document(result.document_id)
    warnings = document["warnings"] + [
        "Page 1: Contains images; text extraction and OCR do not describe figures, charts or other visual content."
    ]
    with manager.catalog.connect() as db:
        db.execute(
            "UPDATE documents SET warnings=? WHERE id=?",
            (json.dumps(warnings), result.document_id),
        )

    def no_models(_settings):
        raise AssertionError("Inspecting already indexed text must not require models.")

    monkeypatch.setattr("core.embeddings.Embeddings", no_models)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("Searchable text: 1 of 2 pages | 1 indexed passages" in item.value for item in app.success)
    assert any("does not mean text extraction failed" in item.value for item in app.info)
    assert any("little selectable text" in item.value for item in app.warning)
    assert any("Page-by-page extraction notes" in item.label for item in app.expander)
    next(button for button in app.button if button.label == "Inspect extracted text").click().run()
    assert not app.exception
    assert not app.error
    assert any("This selectable text was indexed" in item.value for item in app.text)
    next(number for number in app.number_input if number.label == "Physical PDF page").set_value(2).run()
    assert not app.exception
    assert any("No indexed text is stored for this page" in item.value for item in app.warning)
    manager.store.close()


def test_visual_preview_remains_labeled_without_running_models(settings, fake_embedder, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("VISION_ENABLED", "true")
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 200, 200))
        data = pdf.tobytes()

    class Analyzer:
        model_identity = "test-vision@digest"

        def analyze_page(self, *_args):
            return VisualPageResult("Visual interpretation: a rectangular diagram is visible.")

        def close(self):
            pass

    manager = DocumentManager(
        settings.model_copy(update={"vision_enabled": True}),
        fake_embedder, TextChunker(fake_embedder), vision_factory=Analyzer,
    )
    manager.ingest(data, "diagram.pdf")
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("Visual interpretation: 1 pages" in item.value for item in app.caption)
    next(button for button in app.button if button.label == "Inspect extracted text").click().run()
    assert not app.exception
    assert any("Visual interpretation by test-vision@digest" in item.value for item in app.warning)
    assert any("rectangular diagram" in item.value for item in app.text)
    assert app.get("image")
    manager.store.close()


def test_paused_visual_index_does_not_block_keyword_search(settings, fake_embedder, pdf_bytes, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(pdf_bytes("Existing searchable text."), "pending.pdf")
    with manager.catalog.connect() as db:
        db.execute(
            "INSERT INTO generations(id,document_id,profile,state) VALUES (?,?,?,'staging')",
            ("pending-visual", result.document_id, "visual-profile"),
        )
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert not app.chat_input[0].disabled
    assert any("earlier indexing job did not finish" in info.value for info in app.info)
    app.chat_input[0].set_value("searchable").run()
    assert not app.exception
    assert any("Existing searchable text" in item.value for item in app.text)
    with manager.catalog.connect() as db:
        db.execute("DELETE FROM generations WHERE id='pending-visual'")
    next(button for button in app.button if button.label == "Refresh indexing status").click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    manager.store.close()


def test_keyword_search_never_loads_models_and_no_match_never_invokes_ai(
    settings, fake_embedder, pdf_bytes, monkeypatch
):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    manager.ingest(pdf_bytes("E104 indicates an inlet blockage."), "manual.pdf")

    def unexpected_model(*_args, **_kwargs):
        pytest.fail("Ordinary keyword search must not load or call any AI model.")

    for target in (
        "core.embeddings.Embeddings", "core.llm.OllamaClient.__init__",
        "core.vision.VisionAnalyzer.__init__", "core.reranker.load_reranker",
        "core.vector_store.VectorStore.__init__",
    ):
        monkeypatch.setattr(target, unexpected_model)
    monkeypatch.setattr("httpx.Client.send", unexpected_model)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    app.chat_input[0].set_value("What does E104 mean?").run()
    assert not app.exception
    assert any("E104 indicates an inlet blockage" in item.value for item in app.text)
    assert any(button.label == "Explain these results (AI)" for button in app.button)
    assert any(button.label == "Create from these documents (AI)" for button in app.button)
    next(button for button in app.button if button.label == "View page").click().run()
    assert not app.exception
    assert app.get("image")
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    app.chat_input[0].set_value("What is the warranty duration?").run()
    assert not app.exception
    assert any("No matching keywords" in item.value for item in app.info)
    assert not any(button.label == "Explain these results (AI)" for button in app.button)
    assert not any(button.label == "Create from these documents (AI)" for button in app.button)
    manager.close()


def test_reindex_is_text_only_even_with_bulk_vision_configured(
    settings, pdf_bytes, monkeypatch
):
    from core.text_chunker import keyword_chunker

    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("VISION_ENABLED", "true")
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("Searchable passage E104."), "lightweight.pdf")

    def no_models(*_args, **_kwargs):
        pytest.fail("UI reindex must extract text without neural models.")

    monkeypatch.setattr("core.embeddings.Embeddings", no_models)
    monkeypatch.setattr("core.vision.VisionAnalyzer.__init__", no_models)
    monkeypatch.setattr("core.vector_store.VectorStore.__init__", no_models)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    next(button for button in app.button if button.label == "Reindex").click().run()
    assert not app.exception
    assert not app.error
    assert manager.catalog.index_summary(result.document_id).chunk_count == 1
    assert not (settings.data_dir / "chroma").exists()
    manager.close()


def test_image_analysis_requires_explicit_click_and_targets_selected_page(
    settings, pdf_bytes, monkeypatch
):
    from core.text_chunker import keyword_chunker

    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("First page.", "Second page."), "pages.pdf")
    calls = []

    def analyze(_settings, identifier, page):
        calls.append((identifier, page))
        return VisualPageResult("A diagram.")

    monkeypatch.setattr("core.page_analysis.analyze_saved_page", analyze)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    next(button for button in app.button if button.label == "View original").click().run()
    next(number for number in app.number_input if number.label == "Physical PDF page").set_value(2).run()
    assert calls == []
    next(button for button in app.button if button.label == "Analyze this page's images (AI)").click().run()
    assert not app.exception
    assert calls == [(result.document_id, 2)]
    assert any("Page analysis saved" in item.value for item in app.success)
    manager.close()


def test_deleted_search_results_cannot_be_sent_to_ai(settings, pdf_bytes, monkeypatch):
    from core.document_manager import delete_document
    from core.text_chunker import keyword_chunker

    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(pdf_bytes("E104 means inlet blockage."), "removed.pdf")
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    app.chat_input[0].set_value("E104").run()
    delete_document(settings.data_dir, result.document_id)

    def no_ai(*_args, **_kwargs):
        pytest.fail("Outdated evidence must be rejected before any AI call.")

    monkeypatch.setattr("core.llm.OllamaClient.__init__", no_ai)
    next(button for button in app.button if button.label == "Explain these results (AI)").click().run()
    assert not app.exception
    assert any("Search again" in item.value for item in app.error)
    manager.close()


def test_running_library_operation_disables_conflicting_actions(
    settings, fake_embedder, pdf_bytes, monkeypatch
):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("VISION_ENABLED", "true")
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    manager.ingest(pdf_bytes("The last completed index is still safe."), "busy.pdf")
    with FileLock(str(settings.data_dir / "metadata" / "library.lock")):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=10)
        assert not app.exception
        assert not app.error
        assert any("already running" in info.value for info in app.info)
        assert not any("Select Reindex to analyze" in info.value for info in app.info)
        assert app.chat_input[0].disabled
        for label in ("Process documents", "Reindex", "Delete", "Delete all documents"):
            assert next(button for button in app.button if button.label == label).disabled
        assert not next(button for button in app.button if button.label == "View original").disabled
    next(button for button in app.button if button.label == "Refresh indexing status").click().run()
    assert not app.exception
    assert not next(button for button in app.button if button.label == "Reindex").disabled
    assert not app.chat_input[0].disabled
    manager.store.close()


def test_lock_race_is_reported_as_busy_not_an_upload_failure(
    settings, fake_embedder, pdf_bytes, monkeypatch
):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    manager.ingest(pdf_bytes("Keep this original during a competing request."), "race.pdf")
    monkeypatch.setattr("core.embeddings.Embeddings", lambda _settings: fake_embedder)
    monkeypatch.setattr("core.library_lock.library_is_busy", lambda _data: False)
    with FileLock(str(settings.data_dir / "metadata" / "library.lock")):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=10)
        next(button for button in app.button if button.label == "Reindex").click().run(timeout=10)
        assert not app.exception
        assert not app.error
        assert any("busy with another operation" in info.value for info in app.info)
    assert manager.catalog.list_documents()[0]["error"] == ""
    manager.store.close()
