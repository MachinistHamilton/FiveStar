import json
import sqlite3
from dataclasses import replace

import pymupdf
import pytest

from core.catalog import Catalog
from core.citations import Claim, render_claims, validate_sources
from core.document_manager import DocumentManager, delete_document
from core.errors import GenerationError
from core.rag_pipeline import RAGPipeline
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker
from core.types import Chunk, SearchHit, Source, VisualPageResult


class FakeVision:
    model_identity = "vision-test@digest-one"

    def __init__(self):
        self.calls = []
        self.closed = 0
        self.fail_page = None

    def analyze_page(self, image_png, filename, page_number):
        self.calls.append(page_number)
        assert image_png.startswith(b"\x89PNG")
        if page_number == self.fail_page:
            raise GenerationError("Vision inference interrupted.")
        return VisualPageResult(
            "Visible image text: E104. Visual interpretation: the diagram labels "
            "the blocked inlet with E104. The arrow points to the inlet valve.",
            ("The small label may be hard to read; verify the original.",),
        )

    def close(self):
        self.closed += 1


@pytest.fixture
def visual_pdf():
    def make(pages=1):
        with pymupdf.open() as pdf:
            for _ in range(pages):
                page = pdf.new_page()
                page.insert_text((50, 50), "A procedure with a separate diagram below.")
                page.draw_rect(pymupdf.Rect(50, 100, 200, 200), color=(0, 0, 0))
                page.draw_line(pymupdf.Point(200, 150), pymupdf.Point(300, 150))
            return pdf.tobytes()
    return make


def make_manager(settings, fake_embedder, vision):
    return DocumentManager(
        settings.model_copy(update={"vision_enabled": True}),
        fake_embedder, TextChunker(fake_embedder), vision_factory=lambda: vision,
    )


def test_visual_evidence_is_separate_searchable_and_cached(settings, fake_embedder, visual_pdf):
    vision = FakeVision()
    manager = make_manager(settings, fake_embedder, vision)
    result = manager.ingest(visual_pdf(), "visual.pdf")
    assert vision.calls == [1]
    assert vision.closed == 1
    summary = manager.catalog.index_summary(result.document_id)
    assert summary.pages == summary.text_pages == summary.visual_pages == (1,)
    passages = manager.catalog.indexed_passages(result.document_id, summary.generation, 1)
    assert {chunk.source_type for chunk in passages} == {"text", "visual"}
    interpreted = next(chunk for chunk in passages if chunk.source_type == "visual")
    assert interpreted.source_model == vision.model_identity
    assert any("Visual interpretation was indexed" in note for note in result.warnings)
    assert any("small label" in note for note in result.warnings)
    retriever = HybridRetriever(manager, settings)
    assert retriever.retrieve("What does E104 label?")[0].chunk.source_type == "visual"
    manager.reindex(result.document_id)
    assert vision.calls == [1], "Reindex must reuse saved visual analysis."
    restarted = make_manager(settings, fake_embedder, vision)
    restarted.reindex(result.document_id)
    assert vision.calls == [1], "Visual cache must survive new manager instances."
    assert restarted.catalog.index_summary(result.document_id).chunk_count == len(passages)
    manager.store.close()
    restarted.store.close()


def test_visual_failure_does_not_publish_partial_generation_and_can_resume(
    settings, fake_embedder, visual_pdf
):
    vision = FakeVision()
    vision.fail_page = 2
    manager = make_manager(settings, fake_embedder, vision)
    data = visual_pdf(2)
    with pytest.raises(GenerationError, match="interrupted"):
        manager.ingest(data, "two-pages.pdf")
    assert vision.calls == [1, 2]
    assert manager.catalog.ready_generations(manager.profile) == []
    with manager.catalog.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM visual_pages").fetchone()[0] == 1
    vision.fail_page = None
    result = manager.ingest(data, "two-pages.pdf")
    assert vision.calls == [1, 2, 2]
    assert manager.catalog.index_summary(result.document_id).visual_pages == (1, 2)
    manager.store.close()


def test_model_change_invalidates_visual_cache_and_delete_clears_it(
    settings, fake_embedder, visual_pdf
):
    vision = FakeVision()
    manager = make_manager(settings, fake_embedder, vision)
    result = manager.ingest(visual_pdf(), "revision.pdf")
    first = manager.catalog.index_summary(result.document_id)
    before = [
        chunk.chunk_id for chunk in manager.catalog.indexed_passages(
            result.document_id, first.generation, 1
        ) if chunk.source_type == "visual"
    ]
    vision.model_identity = "vision-test@digest-two"
    manager.reindex(result.document_id)
    assert vision.calls == [1, 1]
    second = manager.catalog.index_summary(result.document_id)
    after = [
        chunk.chunk_id for chunk in manager.catalog.indexed_passages(
            result.document_id, second.generation, 1
        ) if chunk.source_type == "visual"
    ]
    assert before != after
    delete_document(settings.data_dir, result.document_id)
    with manager.catalog.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM visual_pages").fetchone()[0] == 0
    manager.store.close()


def test_text_only_pages_do_not_trigger_visual_calls(settings, fake_embedder, pdf_bytes):
    vision = FakeVision()
    manager = make_manager(settings, fake_embedder, vision)
    result = manager.ingest(pdf_bytes("Selectable text without any image or drawing."), "text.pdf")
    assert vision.calls == []
    assert manager.catalog.index_summary(result.document_id).visual_pages == ()
    assert manager.catalog.index_summary(result.document_id).text_pages == (1,)
    manager.store.close()


def test_visual_and_text_only_profiles_are_distinct(settings, fake_embedder, visual_pdf):
    text_manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = text_manager.ingest(visual_pdf(), "mixed.pdf")
    visual_manager = make_manager(settings, fake_embedder, FakeVision())
    assert text_manager.profile != visual_manager.profile
    assert visual_manager.catalog.ready_generations(visual_manager.profile) == []
    visual_manager.reindex(result.document_id)
    assert text_manager.catalog.ready_generations(text_manager.profile)
    assert visual_manager.catalog.index_summary(result.document_id).visual_pages == (1,)
    text_manager.store.close()
    visual_manager.store.close()


def test_schema_migrates_existing_text_chunks_without_data_loss(tmp_path):
    data_dir = tmp_path / "existing"
    (data_dir / "metadata").mkdir(parents=True)
    path = data_dir / "metadata" / "catalog.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE chunks (
                generation TEXT, chunk_id TEXT, document_id TEXT, filename TEXT,
                page_number INTEGER, page_end INTEGER, text TEXT, page_label TEXT,
                PRIMARY KEY (generation, chunk_id)
            )
        """)
        db.execute("INSERT INTO chunks VALUES ('g','c','d','old.pdf',1,1,'Old indexed text','')")
    catalog = Catalog(data_dir)
    chunks = catalog.read_chunks(["g"])
    assert len(chunks) == 1
    assert chunks[0].text == "Old indexed text"
    assert chunks[0].source_type == "text"
    assert chunks[0].source_model == ""


def test_visual_claims_and_sources_are_explicitly_labeled(settings):
    chunk = Chunk(
        "visual", "document", "diagram.pdf", 4, 4,
        "The arrow points to the inlet valve.",
        source_type="visual", source_model="test-vision@digest",
    )
    source = Source(1, SearchHit(chunk, 1.0))
    text = render_claims([Claim("The arrow points to the valve.", (1,))], [source])
    assert "Visual interpretation (verify against the original)" in text
    assert "physical PDF page 4; visual interpretation" in text
    with pytest.raises(GenerationError, match="Source metadata"):
        validate_sources([Source(1, SearchHit(replace(chunk, source_model=""), 1.0))])

    class Retriever:
        def retrieve(self, *_args, **_kwargs):
            return [source.hit]

    class Chat:
        def chat(self, messages, **kwargs):
            assert "unverified_visual_interpretation" in messages[1]["content"]
            assert "test-vision@digest" in messages[1]["content"]
            return json.dumps({
                "sufficient": True,
                "claims": [{"text": "The arrow points to the inlet valve.", "sources": [1]}],
            })

    answer = RAGPipeline(settings, Retriever(), Chat()).ask("Where does the arrow point?")
    assert "Visual interpretation" in answer.text
    assert any("saved model interpretations" in warning for warning in answer.warnings)
