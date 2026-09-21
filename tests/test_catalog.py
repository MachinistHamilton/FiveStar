from core.document_manager import DocumentManager, delete_document
from core.text_chunker import TextChunker


def test_coverage_and_preview_come_from_persisted_ready_chunks(settings, fake_embedder, pdf_bytes):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(
        pdf_bytes("First searchable page with normal text.", "", "Third searchable page."),
        "coverage.pdf",
    )
    summary = manager.catalog.index_summary(result.document_id)
    assert summary is not None
    assert summary.pages == (1, 3)
    assert summary.chunk_count == 2
    assert manager.catalog.indexed_page(result.document_id, summary.generation, 1) == [
        "First searchable page with normal text."
    ]
    assert manager.catalog.indexed_page(result.document_id, summary.generation, 2) == []
    assert manager.catalog.indexed_page("different-document", summary.generation, 1) == []
    manager.store.close()


def test_summary_excludes_staged_and_deleting_generations(settings, fake_embedder, pdf_bytes):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(pdf_bytes("Ready text for a complete index."), "ready.pdf")
    before = manager.catalog.index_summary(result.document_id)
    with manager.catalog.connect() as db:
        db.execute(
            "INSERT INTO generations(id,document_id,profile,state) VALUES (?,?,?,'staging')",
            ("pending", result.document_id, "different-profile"),
        )
    assert manager.catalog.index_summary(result.document_id) == before
    assert manager.catalog.unfinished_documents() == ["ready.pdf"]
    assert manager.catalog.unfinished_documents([]) == []
    assert manager.catalog.unfinished_documents([result.document_id]) == ["ready.pdf"]
    with manager.catalog.connect() as db:
        db.execute("UPDATE documents SET deleting=1 WHERE id=?", (result.document_id,))
    assert manager.catalog.unfinished_documents() == []
    assert manager.catalog.index_summary(result.document_id) is None
    assert manager.catalog.indexed_page(result.document_id, before.generation, 1) == []
    delete_document(settings.data_dir, result.document_id)
    assert manager.catalog.index_summary(result.document_id) is None
    manager.store.close()


def test_summary_uses_single_most_recent_successful_index(settings, fake_embedder, pdf_bytes):
    first = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = first.ingest(pdf_bytes("Some searchable text to index once per profile."), "profiles.pdf")
    original = first.catalog.index_summary(result.document_id)
    fake_embedder.fingerprint = "another-embedding-model"
    second = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    second.reindex(result.document_id)
    summary = second.catalog.index_summary(result.document_id)
    assert summary.profile == second.profile
    assert summary.generation != original.generation
    assert summary.chunk_count == 1
    assert summary.pages == (1,)
    assert second.catalog.list_documents()[0]["chunk_count"] == 2
    first.store.close()
    second.store.close()
