import pytest

from core.document_manager import DocumentManager, clear_library, delete_document
from core.errors import IndexError, PDFError
from core.text_chunker import TextChunker


def make_manager(settings, embedder):
    return DocumentManager(settings, embedder, TextChunker(embedder))


def test_duplicate_restart_and_batched_index(settings, fake_embedder, pdf_bytes):
    manager = make_manager(settings, fake_embedder)
    data = pdf_bytes(*[
        f"Section {page}. The regular inspection interval is twelve months for device {page}."
        for page in range(1, 502)
    ])
    result = manager.ingest(data, "large.pdf")
    assert result.chunk_count == 501
    assert max(fake_embedder.batches) <= settings.embedding_batch_size
    count = len(fake_embedder.batches)
    restarted = make_manager(settings, fake_embedder)
    duplicate = restarted.ingest(data, "renamed.pdf")
    assert duplicate.duplicate
    assert duplicate.document_id == result.document_id
    assert len(fake_embedder.batches) == count
    document = restarted.catalog.get_document(result.document_id)
    assert document["filename"] == "large.pdf"
    assert document["page_count"] == 501
    assert restarted.catalog.document_path(result.document_id).read_bytes() == data
    restarted.catalog.document_path(result.document_id).unlink()
    assert restarted.ingest(data, "restored.pdf").duplicate
    assert restarted.catalog.document_path(result.document_id).read_bytes() == data
    assert len(fake_embedder.batches) == count


def test_failed_reindex_keeps_last_good_generation(settings, fake_embedder, pdf_bytes, monkeypatch):
    manager = make_manager(settings, fake_embedder)
    result = manager.ingest(pdf_bytes("A complete and searchable original document."), "original.pdf")
    before = manager.catalog.ready_generations(manager.profile)

    def fail(_texts):
        raise RuntimeError("Simulated interrupted embedding")

    with monkeypatch.context() as patch:
        patch.setattr(fake_embedder, "encode", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            manager.reindex(result.document_id)
    assert manager.catalog.ready_generations(manager.profile) == before
    assert "interrupted" in manager.catalog.get_document(result.document_id)["error"]
    restarted = make_manager(settings, fake_embedder)
    with restarted.catalog.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == 1
    restarted.reindex(result.document_id)
    assert restarted.catalog.ready_generations(restarted.profile) != before
    assert restarted.store.collection.count() == result.chunk_count
    assert not restarted.catalog.get_document(result.document_id)["error"]


def test_delete_removes_all_profiles_without_touching_neighbor(settings, fake_embedder, pdf_bytes):
    first = make_manager(settings, fake_embedder)
    a = first.ingest(pdf_bytes("Unique searchable document A."), "a.pdf")
    b = first.ingest(pdf_bytes("Unique searchable document B."), "b.pdf")
    fake_embedder.fingerprint = "a-different-model"
    second = make_manager(settings, fake_embedder)
    assert first.profile != second.profile
    second.reindex(a.document_id)
    assert second.store.collection.count() == 1
    delete_document(settings.data_dir, a.document_id)
    assert first.store.collection.count() == 1
    assert second.store.collection.count() == 0
    assert not first.catalog.document_path(a.document_id).exists()
    assert first.catalog.document_path(b.document_id).exists()
    with first.catalog.connect() as db:
        assert db.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id=?", (a.document_id,)
        ).fetchone()[0] == 0
    with pytest.raises(IndexError, match="Confirm"):
        clear_library(settings.data_dir)
    clear_library(settings.data_dir, confirmed=True)
    assert first.catalog.list_documents() == []
    assert first.store.collection.count() == 0


def test_incomplete_generation_not_searchable_and_recovered(settings, fake_embedder, pdf_bytes, monkeypatch):
    manager = make_manager(settings, fake_embedder)
    add = manager.store.add

    def fail_after_vector_write(*args):
        add(*args)
        raise RuntimeError("Disk write failed")

    monkeypatch.setattr(manager.store, "add", fail_after_vector_write)
    with pytest.raises(RuntimeError):
        manager.ingest(pdf_bytes("This data must not appear in search results."), "failure.pdf")
    assert manager.catalog.ready_generations(manager.profile) == []
    assert manager.store.collection.count() == 1
    recovered = make_manager(settings, fake_embedder)
    assert recovered.store.collection.count() == 0
    assert recovered.catalog.list_documents()[0]["error"] == "Disk write failed"


def test_corrupt_and_blank_are_reported_and_retained(settings, fake_embedder, pdf_bytes):
    manager = make_manager(settings, fake_embedder)
    with pytest.raises(PDFError):
        manager.ingest(b"%PDF-1.4\nbroken contents", "bad.pdf")
    with pytest.raises(PDFError, match="No searchable text"):
        manager.ingest(pdf_bytes(""), "scan.pdf")
    assert len(manager.catalog.list_documents()) == 2
    assert all(doc["error"] for doc in manager.catalog.list_documents())
    assert manager.catalog.ready_generations(manager.profile) == []
    assert manager.catalog.list_documents()[1]["warnings"]


def test_upload_limit_and_paths(settings, fake_embedder, pdf_bytes):
    manager = make_manager(settings.model_copy(update={"max_upload_mb": 1}), fake_embedder)
    with pytest.raises(PDFError, match="limit"):
        manager.ingest(b"%PDF-" + b"0" * (1024 * 1024), "large.pdf")
    with pytest.raises(IndexError):
        manager.catalog.document_path("..\\outside")
    result = manager.ingest(pdf_bytes("A document with a safe content hash."), "..\\..\\safe.pdf")
    assert manager.catalog.get_document(result.document_id)["filename"] == "safe.pdf"


def test_delete_failure_hidden_and_retryable(settings, fake_embedder, pdf_bytes, monkeypatch):
    manager = make_manager(settings, fake_embedder)
    result = manager.ingest(pdf_bytes("A document that will be deleted."), "delete.pdf")
    with monkeypatch.context() as patch:
        def fail(*_args):
            raise RuntimeError("Delete interrupted")
        patch.setattr("core.document_manager.VectorStore.delete_document_everywhere", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            delete_document(settings.data_dir, result.document_id)
    assert manager.catalog.ready_generations(manager.profile) == []
    make_manager(settings, fake_embedder)
    assert manager.catalog.list_documents() == []
