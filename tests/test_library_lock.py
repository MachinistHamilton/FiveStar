import time

import pytest
from filelock import FileLock

from core.document_manager import DocumentManager, clear_library, delete_document
from core.errors import LibraryBusyError
from core.library_lock import LibraryLock, library_is_busy
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker


def test_busy_probe_does_not_release_the_owners_lock(tmp_path):
    lock = LibraryLock(tmp_path)
    assert not library_is_busy(tmp_path)
    with lock:
        assert library_is_busy(tmp_path)
        with lock:
            assert library_is_busy(tmp_path)
        assert library_is_busy(tmp_path)
        started = time.monotonic()
        with pytest.raises(LibraryBusyError, match="busy with another operation"):
            with LibraryLock(tmp_path):
                pytest.fail("A competing owner must not enter.")
        assert time.monotonic() - started < 2
        assert library_is_busy(tmp_path)
    assert not library_is_busy(tmp_path)


def test_existing_lock_file_is_not_assumed_to_be_held(tmp_path):
    path = tmp_path / "metadata" / "library.lock"
    path.parent.mkdir()
    path.touch()
    assert not library_is_busy(tmp_path)
    with LibraryLock(tmp_path):
        assert library_is_busy(tmp_path)


def test_all_library_operations_reject_competing_owner_without_changing_data(
    settings, fake_embedder, pdf_bytes
):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(pdf_bytes("An existing indexed document."), "original.pdf")
    before = manager.catalog.index_summary(result.document_id)
    operations = [
        lambda: manager.ingest(pdf_bytes("Another document."), "second.pdf"),
        lambda: manager.reindex(result.document_id),
        lambda: delete_document(settings.data_dir, result.document_id),
        lambda: clear_library(settings.data_dir, confirmed=True),
        lambda: HybridRetriever(manager, settings).retrieve("existing indexed document"),
    ]
    legacy_lock = FileLock(str(settings.data_dir / "metadata" / "library.lock"))
    with legacy_lock:
        for operation in operations:
            started = time.monotonic()
            with pytest.raises(LibraryBusyError, match="Refresh indexing status"):
                operation()
            assert time.monotonic() - started < 2
            assert library_is_busy(settings.data_dir)
        assert manager.catalog.index_summary(result.document_id) == before
        assert len(manager.catalog.list_documents()) == 1
        assert manager.catalog.get_document(result.document_id)["error"] == ""
        assert manager.catalog.document_path(result.document_id).exists()
    manager.reindex(result.document_id)
    assert manager.catalog.index_summary(result.document_id).generation != before.generation
    manager.store.close()


def test_busy_constructor_does_not_open_chroma_or_recover_an_active_generation(
    settings, fake_embedder, pdf_bytes, monkeypatch
):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    result = manager.ingest(pdf_bytes("An already indexed document."), "original.pdf")
    with manager.catalog.connect() as db:
        db.execute(
            "INSERT INTO generations(id,document_id,profile,state) VALUES (?,?,?,'staging')",
            ("active-generation", result.document_id, manager.profile),
        )

    def unexpected_store(*_args):
        pytest.fail("Busy startup must not open a second vector-store client.")

    monkeypatch.setattr("core.document_manager.VectorStore", unexpected_store)
    with FileLock(str(settings.data_dir / "metadata" / "library.lock")):
        with pytest.raises(LibraryBusyError):
            DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
        with manager.catalog.connect() as db:
            assert db.execute(
                "SELECT state FROM generations WHERE id='active-generation'"
            ).fetchone()[0] == "staging"
    manager.store.close()
