import hashlib
import itertools
import json
import logging
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from chromadb.errors import ChromaError

from config.settings import Settings
from core.catalog import Catalog
from core.errors import AssistantError, IndexError, PDFError
from core.library_lock import LibraryLock
from core.pdf_processor import content_hash, iter_pdf_pages, safe_filename
from core.types import Chunker, Embedder, PageVision, Progress
from core.vector_store import VectorStore
from core.visual_cache import CachedPageVision, vision_configuration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    duplicate: bool
    chunk_count: int
    warnings: list[str]


class DocumentManager:
    def __init__(
        self, settings: Settings, embedder: Embedder | None, chunker: Chunker,
        *, vision_factory: Callable[[], PageVision] | None = None,
    ):
        self.settings = settings
        self.catalog = Catalog(settings.data_dir)
        self.embedder = embedder
        self.chunker = chunker
        self.vision_factory = vision_factory
        self.vision_configuration = vision_configuration(settings)
        if embedder is None and settings.vision_enabled:
            raise ValueError("Keyword-only ingestion must not run automatic visual analysis.")
        identity = f"{embedder.fingerprint if embedder else 'keyword-only'}:{chunker.fingerprint}"
        if settings.vision_enabled:
            identity += f":visual-v1:{self.vision_configuration}"
        self.profile = hashlib.sha256(
            identity.encode()
        ).hexdigest()[:40]
        self.lock = LibraryLock(settings.data_dir)
        with self.lock:
            self.store = VectorStore(settings.data_dir / "chroma", self.profile) if embedder else None
            self._recover()

    def _recover(self) -> None:
        with self.catalog.connect() as db:
            abandoned = db.execute(
                "SELECT id FROM generations WHERE profile=? AND state!='ready'", (self.profile,)
            ).fetchall()
            deleting = db.execute("SELECT id FROM documents WHERE deleting=1").fetchall()
        for row in abandoned:
            if self.store is not None:
                self.store.delete_generation(row["id"])
            with self.catalog.connect() as db:
                db.execute("DELETE FROM generations WHERE id=?", (row["id"],))
        for row in deleting:
            _delete_document_locked(self.catalog, row["id"])

    def ingest(
        self, data: bytes, filename: str, *, ocr: bool = False, progress: Progress | None = None
    ) -> IngestResult:
        name = safe_filename(filename)
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise PDFError(f"File exceeds the {self.settings.max_upload_mb} MB upload limit.")
        if not data or b"%PDF-" not in data[:1024]:
            raise PDFError("This file does not have a valid PDF header.")
        document_id = content_hash(data)
        with self.lock:
            self._recover()
            path = self.catalog.document_path(document_id)
            with self.catalog.connect() as db:
                db.execute(
                    "INSERT OR IGNORE INTO documents(id, filename) VALUES (?, ?)",
                    (document_id, name),
                )
            if not path.exists():
                temporary = path.with_suffix(".upload")
                temporary.write_bytes(data)
                temporary.replace(path)
            if self.catalog.ready_generations(self.profile, [document_id]):
                document = self.catalog.get_document(document_id)
                return IngestResult(
                    document_id, True,
                    len(self.catalog.read_chunks(
                        self.catalog.ready_generations(self.profile, [document_id])
                    )),
                    document["warnings"],
                )
            return self._index(document_id, ocr=ocr, progress=progress)

    def reindex(
        self, document_id: str, *, ocr: bool = False, progress: Progress | None = None
    ) -> IngestResult:
        with self.lock:
            self._recover()
            return self._index(document_id, ocr=ocr, progress=progress)

    def _index(self, document_id: str, *, ocr: bool, progress: Progress | None) -> IngestResult:
        document = self.catalog.get_document(document_id)
        path = self.catalog.document_path(document_id)
        if not path.is_file():
            raise IndexError("Original PDF is missing. Upload it again before reindexing.")
        with path.open("rb") as saved_pdf:
            digest = hashlib.file_digest(saved_pdf, "sha256").hexdigest()
        if digest != document_id:
            raise IndexError("The saved PDF has changed. Re-upload it as a new document.")
        generation = uuid.uuid4().hex
        with self.catalog.connect() as db:
            db.execute(
                "INSERT INTO generations(id,document_id,profile,state) VALUES (?,?,?,'staging')",
                (generation, document_id, self.profile),
            )
            db.execute("UPDATE documents SET error='' WHERE id=?", (document_id,))
        warnings: list[str] = []
        page_count = 0
        vision: CachedPageVision | None = None

        def pages():
            nonlocal page_count
            for page in iter_pdf_pages(
                path, document_id, document["filename"], ocr=ocr,
                ocr_language=self.settings.ocr_language,
                tessdata_prefix=self.settings.tessdata_prefix,
                progress=progress,
                vision=vision,
                vision_max_image_edge=self.settings.vision_max_image_edge,
            ):
                page_count = page.page_number
                warnings.extend(page.warnings)
                yield page

        count = 0
        try:
            if self.settings.vision_enabled:
                if self.vision_factory:
                    analyzer = self.vision_factory()
                else:
                    from core.vision import VisionAnalyzer
                    analyzer = VisionAnalyzer(self.settings)
                vision = CachedPageVision(
                    self.catalog, document_id, analyzer, self.vision_configuration
                )
            chunks = iter(self.chunker.chunk_pages(pages()))
            while batch := list(itertools.islice(chunks, self.settings.embedding_batch_size)):
                if self.embedder is not None and self.store is not None:
                    vectors = self.embedder.encode([chunk.text for chunk in batch])
                    self.store.add(generation, batch, vectors)
                with self.catalog.connect() as db:
                    db.executemany("""
                        INSERT INTO chunks(generation,chunk_id,document_id,filename,
                        page_number,page_end,text,page_label,source_type,source_model) VALUES (
                        :generation,:chunk_id,:document_id,:filename,:page_number,
                        :page_end,:text,:page_label,:source_type,:source_model)
                    """, [asdict(chunk) | {"generation": generation} for chunk in batch])
                count += len(batch)
            if not count:
                raise PDFError(
                    "No searchable text was extracted. The original PDF is retained. "
                    "Enable OCR and reindex, or open a page and explicitly analyze its images. "
                    + " ".join(warnings[:3])
                )
            with self.catalog.connect() as db:
                db.execute(
                    "UPDATE generations SET state='retired' "
                    "WHERE document_id=? AND profile=? AND state='ready'",
                    (document_id, self.profile),
                )
                db.execute(
                    "UPDATE generations SET state='ready',chunk_count=? WHERE id=?",
                    (count, generation),
                )
                db.execute(
                    "UPDATE documents SET page_count=?,warnings=?,error='' WHERE id=?",
                    (page_count, json.dumps(warnings), document_id),
                )
        except (AssistantError, OSError, ValueError, RuntimeError, sqlite3.Error,
                ChromaError, pymupdf.mupdf.FzErrorBase) as exc:
            # Keep the journal for deterministic recovery; never expose a partial generation.
            logger.exception("Indexing failed for document %s", document_id)
            with self.catalog.connect() as db:
                db.execute(
                    "UPDATE documents SET error=?,warnings=?,"
                    "page_count=CASE WHEN page_count=0 THEN ? ELSE page_count END WHERE id=?",
                    (str(exc), json.dumps(warnings), page_count, document_id),
                )
            raise
        finally:
            if vision:
                vision.close()
        self._recover()
        if progress:
            progress(1.0, f"Indexed {count} passages from {page_count} PDF pages.")
        return IngestResult(document_id, False, count, warnings)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()


def delete_document(data_dir: Path, document_id: str) -> None:
    catalog = Catalog(data_dir)
    lock = LibraryLock(data_dir)
    with lock:
        _delete_document_locked(catalog, document_id)


def _delete_document_locked(catalog: Catalog, document_id: str) -> None:
    path = catalog.document_path(document_id)
    catalog.get_document(document_id)
    with catalog.connect() as db:
        db.execute("UPDATE documents SET deleting=1 WHERE id=?", (document_id,))
    if (catalog.data_dir / "chroma").exists():
        VectorStore.delete_document_everywhere(catalog.data_dir / "chroma", document_id)
    path.unlink(missing_ok=True)
    path.with_suffix(".upload").unlink(missing_ok=True)
    with catalog.connect() as db:
        db.execute("DELETE FROM documents WHERE id=?", (document_id,))


def clear_library(data_dir: Path, *, confirmed: bool = False) -> None:
    if not confirmed:
        raise IndexError("Confirm permanent deletion before clearing the library.")
    catalog = Catalog(data_dir)
    with LibraryLock(data_dir):
        for document in catalog.list_documents():
            _delete_document_locked(catalog, document["id"])
