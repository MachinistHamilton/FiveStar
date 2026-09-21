import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from core.errors import IndexError
from core.types import Chunk


@dataclass(frozen=True)
class IndexSummary:
    generation: str
    profile: str
    chunk_count: int
    pages: tuple[int, ...]
    text_pages: tuple[int, ...]
    visual_pages: tuple[int, ...]


@dataclass(frozen=True)
class SavedVisualPage:
    cache_key: str
    document_id: str
    filename: str
    page_number: int
    model_identity: str
    text: str
    warnings: tuple[str, ...]


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(**{key: row[key] for key in (
        "chunk_id", "document_id", "filename", "page_number", "page_end",
        "text", "page_label", "source_type", "source_model",
    )})


class Catalog:
    """SQLite is the authority for which index generations are visible to readers."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents_dir = data_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "metadata").mkdir(exist_ok=True)
        self.path = data_dir / "metadata" / "catalog.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    warnings TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT '',
                    deleting INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    profile TEXT NOT NULL, state TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS generations_profile
                    ON generations(profile, state);
                CREATE TABLE IF NOT EXISTS chunks (
                    generation TEXT NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
                    chunk_id TEXT NOT NULL, document_id TEXT NOT NULL,
                    filename TEXT NOT NULL, page_number INTEGER NOT NULL,
                    page_end INTEGER NOT NULL, text TEXT NOT NULL, page_label TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'text',
                    source_model TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(generation, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS visual_pages (
                    cache_key TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL, model_identity TEXT NOT NULL,
                    text TEXT NOT NULL, warnings TEXT NOT NULL
                );
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)")}
            if "source_type" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'text'")
            if "source_model" not in columns:
                db.execute("ALTER TABLE chunks ADD COLUMN source_model TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def document_path(self, document_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", document_id):
            raise IndexError("Invalid document identifier.")
        return self.documents_dir / f"{document_id}.pdf"

    def list_documents(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT d.*, COALESCE(SUM(CASE WHEN g.state='ready' THEN g.chunk_count
                ELSE 0 END), 0) AS chunk_count FROM documents d
                LEFT JOIN generations g ON g.document_id=d.id
                GROUP BY d.id ORDER BY d.created_at, d.filename
            """).fetchall()
        return [dict(row) | {"warnings": json.loads(row["warnings"])} for row in rows]

    def get_document(self, document_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise IndexError("Document no longer exists. Refresh the library.")
        return dict(row) | {"warnings": json.loads(row["warnings"])}

    def unfinished_documents(self, document_ids: list[str] | None = None) -> list[str]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT DISTINCT d.id, d.filename FROM documents d
                JOIN generations g ON g.document_id=d.id
                WHERE g.state='staging' AND d.deleting=0 AND d.error=''
                ORDER BY d.filename
            """).fetchall()
        return [
            row["filename"] for row in rows
            if document_ids is None or row["id"] in document_ids
        ]

    def ready_generations(
        self, profile: str, document_ids: list[str] | None = None
    ) -> list[str]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT g.id, g.document_id FROM generations g
                JOIN documents d ON d.id=g.document_id
                WHERE g.profile=? AND g.state='ready' AND d.deleting=0 ORDER BY g.id
            """, (profile,)).fetchall()
        return [
            row["id"] for row in rows
            if document_ids is None or row["document_id"] in document_ids
        ]

    def latest_ready_generations(self, document_ids: list[str] | None = None) -> list[str]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT g.id, g.document_id FROM generations g
                JOIN documents d ON d.id=g.document_id
                WHERE g.state='ready' AND d.deleting=0 AND g.rowid=(
                    SELECT MAX(newer.rowid) FROM generations newer
                    WHERE newer.document_id=g.document_id AND newer.state='ready'
                ) ORDER BY g.id
            """).fetchall()
        return [
            row["id"] for row in rows
            if document_ids is None or row["document_id"] in document_ids
        ]

    def saved_visual_pages(self, document_ids: list[str] | None = None) -> list[SavedVisualPage]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT v.*, d.filename FROM visual_pages v
                JOIN documents d ON d.id=v.document_id
                WHERE d.deleting=0 AND v.rowid=(
                    SELECT MAX(newer.rowid) FROM visual_pages newer
                    WHERE newer.document_id=v.document_id AND newer.page_number=v.page_number
                ) ORDER BY v.document_id,v.page_number
            """).fetchall()
        return [
            SavedVisualPage(
                row["cache_key"], row["document_id"], row["filename"], row["page_number"],
                row["model_identity"], row["text"], tuple(json.loads(row["warnings"])),
            )
            for row in rows if document_ids is None or row["document_id"] in document_ids
        ]

    def read_chunks(self, generations: list[str]) -> list[Chunk]:
        chunks = []
        with self.connect() as db:
            for generation in generations:
                rows = db.execute(
                    "SELECT * FROM chunks WHERE generation=? ORDER BY chunk_id", (generation,)
                )
                for row in rows:
                    chunks.append(_chunk_from_row(row))
        return chunks

    def index_summary(self, document_id: str) -> IndexSummary | None:
        with self.connect() as db:
            generation = db.execute("""
                SELECT g.id, g.profile FROM generations g
                JOIN documents d ON d.id=g.document_id
                WHERE g.document_id=? AND g.state='ready' AND d.deleting=0
                ORDER BY g.rowid DESC LIMIT 1
            """, (document_id,)).fetchone()
            if generation is None:
                return None
            rows = db.execute("""
                SELECT page_number, source_type, COUNT(*) AS count FROM chunks
                WHERE generation=? GROUP BY page_number, source_type ORDER BY page_number
            """, (generation["id"],)).fetchall()
        return IndexSummary(
            generation["id"], generation["profile"],
            sum(row["count"] for row in rows),
            tuple(sorted({row["page_number"] for row in rows})),
            tuple(sorted({row["page_number"] for row in rows if row["source_type"] == "text"})),
            tuple(sorted({row["page_number"] for row in rows if row["source_type"] == "visual"})),
        )

    def indexed_page(self, document_id: str, generation: str, page_number: int) -> list[str]:
        return [chunk.text for chunk in self.indexed_passages(document_id, generation, page_number)]

    def indexed_passages(self, document_id: str, generation: str, page_number: int) -> list[Chunk]:
        with self.connect() as db:
            rows = db.execute("""
                SELECT c.* FROM chunks c
                JOIN generations g ON g.id=c.generation
                JOIN documents d ON d.id=g.document_id
                WHERE g.id=? AND g.document_id=? AND c.page_number=?
                    AND g.state='ready' AND d.deleting=0
                ORDER BY c.rowid
            """, (generation, document_id, page_number)).fetchall()
        return [_chunk_from_row(row) for row in rows]
