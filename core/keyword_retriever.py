from pathlib import Path

from core.catalog import Catalog
from core.keyword_search import KeywordSearch, document_is_named, exact_terms, has_exact_match
from core.library_lock import LibraryLock
from core.text_chunker import keyword_chunker
from core.types import PageText, SearchHit


class KeywordRetriever:
    """Search completed local text and cached page interpretations without neural models."""

    def __init__(self, data_dir: Path, limit: int = 10):
        self.catalog = Catalog(data_dir)
        self.lock = LibraryLock(data_dir)
        self.limit = limit
        self._snapshot: tuple[str, ...] | None = None
        self._index = KeywordSearch([])

    def retrieve(self, query: str, document_ids: list[str] | None = None) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Enter a question or keywords to search.")
        if len(query.encode("utf-8")) > 4096:
            raise ValueError("Search is limited to 4096 UTF-8 bytes. Enter a shorter question.")
        with self.lock:
            generations = self.catalog.latest_ready_generations()
            saved = self.catalog.saved_visual_pages()
            snapshot = tuple(generations + [page.cache_key for page in saved])
            if snapshot != self._snapshot:
                visual_keys = {(page.document_id, page.page_number) for page in saved}
                chunks = [
                    chunk for chunk in self.catalog.read_chunks(generations)
                    if chunk.source_type != "visual"
                    or (chunk.document_id, chunk.page_number) not in visual_keys
                ]
                pages = [
                    PageText(
                        page.document_id, page.filename, page.page_number, page.text,
                        source_type="visual", source_model=page.model_identity,
                    )
                    for page in saved
                ]
                chunks.extend(keyword_chunker().chunk_pages(pages))
                self._index = KeywordSearch(chunks)
                self._snapshot = snapshot
            scores = self._index.search(query, self.limit, document_ids)
            chunks_by_id = {chunk.chunk_id: chunk for chunk in self._index.chunks}
            terms = exact_terms(query)
            return [
                SearchHit(
                    chunks_by_id[chunk_id], score, keyword_score=score,
                    exact_match=(
                        has_exact_match(chunks_by_id[chunk_id].text, terms)
                        or document_is_named(chunks_by_id[chunk_id].filename, query)
                    ),
                )
                for chunk_id, score in scores
            ]


class SelectedEvidenceRetriever:
    """Expose only the search passages the user chose to have explained."""

    def __init__(self, hits: list[SearchHit]):
        self.hits = hits

    def retrieve(self, query: str, document_ids: list[str] | None = None) -> list[SearchHit]:
        return [
            hit for hit in self.hits
            if document_ids is None or hit.chunk.document_id in document_ids
        ]
