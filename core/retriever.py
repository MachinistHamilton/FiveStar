from config.settings import Settings
from core.document_manager import DocumentManager
from core.errors import ModelError
from core.keyword_search import KeywordSearch, document_is_named, exact_terms, has_exact_match
from core.reranker import Reranker
from core.types import SearchHit


class HybridRetriever:
    def __init__(self, manager: DocumentManager, settings: Settings):
        if manager.embedder is None or manager.store is None:
            raise ValueError("Hybrid retrieval requires an embedding-backed index.")
        self.manager = manager
        self.settings = settings
        self._snapshot: tuple[str, ...] | None = None
        self._keywords = KeywordSearch([])
        self.reranker = Reranker(settings) if settings.rerank_enabled else None

    def retrieve(self, query: str, document_ids: list[str] | None = None) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Enter a nonempty search query.")
        with self.manager.lock:
            all_generations = self.manager.catalog.ready_generations(self.manager.profile)
            snapshot = tuple(all_generations)
            if snapshot != self._snapshot:
                self._keywords = KeywordSearch(self.manager.catalog.read_chunks(all_generations))
                self._snapshot = snapshot
            generations = self.manager.catalog.ready_generations(
                self.manager.profile, document_ids
            )
            if not generations:
                return []
            if self.manager.embedder is None or self.manager.store is None:
                raise ModelError("Hybrid retrieval requires embeddings. Use keyword search for this index.")
            if self.manager.embedder.count_tokens(query) > self.manager.embedder.max_tokens:
                raise ModelError(
                    "The search question exceeds the embedding model's input limit. "
                    "Please shorten it rather than silently losing part of the question."
                )
            limit = self.settings.candidate_count
            semantic = self.manager.store.search(
                self.manager.embedder.encode([query])[0], generations, limit
            )
            lexical = self._keywords.search(query, limit, document_ids)
            fused: dict[str, float] = {}
            for ranking in (semantic, lexical):
                for rank, (chunk_id, _) in enumerate(ranking, start=1):
                    fused[chunk_id] = fused.get(chunk_id, 0) + 1.0 / (60 + rank)
            semantic_scores, keyword_scores = dict(semantic), dict(lexical)
            terms = exact_terms(query)
            hits = [
                SearchHit(
                    chunk, fused[chunk.chunk_id],
                    semantic_scores.get(chunk.chunk_id), keyword_scores.get(chunk.chunk_id),
                    exact_match=(
                        has_exact_match(chunk.text, terms) or document_is_named(chunk.filename, query)
                    ),
                )
                for chunk in self._keywords.chunks if chunk.chunk_id in fused
            ]
            hits.sort(key=lambda hit: (hit.exact_match, hit.score), reverse=True)
            hits = hits[:limit]
        return self.reranker.rerank(query, hits) if self.reranker else hits
