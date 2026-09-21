from dataclasses import replace
from functools import lru_cache

from config.settings import Settings
from core.errors import ModelError
from core.types import SearchHit


@lru_cache(maxsize=2)
def load_reranker(name: str, revision: str, cache: str, device: str, offline: bool):
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(
            name,
            revision=revision,
            cache_folder=cache,
            device=device,
            local_files_only=offline,
            trust_remote_code=False,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelError(
            f"Cannot load reranker {name}. Run 'python -m scripts.download_models' "
            f"while online, or turn off reranking in Advanced settings. Details: {exc}"
        ) from exc


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        model = load_reranker(
            self.settings.reranker_model, self.settings.reranker_revision,
            str(self.settings.model_cache), self.settings.embedding_device,
            self.settings.offline,
        )
        scores = model.predict(
            [(query, f"Document: {hit.chunk.filename}\n{hit.chunk.text}") for hit in hits],
            batch_size=16, show_progress_bar=False,
        )
        ranked = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(
            ranked,
            key=lambda hit: (hit.exact_match, hit.rerank_score, hit.score),
            reverse=True,
        )
