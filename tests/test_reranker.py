from unittest.mock import Mock

from core.reranker import Reranker
from core.types import Chunk, SearchHit


def test_reranker_scores_all_candidates_but_preserves_exact_priority(settings, monkeypatch):
    model = Mock()
    model.predict.return_value = [-2.0, 8.0, 3.0]
    monkeypatch.setattr("core.reranker.load_reranker", lambda *args: model)
    hits = [
        SearchHit(Chunk("a", "d", "a.pdf", 1, 1, "E104 is a blockage."), 0.1, exact_match=True),
        SearchHit(Chunk("b", "d", "a.pdf", 2, 2, "Errors in general."), 0.3),
        SearchHit(Chunk("c", "d", "a.pdf", 3, 3, "Other information."), 0.2),
    ]
    reranker = Reranker(settings)
    ranked = reranker.rerank("What does E104 mean?", hits)
    assert [hit.chunk.chunk_id for hit in ranked] == ["a", "b", "c"]
    assert [hit.rerank_score for hit in ranked] == [-2.0, 8.0, 3.0]
    assert len(model.predict.call_args.args[0]) == 3
    assert reranker.rerank("nothing", []) == []
