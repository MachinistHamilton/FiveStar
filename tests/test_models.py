import json

import pytest

from config.settings import PROJECT_ROOT, Settings
from core.document_manager import DocumentManager
from core.embeddings import Embeddings
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker
from scripts.create_examples import create_examples

pytestmark = pytest.mark.models


def test_real_offline_models_retrieve_known_sources(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("Offline evaluation must not open a network connection.")

    monkeypatch.setattr("socket.socket.connect", no_network)
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", offline=True)
    embedder = Embeddings(settings)
    manager = DocumentManager(settings, embedder, TextChunker(embedder))
    for path in create_examples(tmp_path / "examples"):
        manager.ingest(path.read_bytes(), path.name)
    retriever = HybridRetriever(manager, settings)
    cases = json.loads((PROJECT_ROOT / "examples" / "questions.json").read_text())
    for case in cases:
        if "document" not in case:
            continue
        hits = retriever.retrieve(case["question"])
        assert (hits[0].chunk.filename, hits[0].chunk.page_number) == (
            case["document"], case["page"]
        )
        assert all(fact in hits[0].chunk.text for fact in case["required_facts"])
        assert hits[0].rerank_score is not None
    conflict = retriever.retrieve(cases[3]["question"])[:settings.evidence_count]
    assert {"Equipment_Manual.pdf", "Service_Update.pdf"} <= {
        hit.chunk.filename for hit in conflict
    }
