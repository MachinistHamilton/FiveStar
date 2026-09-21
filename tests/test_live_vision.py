import pymupdf
import pytest

from config.settings import Settings
from core.document_manager import DocumentManager
from core.embeddings import Embeddings
from core.llm import OllamaClient
from core.rag_pipeline import RAGPipeline
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker
from scripts.create_examples import create_visual_example

pytestmark = pytest.mark.vision


def test_real_local_vision_indexes_and_answers_image_only_diagram(tmp_path):
    settings = Settings(_env_file=None, vision_enabled=True, data_dir=tmp_path / "data")
    settings = settings.model_copy(update={"ollama_model": settings.vision_model})
    path = create_visual_example(tmp_path / "examples")
    with pymupdf.open(path) as pdf:
        assert not pdf[0].get_text().strip(), "Fixture must require actual image interpretation."
    embedder = Embeddings(settings)
    manager = DocumentManager(settings, embedder, TextChunker(embedder))
    client = OllamaClient(settings)
    try:
        result = manager.ingest(path.read_bytes(), path.name)
        summary = manager.catalog.index_summary(result.document_id)
        assert summary.text_pages == ()
        assert summary.visual_pages == (1,)
        passages = manager.catalog.indexed_passages(result.document_id, summary.generation, 1)
        text = "\n".join(chunk.text for chunk in passages).casefold()
        assert "e104" in text
        assert "inlet" in text
        assert "outlet" in text
        assert "arrow" in text
        assert all(chunk.source_type == "visual" and chunk.source_model for chunk in passages)
        retriever = HybridRetriever(manager, settings)
        answer = RAGPipeline(settings, retriever, client).ask("Which box contains the label E104?")
        assert "inlet" in answer.text.casefold()
        assert "Visual interpretation" in answer.text
        assert answer.sources
        assert all(source.hit.chunk.page_number == 1 for source in answer.sources)
    finally:
        client.close()
        manager.store.close()
