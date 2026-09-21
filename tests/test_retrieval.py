from dataclasses import replace

from core.document_manager import DocumentManager, delete_document
from core.keyword_search import KeywordSearch, document_is_named, has_exact_match
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker
from core.types import Chunk


def test_keyword_identifiers_phrases_and_no_false_match():
    chunks = [
        Chunk("1", "a", "a.pdf", 1, 1, "E104 inlet blockage; reset the sensor."),
        Chunk("2", "b", "b.pdf", 2, 2, "E1040 error code is unrelated."),
        Chunk("3", "b", "b.pdf", 3, 3, "The manual calls it an inlet blockage."),
    ]
    index = KeywordSearch(chunks)
    assert index.search("What is E104?", 3)[0][0] == "1"
    assert not has_exact_match(chunks[1].text, ["E104"])
    assert index.search('"inlet blockage"', 3)[0][0] in {"1", "3"}
    assert index.search("E104", 3, ["b"]) == []
    assert index.search("nonexistent", 3) == []
    assert KeywordSearch([]).search("anything", 2) == []


def test_explicit_document_names_are_prioritized_without_substring_matches():
    assert document_is_named("Equipment_Manual.pdf", "What does the Equipment Manual say?")
    assert not document_is_named("Equipment_Manual.pdf", "Equipment ManualXYZ?")
    assert not document_is_named("a.pdf", "a generic question")
    chunks = [
        Chunk("a", "a", "Equipment_Manual.pdf", 1, 1, "Inspection interval is twelve months."),
        Chunk("b", "b", "Service_Update.pdf", 1, 1, "The Equipment Manual inspection interval changed."),
    ]
    hits = KeywordSearch(chunks).search("inspection interval in Equipment Manual", 2)
    assert hits[0][0] == "a"


def test_hybrid_filter_rerank_and_delete_refresh(settings, fake_embedder, pdf_bytes):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    a = manager.ingest(
        pdf_bytes("Error code E104 means inlet blockage.", "Inspect every twelve months."),
        "manual.pdf",
    )
    b = manager.ingest(pdf_bytes("E1040 is unrelated to E1041."), "other.pdf")
    retriever = HybridRetriever(manager, settings)
    assert retriever.retrieve("Explain error E104")[0].chunk.document_id == a.document_id
    assert all(
        hit.chunk.document_id == b.document_id
        for hit in retriever.retrieve("error", [b.document_id])
    )
    assert retriever.retrieve("error", []) == []

    class ReverseReranker:
        def rerank(self, query, hits):
            return [replace(hit, rerank_score=0.5) for hit in reversed(hits)]
    retriever.reranker = ReverseReranker()
    assert all(hit.rerank_score == 0.5 for hit in retriever.retrieve("inspection"))
    delete_document(settings.data_dir, a.document_id)
    assert all(hit.chunk.document_id != a.document_id for hit in retriever.retrieve("E104"))


def test_real_vectors_persist_and_filter(settings, fake_embedder, pdf_bytes):
    manager = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    a = manager.ingest(pdf_bytes("Alpha E104 identifier and inlet blockage."), "a.pdf")
    manager.ingest(pdf_bytes("Beta travel schedules and seat reservations."), "b.pdf")
    vector = fake_embedder.encode(["Alpha E104 identifier and inlet blockage."])[0]
    generations = manager.catalog.ready_generations(manager.profile, [a.document_id])
    result = manager.store.search(vector, generations, 20)
    assert len(result) == 1
    assert result[0][1] > 0.99
    reopened = DocumentManager(settings, fake_embedder, TextChunker(fake_embedder))
    assert reopened.store.search(vector, generations, 20) == result
