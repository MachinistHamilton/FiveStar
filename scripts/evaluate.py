import argparse
import json
import tempfile
from pathlib import Path

from config.settings import PROJECT_ROOT, Settings
from core.document_manager import DocumentManager
from core.embeddings import Embeddings
from core.llm import OllamaClient
from core.rag_pipeline import RAGPipeline
from core.retriever import HybridRetriever
from core.text_chunker import TextChunker
from scripts.create_examples import create_examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeatable evaluation using only synthetic PDFs.")
    parser.add_argument("--answers", action="store_true", help="Also evaluate actual Ollama answers.")
    args = parser.parse_args()
    settings = Settings()
    embedder = Embeddings(settings)
    cases = json.loads((PROJECT_ROOT / "examples" / "questions.json").read_text())
    client = OllamaClient(settings)
    failures = 0
    try:
        if args.answers:
            client.check_available()
        with tempfile.TemporaryDirectory(prefix="research-evaluation-") as temporary:
            root = Path(temporary)
            isolated = Settings(**(settings.model_dump() | {"data_dir": root / "data"}))
            manager = DocumentManager(
                isolated, embedder,
                TextChunker(embedder, isolated.chunk_tokens, isolated.chunk_overlap),
            )
            try:
                for path in create_examples(root / "pdfs"):
                    manager.ingest(path.read_bytes(), path.name)
                retriever = HybridRetriever(manager, isolated)
                pipeline = RAGPipeline(isolated, retriever, client)
                retrieval_passes = 0
                answer_passes = 0
                for case in cases:
                    hits = retriever.retrieve(case["question"])[:settings.evidence_count]
                    expected = case.get("sources", [])
                    if "document" in case:
                        expected = [{"document": case["document"], "page": case["page"]}]
                    actual = {(hit.chunk.filename, hit.chunk.page_number) for hit in hits}
                    retrieved = all((item["document"], item["page"]) in actual for item in expected)
                    if expected:
                        retrieval_passes += int(retrieved)
                        failures += int(not retrieved)
                    print(f"\nQ: {case['question']}")
                    print(f"  Evidence: {sorted(actual)}")
                    if expected:
                        print(f"  Source recall check: {'PASS' if retrieved else 'FAIL'}")
                    else:
                        print("  Unanswerable: retrieval alone is not an answerability test.")
                    if args.answers:
                        answer = pipeline.ask(case["question"])
                        facts = case.get("required_facts", [])
                        passed = all(fact.casefold() in answer.text.casefold() for fact in facts)
                        if case.get("unanswerable"):
                            passed = answer.text == (
                                "I could not find sufficient information in the uploaded documents "
                                "to answer this question."
                            )
                        else:
                            passed = passed and bool(answer.sources)
                        answer_passes += int(passed)
                        failures += int(not passed)
                        print(f"  Answer fact check: {'PASS' if passed else 'FAIL'}")
                        print(f"  {answer.text}")
                        for warning in answer.warnings:
                            print(f"  Warning: {warning}")
                print(f"\nExpected-source recall cases: {retrieval_passes}/4")
                if args.answers:
                    print(f"Answer fact/abstention cases: {answer_passes}/{len(cases)}")
                    print("Manually check claim support, cited pages, conflict wording and follow-ups.")
                print("These small synthetic checks do not establish real-world factual accuracy.")
            finally:
                manager.store.close()
    finally:
        client.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
