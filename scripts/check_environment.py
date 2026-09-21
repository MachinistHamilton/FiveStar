import argparse
import importlib.metadata
import shutil
import sys

from config.settings import Settings
from core.errors import AssistantError
from core.llm import OllamaClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local research assistant prerequisites.")
    parser.add_argument("--models", action="store_true", help="Also load cached embedding/reranker.")
    parser.add_argument("--vision", action="store_true", help="Check the installed local vision model.")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):  # noqa: UP036 - this is an environment diagnostic
        print("ERROR: Python 3.11 or newer is required.")
        return 1
    for name in ("streamlit", "chromadb", "sentence-transformers", "PyMuPDF"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"ERROR: {name} is missing. Run pip install -r requirements.txt")
            return 1
    settings = Settings()
    print(f"Documents/indexes: {settings.data_dir}")
    print(f"Model cache: {settings.model_cache} | Offline: {settings.offline}")
    print(f"Tesseract command: {shutil.which('tesseract') or 'not on PATH (OCR is optional)'}")
    print(f"Tessdata override: {settings.tessdata_prefix or 'auto-discovery'}")
    ok = True
    if args.models:
        from core.embeddings import Embeddings
        from core.reranker import load_reranker
        try:
            embedder = Embeddings(settings)
            print(f"Embedding input limit: {embedder.max_tokens} tokens")
            print(f"Test embedding dimensions: {len(embedder.encode(['Local test.'])[0])}")
            if settings.rerank_enabled:
                load_reranker(
                    settings.reranker_model, settings.reranker_revision,
                    str(settings.model_cache), settings.embedding_device, settings.offline,
                )
                print("Reranker loaded.")
        except AssistantError as exc:
            print(f"ERROR: {exc}")
            ok = False
    client = OllamaClient(settings)
    try:
        client.check_available()
        print(f"Ollama ready: {settings.ollama_model}")
    except AssistantError as exc:
        print(f"ERROR: {exc}")
        ok = False
    finally:
        client.close()
    if args.vision or settings.vision_enabled:
        from core.vision import VisionAnalyzer
        vision = None
        try:
            vision = VisionAnalyzer(settings)
            vision.check_available()
            print(f"Local vision ready: {vision.model_identity}")
        except AssistantError as exc:
            print(f"ERROR: {exc}")
            ok = False
        finally:
            if vision is not None:
                vision.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
