import os

from config.settings import Settings
from core.embeddings import load_embedding_model
from core.reranker import load_reranker


def main() -> None:
    # This explicit setup command is the only workflow that requires Internet access.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    settings = Settings()
    print(f"Downloading embedding weights to {settings.model_cache} ...")
    load_embedding_model(
        settings.embedding_model, settings.embedding_revision,
        str(settings.model_cache), settings.embedding_device, False,
    )
    print("Downloading reranker weights ...")
    load_reranker(
        settings.reranker_model, settings.reranker_revision,
        str(settings.model_cache), settings.embedding_device, False,
    )
    print("Models are cached. Leave OFFLINE=true for normal use.")
    print(f"Also run: ollama pull {settings.ollama_model}")


if __name__ == "__main__":
    main()
