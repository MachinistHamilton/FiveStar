import hashlib
import json
from functools import lru_cache

from config.settings import Settings
from core.errors import ModelError


@lru_cache(maxsize=2)
def load_embedding_model(name: str, revision: str, cache: str, device: str, offline: bool):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            name,
            revision=revision,
            cache_folder=cache,
            device=device,
            local_files_only=offline,
            trust_remote_code=False,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelError(
            f"Cannot load embedding model {name}. Run "
            "'python -m scripts.download_models' while online, then retry. "
            f"Details: {exc}"
        ) from exc


class Embeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = load_embedding_model(
            settings.embedding_model,
            settings.embedding_revision,
            str(settings.model_cache),
            settings.embedding_device,
            settings.offline,
        )
        self.max_tokens = int(self.model.max_seq_length)
        config = self.model[0].auto_model.config
        resolved_revision = getattr(config, "_commit_hash", None)
        if not resolved_revision:
            raise ModelError(
                "The embedding model must be a versioned Hugging Face repository model. "
                "A resolved revision is required to prevent mixing incompatible embeddings."
            )
        identity = {
            "model": settings.embedding_model,
            "revision": resolved_revision,
            "max_tokens": self.max_tokens,
            "dimension": self.model.get_embedding_dimension(),
            "normalize": True,
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(
            text, add_special_tokens=True, truncation=False, verbose=False
        ))

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(self.count_tokens(text) > self.max_tokens for text in texts):
            raise ModelError(
                f"Embedding input exceeds {self.max_tokens} tokens. Reindex with smaller chunks."
            )
        return self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
