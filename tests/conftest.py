import hashlib
import math
import re

import pymupdf
import pytest

from config.settings import Settings


class FakeEmbedder:
    max_tokens = 128
    fingerprint = "test-embedding-v1"

    def __init__(self):
        self.batches: list[int] = []

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\S+", text)) + 2

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        result = []
        for text in texts:
            vector = [0.0] * 32
            for word in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % len(vector)
                vector[bucket] += 1
            length = math.sqrt(sum(x * x for x in vector)) or 1
            result.append([x / length for x in vector])
        return result


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, data_dir=tmp_path / "data", model_cache=tmp_path / "models",
        rerank_enabled=False, verify_claims=False,
    )


@pytest.fixture
def pdf_bytes():
    def make(*pages: str) -> bytes:
        with pymupdf.open() as pdf:
            for text in pages:
                page = pdf.new_page()
                page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=11)
            return pdf.tobytes()
    return make
