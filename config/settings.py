from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore", frozen=True
    )

    data_dir: Path = PROJECT_ROOT / "storage"
    model_cache: Path = PROJECT_ROOT / "models"
    offline: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "main"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "main"
    embedding_device: str = "cpu"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    context_window: int = Field(default=8192, ge=2048, le=131072)
    max_output_tokens: int = Field(default=1024, ge=128, le=8192)
    temperature: float = Field(default=0.1, ge=0, le=1)
    inference_threads: int = Field(default=2, ge=1, le=32)
    candidate_count: int = Field(default=20, ge=1, le=100)
    evidence_count: int = Field(default=6, ge=1, le=20)
    rerank_enabled: bool = True
    verify_claims: bool = True
    chunk_tokens: int = Field(default=650, ge=32, le=8192)
    chunk_overlap: int = Field(default=40, ge=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    max_upload_mb: int = Field(default=200, ge=1, le=2000)
    ocr_language: str = "eng"
    tessdata_prefix: str = ""
    vision_enabled: bool = False
    vision_model: str = "qwen3-vl:4b-instruct"
    vision_context_window: int = Field(default=16384, ge=8192, le=32768)
    vision_max_output_tokens: int = Field(default=2048, ge=512, le=4096)
    vision_max_image_edge: int = Field(default=1536, ge=768, le=2048)
    vision_timeout_seconds: int = Field(default=600, ge=60, le=1800)

    @field_validator("data_dir", "model_cache")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("ollama_url")
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        url = urlparse(value)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("Ollama must use an HTTP loopback URL, e.g. http://127.0.0.1:11434")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        if self.max_output_tokens + 1024 >= self.context_window:
            raise ValueError("Context window must leave more than 1024 tokens beyond output.")
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("Chunk overlap must be smaller than the requested chunk size.")
        if self.evidence_count > self.candidate_count:
            raise ValueError("Evidence count must not exceed candidate count.")
        return self
