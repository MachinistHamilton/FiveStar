from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class PageText:
    document_id: str
    filename: str
    page_number: int
    text: str
    page_label: str = ""
    warnings: tuple[str, ...] = ()
    source_type: str = "text"
    source_model: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    page_end: int
    text: str
    page_label: str = ""
    source_type: str = "text"
    source_model: str = ""


@dataclass(frozen=True)
class VisualPageResult:
    text: str
    warnings: tuple[str, ...] = ()


class PageVision(Protocol):
    model_identity: str

    def analyze_page(
        self, image_png: bytes, filename: str, page_number: int
    ) -> VisualPageResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    semantic_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    exact_match: bool = False


@dataclass(frozen=True)
class Source:
    source_id: int
    hit: SearchHit


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    query: str = ""


@dataclass(frozen=True)
class Draft:
    request: str
    content: str
    kind: Literal["code", "text"]
    basis: Answer
    assumptions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


class TokenCounter(Protocol):
    max_tokens: int
    fingerprint: str

    def count_tokens(self, text: str) -> int: ...


class Embedder(TokenCounter, Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Chunker(Protocol):
    fingerprint: str

    def chunk_pages(self, pages: Iterable[PageText]) -> Iterable[Chunk]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, document_ids: list[str] | None = None) -> list[SearchHit]: ...


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str: ...
