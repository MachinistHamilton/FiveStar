"""Page-local chunking with a model-free byte budget or a real embedding tokenizer."""

import hashlib
import json
import re
from bisect import bisect_right
from collections.abc import Iterable, Iterator

from core.types import Chunk, PageText, TokenCounter

_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)(?:[ \t]*\r?\n)*")
_SENTENCE_END = re.compile(r"""(?:[.!?]["'”’)\]]*(?:\s+|$)|[。！？]["'”’)\]]*\s*)""")
_WHITESPACE = re.compile(r"\s+")
_HEADING = re.compile(r"^(?:#{1,6}\s+\S.*|(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Z \d:–—-]*)$")


class LocalTextCounter:
    """Bound keyword-only passages by UTF-8 bytes without loading a tokenizer model."""

    max_tokens = 2400
    fingerprint = "local-utf8-bytes-v1"

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))


def keyword_chunker() -> "TextChunker":
    return TextChunker(LocalTextCounter(), target_tokens=2400, overlap_tokens=160)


class TextChunker:
    """Preserve page provenance and text, preferring paragraphs before sentences.

    Both ``target_tokens`` and ``max_tokens`` include tokenizer special tokens.
    Overlap is a content-token budget, reduced when necessary to make progress.
    Leading/trailing page whitespace is ignored; interior characters are never
    normalized or discarded, even when a tiny limit splits a whitespace run.
    An unrepresentable Unicode code point raises ValueError rather than being
    discarded when the configured limit is too small.
    """

    def __init__(
        self, tokenizer: TokenCounter, target_tokens: int = 650, overlap_tokens: int = 40
    ):
        if target_tokens < 1 or tokenizer.max_tokens < 1:
            raise ValueError("Chunk and model token limits must be positive.")
        if overlap_tokens < 0:
            raise ValueError("Chunk overlap must not be negative.")
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.max_tokens = min(target_tokens, tokenizer.max_tokens)
        self._special_tokens = tokenizer.count_tokens("")
        if self.max_tokens <= self._special_tokens:
            raise ValueError(
                "The chunk token limit must leave room for text after tokenizer special tokens."
            )
        self.overlap_tokens = min(
            overlap_tokens, self.max_tokens - self._special_tokens - 1
        )
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "algorithm": "page-paragraph-sentence-v1",
                    "tokenizer": tokenizer.fingerprint,
                    "max_tokens": self.max_tokens,
                    "overlap_tokens": self.overlap_tokens,
                    "special_tokens": self._special_tokens,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _boundaries(text: str) -> tuple[list[int], list[int], list[int]]:
        paragraphs = []
        paragraph_start = 0
        for match in _PARAGRAPH_BREAK.finditer(text):
            paragraph = text[paragraph_start:match.start()].strip()
            # Keep explicit headings with their following paragraph where possible.
            if "\n" in paragraph or not _HEADING.fullmatch(paragraph):
                paragraphs.append(match.end())
            paragraph_start = match.end()
        for match in re.finditer(r"(?m)^.+$", text):
            if match.start() and _HEADING.fullmatch(match.group().strip()):
                paragraphs.append(match.start())
        return (
            sorted(set(paragraphs)),
            [match.end() for match in _SENTENCE_END.finditer(text)],
            [match.end() for match in _WHITESPACE.finditer(text)],
        )

    def _fitting_end(self, text: str, start: int) -> int:
        """Find a measured fitting prefix; never estimate token counts for output."""
        low = start
        high = min(len(text), start + self.max_tokens * 4)
        while self.tokenizer.count_tokens(text[start:high]) <= self.max_tokens:
            low = high
            if high == len(text):
                return high
            high = min(len(text), start + (high - start) * 2)
        while high - low > 1:
            middle = (low + high) // 2
            if self.tokenizer.count_tokens(text[start:middle]) <= self.max_tokens:
                low = middle
            else:
                high = middle
        return low

    def _overlap_start(self, text: str, start: int, end: int, words: list[int]) -> int:
        if not self.overlap_tokens:
            return end
        low, high = start + 1, end
        while low < high:
            middle = (low + high) // 2
            content_tokens = self.tokenizer.count_tokens(text[middle:end]) - self._special_tokens
            if content_tokens <= self.overlap_tokens:
                high = middle
            else:
                low = middle + 1
        # Prefer whole words; a code-point suffix is still necessary for long formulas/CJK.
        word_index = bisect_right(words, low - 1)
        if word_index < len(words) and words[word_index] < end:
            candidate = words[word_index]
            if (
                self.tokenizer.count_tokens(text[candidate:end]) - self._special_tokens
                <= self.overlap_tokens
            ):
                low = candidate
        if (
            self.tokenizer.count_tokens(text[low:end]) - self._special_tokens
            > self.overlap_tokens
        ):
            return end
        return low

    def chunk_pages(self, pages: Iterable[PageText]) -> Iterator[Chunk]:
        for page in pages:
            text = page.text.strip()
            if not text:
                continue
            boundaries = self._boundaries(text)
            start = covered = 0
            while covered < len(text):
                end = self._fitting_end(text, start)
                if end <= covered and start < covered:
                    # Discard overlap rather than outputting a repeated-only chunk.
                    start = covered
                    end = self._fitting_end(text, start)
                if end <= covered:
                    raise ValueError(
                        f"Token limit {self.max_tokens} cannot represent text on page "
                        f"{page.page_number} at character {covered}. Increase chunk/model limits."
                    )
                if end < len(text):
                    for candidates in boundaries:
                        index = bisect_right(candidates, end) - 1
                        if index >= 0 and candidates[index] > covered:
                            candidate = candidates[index]
                            # Tokenizers can be non-monotonic at a new boundary.
                            if (
                                self.tokenizer.count_tokens(text[start:candidate])
                                <= self.max_tokens
                            ):
                                end = candidate
                                break
                chunk_text = text[start:end]
                identity = [
                    self.fingerprint, page.document_id, page.page_number,
                    page.page_label, start, end, chunk_text,
                ]
                if page.source_type != "text":
                    identity.extend([page.source_type, page.source_model])
                chunk_id = hashlib.sha256(
                    json.dumps(identity, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                yield Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    page_end=page.page_number,
                    text=chunk_text,
                    page_label=page.page_label,
                    source_type=page.source_type,
                    source_model=page.source_model,
                )
                covered = end
                start = self._overlap_start(text, start, end, boundaries[2])
