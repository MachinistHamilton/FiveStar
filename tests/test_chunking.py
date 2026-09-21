import re
from dataclasses import replace

import pytest

from core.text_chunker import TextChunker
from core.types import PageText


class WordCounter:
    fingerprint = "word-counter-v1"

    def __init__(self, max_tokens=12, specials=2):
        self.max_tokens = max_tokens
        self.specials = specials

    def count_tokens(self, text):
        return len(re.findall(r"\S+", text)) + self.specials


class CharacterCounter(WordCounter):
    fingerprint = "character-counter-v1"

    def count_tokens(self, text):
        return len(text) + self.specials


def page(text, number=1, label="", document="document-id"):
    return PageText(document, "source.pdf", number, text, label)


@pytest.mark.parametrize("target", [4, 8, 650])
def test_exact_model_and_requested_token_limits_include_special_tokens(target):
    tokenizer = WordCounter(max_tokens=10, specials=3)
    chunker = TextChunker(tokenizer, target_tokens=target, overlap_tokens=0)
    text = " ".join(f"word-{index}" for index in range(100))
    chunks = list(chunker.chunk_pages([page(text)]))
    assert chunker.max_tokens == min(target, tokenizer.max_tokens)
    assert all(tokenizer.count_tokens(chunk.text) <= chunker.max_tokens for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text
    assert len(chunks) > 1


def test_paragraphs_are_preferred_to_mid_sentence_or_word_splits():
    text = "First complete paragraph.\n\nSecond paragraph has many more words to read.\n\nLast one."
    tokenizer = WordCounter(max_tokens=11)
    chunks = list(TextChunker(tokenizer, overlap_tokens=0).chunk_pages([page(text)]))
    assert [chunk.text.strip() for chunk in chunks] == [
        "First complete paragraph.",
        "Second paragraph has many more words to read.",
        "Last one.",
    ]
    assert "".join(chunk.text for chunk in chunks) == text


def test_sentences_are_preferred_inside_an_overlong_paragraph():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = list(TextChunker(WordCounter(7), overlap_tokens=0).chunk_pages([page(text)]))
    assert [chunk.text.strip() for chunk in chunks] == [
        "First sentence here.", "Second sentence here.", "Third sentence here.",
    ]
    assert "".join(chunk.text for chunk in chunks) == text


@pytest.mark.parametrize("heading", ["# Experiment", "EXPERIMENT", "2. EXPERIMENT"])
def test_heading_is_kept_with_its_following_paragraph(heading):
    text = f"Opening sentence ends here.\n\n{heading}\n\nMeasured values are reproducible."
    chunks = list(TextChunker(WordCounter(9), overlap_tokens=0).chunk_pages([page(text)]))
    assert chunks[0].text.strip() == "Opening sentence ends here."
    assert chunks[1].text.strip() == f"{heading}\n\nMeasured values are reproducible."


def test_heading_on_a_single_line_starts_a_new_section():
    text = "Opening paragraph text.\n# Results\nThese results have some more descriptive words."
    chunks = list(TextChunker(WordCounter(10), overlap_tokens=0).chunk_pages([page(text)]))
    assert chunks[0].text.strip() == "Opening paragraph text."
    assert chunks[1].text.startswith("# Results")
    assert "".join(chunk.text for chunk in chunks) == text


@pytest.mark.parametrize(
    "text",
    [
        "αβγδεζηθικλμνξοπρστυφχψω" * 5,
        "中文句子没有空格。下一句仍然保持完整！这是最后一句？" * 4,
        "E=mc²;∫₀∞e⁻ˣdx=1;∑ᵢxᵢ²≤(∑ᵢxᵢ)²;" * 5,
        "👩🏽‍🔬🙂🚀e\u0301🌍" * 12,
        "A long sentence, with formulas ∑ᵢxᵢ and Unicode 日本語, must preserve every character.",
        "First paragraph.\r\n\r\nSecond paragraph with accented résumé and naïve words.",
    ],
)
def test_overlong_unicode_sentences_and_formulas_are_lossless(text):
    tokenizer = CharacterCounter(max_tokens=17)
    chunks = list(TextChunker(tokenizer, overlap_tokens=0).chunk_pages([page(text)]))
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(tokenizer.count_tokens(chunk.text) <= 17 for chunk in chunks)
    assert all("\ufffd" not in chunk.text for chunk in chunks)


def test_overlapping_words_are_bounded_and_make_forward_progress():
    tokenizer = WordCounter(max_tokens=8)
    chunker = TextChunker(tokenizer, overlap_tokens=2)
    words = [f"w{index:03}" for index in range(75)]
    chunks = list(chunker.chunk_pages([page(" ".join(words))]))
    seen = []
    had_overlap = False
    for chunk in chunks:
        current = chunk.text.split()
        overlap = 0
        for count in range(1, min(len(seen), len(current)) + 1):
            if seen[-count:] == current[:count]:
                overlap = count
        assert overlap <= chunker.overlap_tokens
        assert len(current) > overlap
        had_overlap |= overlap > 0
        seen.extend(current[overlap:])
        assert tokenizer.count_tokens(chunk.text) <= chunker.max_tokens
    assert seen == words
    assert had_overlap
    assert len(chunks) < len(words)


@pytest.mark.parametrize("limit", [3, 4, 5])
def test_tiny_effective_limits_reduce_overlap_instead_of_looping(limit):
    tokenizer = CharacterCounter(max_tokens=limit)
    chunker = TextChunker(tokenizer, target_tokens=650, overlap_tokens=40)
    text = "abcdefghijklmno"
    chunks = list(chunker.chunk_pages([page(text)]))
    assert chunker.overlap_tokens <= limit - 3
    assert 1 <= len(chunks) <= len(text)
    offsets = [(text.index(chunk.text), text.index(chunk.text) + len(chunk.text)) for chunk in chunks]
    covered = 0
    for chunk, (start, end) in zip(chunks, offsets, strict=True):
        assert start <= covered < end
        assert covered - start <= chunker.overlap_tokens
        assert tokenizer.count_tokens(chunk.text) <= limit
        covered = end
    assert covered == len(text)


def test_tiny_limits_preserve_internal_whitespace_and_formula_spacing():
    text = "a  b\n\nc \n\n d"
    tokenizer = CharacterCounter(3)
    chunks = list(TextChunker(tokenizer, overlap_tokens=0).chunk_pages([page(text)]))
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(tokenizer.count_tokens(chunk.text) <= 3 for chunk in chunks)


def test_overlap_never_replaces_new_content_when_paragraph_is_tiny():
    text = "one.\n\ntwo three four five six seven eight nine ten."
    tokenizer = WordCounter(max_tokens=5)
    chunker = TextChunker(tokenizer, overlap_tokens=1000)
    chunks = list(chunker.chunk_pages([page(text)]))
    assert len(chunks) <= len(text.split())
    assert chunks[-1].text.rstrip().endswith("ten.")
    assert all(tokenizer.count_tokens(chunk.text) <= 5 for chunk in chunks)


@pytest.mark.parametrize("text", ["", " ", "\n\r\n\t", "\u2003\u2002"])
def test_blank_pages_produce_no_chunks(text):
    assert list(TextChunker(WordCounter()).chunk_pages([page(text)])) == []


def test_chunks_stay_on_physical_pages_and_preserve_labels_and_metadata():
    pages = [
        page("first page has text", 1, "iv"),
        page("", 2, "v"),
        page("third page has text", 3, "A-1"),
    ]
    chunks = list(TextChunker(WordCounter(), overlap_tokens=0).chunk_pages(iter(pages)))
    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert [chunk.page_end for chunk in chunks] == [1, 3]
    assert [chunk.page_label for chunk in chunks] == ["iv", "A-1"]
    assert all(chunk.filename == "source.pdf" and chunk.document_id == "document-id" for chunk in chunks)


def test_page_input_is_consumed_lazily():
    visited = []

    def pages():
        visited.append(1)
        yield page("first page")
        visited.append(2)
        yield page("second page", 2)

    chunks = TextChunker(WordCounter()).chunk_pages(pages())
    assert visited == []
    next(chunks)
    assert visited == [1]
    list(chunks)
    assert visited == [1, 2]


def test_stable_ids_include_document_page_content_and_position():
    chunker = TextChunker(WordCounter(4), overlap_tokens=0)
    pages = [page("same text same text same text", 1, "i"), page("same text", 2, "ii")]
    first = list(chunker.chunk_pages(pages))
    assert first == list(chunker.chunk_pages(pages))
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    for changed in [
        replace(pages[0], document_id="another-document"),
        replace(pages[0], page_number=3),
        replace(pages[0], text="new words"),
        replace(pages[0], page_label="new-label"),
    ]:
        changed_ids = {chunk.chunk_id for chunk in chunker.chunk_pages([changed])}
        assert changed_ids.isdisjoint(chunk.chunk_id for chunk in first)


def test_fingerprint_is_stable_and_reflects_effective_settings():
    tokenizer = WordCounter(12)
    first = TextChunker(tokenizer, target_tokens=650, overlap_tokens=2)
    assert first.fingerprint == TextChunker(WordCounter(12), 12, 2).fingerprint
    assert first.fingerprint != TextChunker(WordCounter(12), 10, 2).fingerprint
    assert first.fingerprint != TextChunker(WordCounter(12), 12, 3).fingerprint
    tokenizer.fingerprint = "another-tokenizer"
    assert first.fingerprint != TextChunker(tokenizer, 12, 2).fingerprint


@pytest.mark.parametrize(
    ("max_tokens", "target", "overlap"), [(0, 10, 0), (12, 0, 0), (12, 10, -1), (2, 10, 0), (12, 2, 0)]
)
def test_invalid_limits_fail_with_actionable_errors(max_tokens, target, overlap):
    with pytest.raises(ValueError, match="(?i)(limit|overlap)"):
        TextChunker(WordCounter(max_tokens), target, overlap)


def test_unrepresentable_character_is_not_silently_discarded():
    class ByteCounter(WordCounter):
        def count_tokens(self, text):
            return len(text.encode("utf-8")) + self.specials

    chunker = TextChunker(ByteCounter(3), overlap_tokens=0)
    with pytest.raises(ValueError, match="Increase chunk/model limits"):
        list(chunker.chunk_pages([page("界")]))


def test_boundary_token_counts_are_rechecked_for_nonmonotonic_tokenizers():
    class BoundaryCounter(CharacterCounter):
        def count_tokens(self, text):
            return super().count_tokens(text) + (30 if text.endswith("\n\n") else 0)

    tokenizer = BoundaryCounter(14)
    text = "abc\n\ndefghijklmno"
    chunks = list(TextChunker(tokenizer, overlap_tokens=0).chunk_pages([page(text)]))
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(tokenizer.count_tokens(chunk.text) <= tokenizer.max_tokens for chunk in chunks)
