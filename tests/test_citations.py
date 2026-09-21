import json
from dataclasses import replace

import pytest

from core.citations import (
    MISSING_INFORMATION,
    Claim,
    escape_markdown,
    load_json_object,
    render_claims,
    validate_answer,
    validate_sources,
    validate_support,
)
from core.errors import GenerationError
from core.types import Chunk, SearchHit, Source


def source(identifier=1, **changes):
    chunk = Chunk(
        chunk_id="chunk", document_id="document", filename="notes.pdf",
        page_number=3, page_end=4, text="Document fact.", page_label="A-99",
    )
    return Source(identifier, SearchHit(replace(chunk, **changes), score=0.9))


def answer(claims, sufficient=True):
    return json.dumps({"sufficient": sufficient, "claims": claims})


def test_valid_claims_retain_only_verified_ids_and_deduplicate_references():
    result = validate_answer(answer([{"text": "A supported fact.", "sources": [2, 2, 1]}]), [source(), source(2)])
    assert result.sufficient
    assert result.claims == [Claim("A supported fact.", (2, 1))]
    assert not result.warnings


@pytest.mark.parametrize("claim", [
    None, [], "fact", {}, {"text": "fact", "sources": [1], "page": 99},
    {"text": 7, "sources": [1]}, {"text": "", "sources": [1]},
    {"text": " \n ", "sources": [1]}, {"text": "fact", "sources": []},
    {"text": "fact", "sources": "1"}, {"text": "fact", "sources": [0]},
    {"text": "fact", "sources": [-1]}, {"text": "fact", "sources": [2]},
    {"text": "fact", "sources": ["1"]}, {"text": "fact", "sources": [True]},
    {"text": "fact", "sources": [1.0]}, {"text": "fact", "sources": [1, 2]},
    {"text": "fact", "sources": [{}]}, {"text": "\ud800", "sources": [1]},
])
def test_invalid_uncited_claims_removed_with_explicit_warnings(claim):
    result = validate_answer(answer([claim]), [source()])
    assert not result.claims
    assert any("Removed claim 1" in warning for warning in result.warnings)


@pytest.mark.parametrize("text", [
    "The result is seven [1].", "The result [source A] is seven.",
    "The result is seven (page 99).", "See pp. 4-6 for details.",
    "The result is on page: 99.", "The result is on page number 99.",
    "The result is on page **99**.", "The result is on pa\u200bge 99.",
    "The result is on page <b>99</b>.", "The result is on page nine.",
    "The result is on p. IX.", "The result is in secret.pdf.",
    "The result is seven \u3010source 1\u3011.", "The result is seven ［1］.",
    "The result is seven &#91;1&#93;.", "Source: 999 supports this.",
])
def test_model_citations_and_page_claims_cannot_be_rendered_as_verified(text):
    result = validate_answer(answer([{"text": text, "sources": [1]}]), [source()])
    assert not result.claims
    assert "not verified" in result.warnings[0]
    with pytest.raises(GenerationError, match="verified citations"):
        render_claims([Claim(text, (1,))], [source()])


def test_invalid_claim_does_not_hide_valid_claim_or_warning():
    result = validate_answer(answer([
        {"text": "Invented", "sources": [77]},
        {"text": "Supported", "sources": [1]},
    ]), [source()])
    assert result.claims == [Claim("Supported", (1,))]
    assert "Removed claim 1" in result.warnings[0]


def test_insufficient_disregards_claims():
    result = validate_answer(answer([{"text": "Must not appear", "sources": [1]}], sufficient=False), [source()])
    assert result.sufficient is False
    assert not result.claims
    assert render_claims([], []) == MISSING_INFORMATION


@pytest.mark.parametrize("raw", [
    "not json", "```json\n{}\n```", "[]", "null", '{"a":NaN}', '{"a":Infinity}',
    '{"sufficient":true,"sufficient":false,"claims":[]}',
])
def test_non_strict_json_is_rejected(raw):
    with pytest.raises(GenerationError):
        load_json_object(raw)


@pytest.mark.parametrize("payload", [
    {}, {"sufficient": 1, "claims": []}, {"sufficient": "true", "claims": []},
    {"sufficient": True, "claims": {}},
    {"sufficient": True, "claims": [], "answer": "uncited"},
])
def test_invalid_answer_envelopes_are_errors(payload):
    with pytest.raises(GenerationError):
        validate_answer(json.dumps(payload), [source()])


def test_rendered_references_use_physical_pages_only_and_preserve_ids():
    text = render_claims([Claim("Document fact.", (7, 1))], [
        source(1, page_number=2, page_end=2), source(7),
    ])
    assert "[7: notes\\.pdf, physical PDF pages 3\u20134]" in text
    assert "[1: notes\\.pdf, physical PDF page 2]" in text
    assert "A-99" not in text
    assert "Document fact\\." in text


def test_markdown_and_html_from_filenames_and_claims_are_inert():
    malicious_name = '<img src=x> [click](https://evil.example)/a.pdf\n# header'
    text = render_claims([Claim("<script>alert('x')</script> *bold* https://evil.example", (1,))], [
        source(filename=malicious_name),
    ])
    assert "<img" not in text and "<script" not in text
    assert "[click](" not in text
    assert "&lt;img" in text and "&lt;script&gt;" in text
    assert "\\[click\\]\\(https\\://" in text
    assert "\\*bold\\*" in text
    assert "\n# header" not in text
    assert "\u202e" not in escape_markdown("name\u202egpj.pdf")


@pytest.mark.parametrize("changes", [
    {"page_number": 0}, {"page_number": True}, {"page_end": 2},
    {"page_end": "4"}, {"filename": ""}, {"filename": None},
])
def test_bad_metadata_fails_closed(changes):
    with pytest.raises(GenerationError, match="metadata"):
        validate_sources([source(**changes)])


@pytest.mark.parametrize("identifiers", [[1, 1], [True], [0], [-1], ["1"]])
def test_invalid_metadata_source_ids_fail_closed(identifiers):
    with pytest.raises(GenerationError, match="metadata"):
        validate_sources([source(identifier) for identifier in identifiers])


def test_rendering_unvalidated_claim_with_bad_source_fails():
    with pytest.raises(GenerationError):
        render_claims([Claim("fact", (999,))], [source()])


def test_verification_requires_every_claim_and_accepts_reordered_decisions():
    raw = json.dumps({"support": [
        {"claim_id": 2, "supported": False}, {"claim_id": 1, "supported": True},
    ]})
    assert validate_support(raw, 2) == [True, False]


@pytest.mark.parametrize("payload", [
    {"support": []}, {"support": [True]}, {"support": [{"claim_id": 1, "supported": "yes"}]},
    {"support": [{"claim_id": True, "supported": True}]},
    {"support": [{"claim_id": 0, "supported": True}]},
    {"support": [{"claim_id": 2, "supported": True}]},
    {"support": [{"claim_id": 1, "supported": 1}]},
    {"support": [{"claim_id": 1, "supported": True, "reason": "extra"}]},
    {"support": [{"claim_id": 1, "supported": True}, {"claim_id": 1, "supported": False}]},
    {"support": [{"claim_id": 1, "supported": True}], "extra": 1},
])
def test_invalid_support_is_an_error_not_success(payload):
    with pytest.raises(GenerationError):
        validate_support(json.dumps(payload), 1)
