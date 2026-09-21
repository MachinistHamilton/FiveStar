"""Strict claim validation and application-owned provenance rendering.

MISSING_INFORMATION is the shared, exact abstention response. Citation validation
checks provenance, not truth; even a second model support check is only heuristic.
"""

import html
import json
import re
import unicodedata
from dataclasses import dataclass, field

from core.errors import GenerationError
from core.types import Source

MISSING_INFORMATION = (
    "I could not find sufficient information in the uploaded documents to answer this question."
)

CLAIMS_SCHEMA = {
    "type": "array",
    "maxItems": 12,
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "sources": {
                "type": "array", "minItems": 1,
                "items": {"type": "integer", "minimum": 1},
            },
        },
        "required": ["text", "sources"],
        "additionalProperties": False,
    },
}

_RAW_REFERENCE = re.compile(
    r"(?:\[[^\]]+\]|【[^】]*】|"
    r"\b(?:source|citation|reference)\s*[:#]?\s*\d|"
    r"\b(?:pages?|pp?|pgs?)\.?\s*(?:number\s*)?[:#]?\s*(?:\d+|[ivxlcdm]+|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b|"
    r"\S+\.pdf\b)",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|~$:@])")


def _has_raw_reference(text: str) -> bool:
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"[*_`]", "", text)
    return bool(_RAW_REFERENCE.search(text))


@dataclass(frozen=True)
class Claim:
    text: str
    sources: tuple[int, ...]


@dataclass
class ValidatedAnswer:
    sufficient: bool
    claims: list[Claim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_json_object(raw: str) -> dict:
    """Reject fences, duplicate keys, non-finite numbers, and non-object roots."""
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise GenerationError("The model returned malformed structured JSON. Retry with a narrower question.") from exc
    if not isinstance(value, dict):
        raise GenerationError("The model must return a structured JSON object.")
    return value


def validate_sources(sources: list[Source]) -> dict[int, Source]:
    """Validate immutable index metadata, never model-written filenames/page labels."""
    result = {}
    for source in sources:
        chunk = source.hit.chunk
        if (
            type(source.source_id) is not int
            or source.source_id <= 0
            or source.source_id in result
            or not isinstance(chunk.filename, str)
            or not chunk.filename.strip()
            or type(chunk.page_number) is not int
            or type(chunk.page_end) is not int
            or chunk.page_number < 1
            or chunk.page_end < chunk.page_number
            or chunk.source_type not in {"text", "visual"}
            or (
                chunk.source_type == "visual"
                and (not isinstance(chunk.source_model, str) or not chunk.source_model.strip())
            )
        ):
            raise GenerationError("Source metadata has invalid IDs, filenames, or physical PDF pages. Reindex the affected PDF.")
        result[source.source_id] = source
    return result


def validate_answer(raw: str, sources: list[Source]) -> ValidatedAnswer:
    """Remove each invalid claim with an explicit warning; malformed envelopes fail."""
    available = validate_sources(sources)
    payload = load_json_object(raw)
    if (
        set(payload) != {"sufficient", "claims"}
        or type(payload["sufficient"]) is not bool
        or not isinstance(payload["claims"], list)
    ):
        raise GenerationError("The model answer must contain only sufficient (boolean) and claims (list).")
    result = ValidatedAnswer(sufficient=payload["sufficient"])
    if not result.sufficient:
        return result
    for index, item in enumerate(payload["claims"], 1):
        reason = ""
        if not isinstance(item, dict) or set(item) != {"text", "sources"}:
            reason = "invalid claim structure"
        elif not isinstance(item["text"], str) or not item["text"].strip():
            reason = "empty or non-text claim"
        elif not isinstance(item["sources"], list) or not item["sources"]:
            reason = "missing source citations"
        elif any(type(identifier) is not int or identifier not in available for identifier in item["sources"]):
            reason = "invalid or unselected source IDs"
        elif _has_raw_reference(item["text"]):
            reason = "model-written citations, filenames, or page references are not verified"
        elif any(unicodedata.category(char) == "Cs" for char in item["text"]):
            reason = "invalid Unicode text"
        if reason:
            result.warnings.append(f"Removed claim {index}: {reason}.")
        else:
            result.claims.append(Claim(item["text"].strip(), tuple(dict.fromkeys(item["sources"]))))
    if not result.claims:
        result.warnings.append("No valid cited claims remained in the model answer.")
    return result


def validate_support(raw: str, claim_count: int) -> list[bool]:
    """Parse {support: [{claim_id: 1, supported: true}, ...]} with complete coverage."""
    payload = load_json_object(raw)
    if set(payload) != {"support"} or not isinstance(payload["support"], list):
        raise GenerationError("Claim verification returned an invalid support list.")
    decisions = {}
    for item in payload["support"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"claim_id", "supported"}
            or type(item["claim_id"]) is not int
            or not 1 <= item["claim_id"] <= claim_count
            or item["claim_id"] in decisions
            or type(item["supported"]) is not bool
        ):
            raise GenerationError("Claim verification returned invalid or duplicate support decisions.")
        decisions[item["claim_id"]] = item["supported"]
    if set(decisions) != set(range(1, claim_count + 1)):
        raise GenerationError("Claim verification omitted one or more support decisions.")
    return [decisions[index] for index in range(1, claim_count + 1)]


def escape_markdown(text: str) -> str:
    """Render source/model text as inert single-line text, not links or raw HTML."""
    text = "".join(char for char in text if unicodedata.category(char) not in {"Cf", "Cs"})
    text = " ".join(text.split())
    return _MARKDOWN.sub(r"\\\1", html.escape(text, quote=False))


def render_claims(claims: list[Claim], sources: list[Source]) -> str:
    """Only this function creates citations, using physical pages from the index."""
    available = validate_sources(sources)
    lines = []
    for claim in claims:
        if not claim.sources or any(type(identifier) is not int or identifier not in available for identifier in claim.sources):
            raise GenerationError("Cannot render a claim without valid selected sources.")
        if _has_raw_reference(claim.text):
            raise GenerationError("Cannot render model-written page references as verified citations.")
        references = []
        has_visual_source = False
        for identifier in dict.fromkeys(claim.sources):
            chunk = available[identifier].hit.chunk
            has_visual_source |= chunk.source_type == "visual"
            pages = (
                f"page {chunk.page_number}"
                if chunk.page_number == chunk.page_end
                else f"pages {chunk.page_number}\u2013{chunk.page_end}"
            )
            references.append(
                f"[{identifier}: {escape_markdown(chunk.filename)}, physical PDF {pages}"
                + ("; visual interpretation]" if chunk.source_type == "visual" else "]")
            )
        prefix = "Visual interpretation (verify against the original): " if has_visual_source else ""
        lines.append(f"- {prefix}{escape_markdown(claim.text)} {' '.join(references)}")
    return "\n\n".join(lines) if lines else MISSING_INFORMATION
