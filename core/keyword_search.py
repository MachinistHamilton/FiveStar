import re

from rank_bm25 import BM25Plus

from core.types import Chunk

# Keep domain words and identifiers; omit common question scaffolding from BM25.
_QUESTION_WORDS = frozenset(
    "a an the and or of to in on at by for from with as is are was were be been "
    "what which who where when why how do does did can could would should i you "
    "we it its this that these those my me please tell about".split()
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+(?:[-./+^]\w+)*", text.casefold())


def exact_terms(query: str) -> list[str]:
    identifiers = [
        token for token in tokenize(query)
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token)
    ]
    phrases = re.findall(r'"([^"]+)"', query.casefold())
    return list(dict.fromkeys(identifiers + phrases))


def has_exact_match(text: str, terms: list[str]) -> bool:
    normalized = " ".join(text.casefold().split())
    return any(
        re.search(r"(?<!\w)" + re.escape(" ".join(term.split())) + r"(?!\w)", normalized)
        for term in terms
    )


def document_is_named(filename: str, query: str) -> bool:
    title = " ".join(re.findall(r"[^\W_]+", filename.rsplit(".", 1)[0].casefold()))
    normalized_query = " ".join(re.findall(r"[^\W_]+", query.casefold()))
    return len(title) >= 4 and bool(
        re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", normalized_query)
    )


class KeywordSearch:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [set(tokenize(chunk.text)) for chunk in chunks]
        corpus = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Plus(corpus) if any(corpus) else None

    def search(
        self, query: str, limit: int, document_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        query_tokens = [token for token in tokenize(query) if token not in _QUESTION_WORDS]
        token_set = set(query_tokens)
        terms = exact_terms(query)
        scores = self.index.get_scores(query_tokens)
        matches = [
            (
                chunk.chunk_id, float(scores[i]),
                has_exact_match(chunk.text, terms) or document_is_named(chunk.filename, query),
            )
            for i, chunk in enumerate(self.chunks)
            if (document_ids is None or chunk.document_id in document_ids)
            and (
                token_set & self.tokens[i] or has_exact_match(chunk.text, terms)
                or document_is_named(chunk.filename, query)
            )
        ]
        matches.sort(key=lambda item: (item[2], item[1], item[0]), reverse=True)
        return [(chunk_id, score) for chunk_id, score, _ in matches[:limit]]
