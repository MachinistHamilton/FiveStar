import hashlib
import json

from config.settings import Settings
from core.catalog import Catalog
from core.types import PageVision, VisualPageResult


def vision_configuration(settings: Settings) -> str:
    return json.dumps({
        "model": settings.vision_model,
        "context": settings.vision_context_window,
        "output": settings.vision_max_output_tokens,
        "edge": settings.vision_max_image_edge,
    }, sort_keys=True)


class CachedPageVision:
    def __init__(
        self, catalog: Catalog, document_id: str, analyzer: PageVision, configuration: str
    ):
        self.catalog = catalog
        self.document_id = document_id
        self.analyzer = analyzer
        self.model_identity = analyzer.model_identity
        self.configuration = configuration

    def analyze_page(self, image_png: bytes, filename: str, page_number: int) -> VisualPageResult:
        # Bump this version when the vision prompt or result schema changes.
        identity = [
            "visual-analysis-v1", self.document_id, page_number, self.model_identity,
            self.configuration, hashlib.sha256(image_png).hexdigest(),
        ]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        with self.catalog.connect() as db:
            row = db.execute(
                "SELECT text,warnings FROM visual_pages WHERE cache_key=?", (key,)
            ).fetchone()
        if row is not None:
            return VisualPageResult(row["text"], tuple(json.loads(row["warnings"])))
        result = self.analyzer.analyze_page(image_png, filename, page_number)
        if not result.text.strip():
            raise ValueError(f"Vision model returned no usable result for PDF page {page_number}.")
        with self.catalog.connect() as db:
            db.execute("""
                INSERT OR REPLACE INTO visual_pages
                (cache_key,document_id,page_number,model_identity,text,warnings) VALUES (?,?,?,?,?,?)
            """, (
                key, self.document_id, page_number, self.model_identity,
                result.text, json.dumps(result.warnings),
            ))
        return result

    def close(self) -> None:
        self.analyzer.close()
