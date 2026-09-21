import hashlib
from collections.abc import Callable

import pymupdf

from config.settings import Settings
from core.catalog import Catalog
from core.errors import IndexError, PDFError
from core.library_lock import LibraryLock
from core.pdf_processor import render_page_png
from core.types import PageVision, VisualPageResult
from core.visual_cache import CachedPageVision, vision_configuration


def analyze_saved_page(
    settings: Settings, document_id: str, page_number: int,
    *, analyzer_factory: Callable[[], PageVision] | None = None,
) -> VisualPageResult:
    catalog = Catalog(settings.data_dir)
    with LibraryLock(settings.data_dir):
        document = catalog.get_document(document_id)
        if document["deleting"]:
            raise IndexError("This document is being deleted; refresh the library.")
        path = catalog.document_path(document_id)
        with path.open("rb") as original:
            if hashlib.file_digest(original, "sha256").hexdigest() != document_id:
                raise PDFError("The saved PDF has changed. Upload it again before analyzing images.")
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass or pdf.is_encrypted:
                raise PDFError("An encrypted PDF must be saved as an authorized unencrypted copy.")
            if type(page_number) is not int or not 1 <= page_number <= len(pdf):
                raise PDFError("Select a valid physical PDF page for image analysis.")
            image = render_page_png(pdf[page_number - 1], settings.vision_max_image_edge)
        if analyzer_factory is None:
            from core.vision import VisionAnalyzer
            analyzer = VisionAnalyzer(settings)
        else:
            analyzer = analyzer_factory()
        try:
            cached = CachedPageVision(
                catalog, document_id, analyzer, vision_configuration(settings)
            )
            return cached.analyze_page(image, document["filename"], page_number)
        finally:
            analyzer.close()
