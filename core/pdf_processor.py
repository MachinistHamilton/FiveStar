"""Validated, page-local PDF extraction with optional, explicitly reported OCR."""

import hashlib
import ntpath
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pymupdf

from core.errors import PDFError
from core.types import PageText, PageVision, Progress

_LOW_TEXT_CHARACTERS = 40
_DEVICE_NAME = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$", re.I)
_PDF_ERRORS = (pymupdf.FileDataError, pymupdf.mupdf.FzErrorBase)


def render_page_png(page: pymupdf.Page, max_edge: int) -> bytes:
    scale = min(2.0, (max_edge - 1) / max(page.rect.width, page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")


def safe_filename(name: str) -> str:
    """Return a portable PDF basename, never a caller-supplied directory path."""
    basename = ntpath.basename(name)
    basename = "".join(
        character for character in basename
        if unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    ).strip()
    if not basename.lower().endswith(".pdf") or not basename[:-4].strip(" ."):
        raise PDFError("Choose a file with a .pdf extension and a nonempty filename.")
    stem = re.sub(r'[<>:"|?*]', "_", basename[:-4]).strip(" .")
    if _DEVICE_NAME.fullmatch(stem.split(".")[0].rstrip()):
        stem = "_" + stem
    # Stay within common filesystem byte limits, without cutting a Unicode code point.
    stem = stem.encode("utf-8")[:240].decode("utf-8", errors="ignore").rstrip(" .")
    return stem + basename[-4:]


def content_hash(data: bytes) -> str:
    """Identify an upload by its complete original bytes."""
    return hashlib.sha256(data).hexdigest()


def _ocr_instructions(language: str) -> str:
    return (
        f"Install Tesseract with the '{language}' language data, set TESSDATA_PREFIX "
        "(or the tessdata_prefix setting) to its tessdata directory, then enable OCR "
        "and reindex."
    )


def _ocr_text(
    page: pymupdf.Page, original: str, language: str, tessdata: str
) -> tuple[str, str]:
    instructions = _ocr_instructions(language)
    get_ocr = getattr(page, "get_textpage_ocr", None)
    if get_ocr is None:
        return original, f"OCR is unavailable in this PyMuPDF installation. {instructions}"
    try:
        textpage = get_ocr(language=language, full=True, tessdata=tessdata or None)
    except (RuntimeError, OSError, pymupdf.mupdf.FzErrorBase) as exc:
        # Only engine/data availability errors are recoverable. Programming errors
        # and unrelated failures must not masquerade as successful extraction.
        if not any(
            word in str(exc).lower()
            for word in ("ocr", "tesseract", "tessdata", "traineddata", "language data")
        ):
            raise
        return original, f"OCR is unavailable or failed: {exc}. {instructions}"
    recovered = page.get_text("text", textpage=textpage, sort=True)
    if not recovered.strip():
        return original, (
            "OCR recovered no searchable text; the page may be blank or unreadable. "
            f"Check the scan quality and OCR language. {instructions}"
        )
    if original.strip() and original.strip() not in recovered:
        recovered = original.rstrip() + "\n\n" + recovered
    return recovered, (
        "OCR was used; verify recognition of names, numbers, formulas and tables "
        "against the original page. Any selectable text was retained."
    )


def iter_pdf_pages(
    path: Path,
    document_id: str,
    filename: str,
    *,
    ocr: bool = False,
    ocr_language: str = "eng",
    tessdata_prefix: str = "",
    progress: Progress | None = None,
    vision: PageVision | None = None,
    vision_max_image_edge: int = 1536,
) -> Iterator[PageText]:
    """Yield every physical page (including blanks), with independent page labels.

    Low-text means fewer than 40 non-whitespace selectable characters. OCR is
    attempted only for those pages, and never removes their selectable text.
    Progress is a completed-page fraction in (0, 1], not a percentage.
    """
    filename = safe_filename(filename)
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise PDFError("The saved file must have a .pdf extension. Upload a PDF file.")
    try:
        with path.open("rb") as source:
            header = source.read(1024)
    except OSError as exc:
        raise PDFError(f"Cannot read PDF '{filename}'. Check the file and permissions: {exc}") from exc
    if not re.search(rb"%PDF-\d+\.\d+(?:[\r\n \t]|$)", header):
        raise PDFError("This file has no valid PDF header. Export it as a PDF and upload it again.")
    try:
        pdf = pymupdf.open(filename=str(path), filetype="pdf")
    except (OSError, *_PDF_ERRORS) as exc:
        raise PDFError(
            f"Cannot open PDF '{filename}'; it may be corrupt or incomplete. "
            f"Export a fresh, unencrypted PDF and upload it again. Details: {exc}"
        ) from exc

    with pdf:
        if not pdf.is_pdf:
            raise PDFError("This is not a PDF document. Export it as a PDF before uploading.")
        if pdf.needs_pass or pdf.is_encrypted or (pdf.metadata or {}).get("encryption"):
            raise PDFError(
                "This PDF is encrypted or password-protected. Save an unencrypted copy "
                "using your PDF reader, then upload that copy."
            )
        if pdf.page_count == 0:
            raise PDFError("This PDF contains zero pages. Export a PDF containing at least one page.")
        for index in range(pdf.page_count):
            number = index + 1
            try:
                page = pdf.load_page(index)
                text = page.get_text("text", sort=True)
                label = page.get_label() or ""
                has_images = bool(page.get_image_info())
                has_drawings = bool(page.get_drawings())
                has_layout = bool(text.strip()) or has_images or has_drawings
            except _PDF_ERRORS as exc:
                raise PDFError(
                    f"Cannot read page {number} of '{filename}'; the PDF may be damaged. "
                    f"Export a fresh PDF and upload it again. Details: {exc}"
                ) from exc
            warnings = []
            if len("".join(text.split())) < _LOW_TEXT_CHARACTERS:
                warnings.append(
                    f"Page {number}: Little or no selectable text was found; this may "
                    "be a scanned or blank page."
                )
                if ocr:
                    text, message = _ocr_text(page, text, ocr_language, tessdata_prefix)
                    warnings.append(f"Page {number}: {message}")
                else:
                    warnings.append(
                        f"Page {number}: OCR is disabled. Enable OCR and reindex scanned "
                        f"pages. {_ocr_instructions(ocr_language)}"
                    )
                    if vision is not None and (has_images or has_drawings):
                        warnings.append(
                            f"Page {number}: Visual indexing will interpret this page separately; "
                            "OCR would provide an additional text-recognition source."
                        )
            if has_layout or text.strip():
                warnings.append(
                    f"Page {number}: PDF reading order, tables, formulas and multi-column "
                    "layout may not be preserved by text extraction; verify the original page."
                )
            if has_images:
                warnings.append(
                    f"Page {number}: Contains images; text extraction and OCR do not "
                    "describe figures, charts or other visual content."
                )
            visual_result = None
            if vision is not None and (has_images or has_drawings):
                if progress is not None:
                    progress(index / pdf.page_count, f"Interpreting images and diagrams on PDF page {number}...")
                image_png = render_page_png(page, vision_max_image_edge)
                visual_result = vision.analyze_page(image_png, filename, number)
                if not visual_result.text.strip():
                    raise PDFError(f"Visual analysis of page {number} returned no usable interpretation.")
                warnings = [note for note in warnings if "Contains images;" not in note]
                warnings.append(
                    f"Page {number}: Visual interpretation was indexed using {vision.model_identity}. "
                    "It is a model interpretation, not verified extracted text; compare with the original."
                )
                warnings.extend(f"Page {number}: {warning}" for warning in visual_result.warnings)
            if progress is not None:
                progress(number / pdf.page_count, f"Read page {number} of {pdf.page_count}.")
            yield PageText(
                document_id=document_id,
                filename=filename,
                page_number=number,
                text=text,
                page_label=label,
                warnings=tuple(warnings),
            )
            if visual_result is not None:
                yield PageText(
                    document_id=document_id,
                    filename=filename,
                    page_number=number,
                    text=visual_result.text,
                    page_label=label,
                    source_type="visual",
                    source_model=vision.model_identity,
                )
