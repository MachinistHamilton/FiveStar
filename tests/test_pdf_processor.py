import hashlib
import shutil
import uuid
from pathlib import Path

import pymupdf
import pytest

from core.errors import PDFError
from core.pdf_processor import content_hash, iter_pdf_pages, safe_filename


@pytest.fixture
def pdf_path():
    directory = Path.cwd() / ".pytest_cache" / f"pdf-extraction-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory / "document.pdf"
    finally:
        shutil.rmtree(directory)


def write_pdf(path, texts, *, labels=None, encrypted=False, password="reader-password", images=False):
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            if text:
                page.insert_text((40, 70), text)
            if images:
                image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
                image.clear_with(255)
                page.insert_image(pymupdf.Rect(40, 100, 140, 200), pixmap=image)
        if labels:
            pdf.set_page_labels(labels)
        options = (
            {
                "encryption": pymupdf.PDF_ENCRYPT_AES_256,
                "owner_pw": "owner-password",
                "user_pw": password,
            }
            if encrypted else {}
        )
        path.write_bytes(pdf.tobytes(**options))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.pdf", "report.pdf"),
        (r"C:\uploads\..\report.pdf", "report.pdf"),
        ("C:report.pdf", "report.pdf"),
        ("../../uploads/report.pdf", "report.pdf"),
        (r"..\uploads/another\REPORT.PDF", "REPORT.PDF"),
        ("bad\x00na\nme\x7f\u202e.pdf", "badname.pdf"),
        ('a<b>c:d"e|f?g*.pdf', "a_b_c_d_e_f_g_.pdf"),
        ("CON.pdf", "_CON.pdf"),
        ("lpt1.details.pdf", "_lpt1.details.pdf"),
        ("COM¹.pdf", "_COM¹.pdf"),
        ("Résumé 日本語.pdf", "Résumé 日本語.pdf"),
    ],
)
def test_safe_filename_is_portable(name, expected):
    assert safe_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", ".pdf", "  .PDF", "file.txt", "file.pdf.exe", "file.pdf.", "file.pdf/"],
)
def test_safe_filename_rejects_invalid_suffixes(name):
    with pytest.raises(PDFError, match=".pdf"):
        safe_filename(name)


def test_safe_filename_limits_unicode_bytes():
    name = safe_filename("日" * 200 + ".pdf")
    assert len(name.encode("utf-8")) <= 255
    assert name.endswith(".pdf")
    assert "\ufffd" not in name


def test_hash_covers_original_bytes():
    assert content_hash(b"abc") == hashlib.sha256(b"abc").hexdigest()
    assert content_hash(b"abc") != content_hash(b"abc\n")
    assert content_hash(b"") == hashlib.sha256(b"").hexdigest()


def test_every_physical_page_has_its_own_label_metadata_and_progress(pdf_path):
    text = "There is enough selectable text here to avoid treating this page as a scan."
    write_pdf(
        pdf_path, [text, "", text],
        labels=[
            {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
            {"startpage": 2, "prefix": "A-", "style": "D", "firstpagenum": 7},
        ],
    )
    progress = []
    pages = list(iter_pdf_pages(
        pdf_path, "document-id", r"..\source.pdf",
        progress=lambda value, message: progress.append((value, message)),
    ))
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert [page.page_label for page in pages] == ["i", "ii", "A-7"]
    assert all(page.document_id == "document-id" and page.filename == "source.pdf" for page in pages)
    assert text in pages[0].text
    assert pages[1].text == ""
    assert any("blank" in warning and "Page 2" in warning for warning in pages[1].warnings)
    assert any("Enable OCR" in warning for warning in pages[1].warnings)
    assert any("tables" in warning for warning in pages[0].warnings)
    assert [item[0] for item in progress] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert all("page" in item[1] for item in progress)


def test_absent_labels_do_not_replace_physical_page_numbers(pdf_path):
    write_pdf(pdf_path, ["page one", "page two"])
    pages = list(iter_pdf_pages(pdf_path, "doc", "source.pdf"))
    assert [page.page_label for page in pages] == ["", ""]
    assert [page.page_number for page in pages] == [1, 2]


@pytest.mark.parametrize(
    "data", [b"", b"hello", b"<html>not a PDF</html>", b"\x89PNG\r\n\x1a\n", b"%PDF-1.7\nbroken"]
)
def test_invalid_and_corrupt_documents_are_actionable(pdf_path, data):
    pdf_path.write_bytes(data)
    with pytest.raises(PDFError, match="(?i)(upload|export)"):
        list(iter_pdf_pages(pdf_path, "doc", "broken.pdf"))


def test_missing_file_is_actionable(pdf_path):
    with pytest.raises(PDFError, match="Check the file"):
        list(iter_pdf_pages(pdf_path, "doc", "missing.pdf"))


def test_suffix_is_checked_on_display_and_saved_names(pdf_path):
    write_pdf(pdf_path, ["valid PDF content"])
    with pytest.raises(PDFError, match=".pdf"):
        list(iter_pdf_pages(pdf_path, "doc", "fake.txt"))
    other = pdf_path.with_suffix(".txt")
    pdf_path.rename(other)
    with pytest.raises(PDFError, match=".pdf"):
        list(iter_pdf_pages(other, "doc", "real.pdf"))


def test_pymupdf_cannot_auto_detect_a_renamed_image(pdf_path):
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    image.clear_with(255)
    pdf_path.write_bytes(image.tobytes("png"))
    with pytest.raises(PDFError, match="header"):
        list(iter_pdf_pages(pdf_path, "doc", "image.pdf"))


def test_explicit_pdf_parser_rejects_forged_header_on_image(pdf_path):
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    image.clear_with(255)
    pdf_path.write_bytes(b"%PDF-1.7\n" + image.tobytes("png"))
    with pytest.raises(PDFError, match="(?i)(corrupt|not a PDF)"):
        list(iter_pdf_pages(pdf_path, "doc", "image.pdf"))


@pytest.mark.parametrize("password", ["reader-password", ""])
def test_encrypted_pdf_requests_an_unencrypted_copy(pdf_path, password):
    write_pdf(pdf_path, ["secret"], encrypted=True, password=password)
    with pytest.raises(PDFError, match="encrypted.*password-protected"):
        list(iter_pdf_pages(pdf_path, "doc", "locked.pdf"))


def test_zero_page_pdf_is_rejected(pdf_path):
    # MuPDF cannot save a new empty document, so construct a minimal valid page tree.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
    ]
    data = b"%PDF-1.7\n"
    offsets = []
    for index, content in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{index} 0 obj\n".encode() + content + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 3\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets)
    data += (
        b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode() + b"\n%%EOF\n"
    )
    pdf_path.write_bytes(data)
    with pytest.raises(PDFError, match="zero pages"):
        list(iter_pdf_pages(pdf_path, "doc", "empty.pdf"))


def test_scanned_page_reports_ocr_and_visual_limitations(pdf_path):
    write_pdf(pdf_path, [""], images=True)
    page, = iter_pdf_pages(pdf_path, "doc", "scan.pdf")
    assert not page.text.strip()
    warnings = " ".join(page.warnings)
    assert "scanned or blank" in warnings
    assert "OCR is disabled" in warnings and "Enable OCR" in warnings
    assert "Contains images" in warnings and "tables" in warnings


def test_ocr_only_runs_for_low_text_pages_with_full_page_and_language_options(
    pdf_path, monkeypatch
):
    long_text = "This selectable text is long enough that OCR should never run on this page."
    write_pdf(pdf_path, ["native", long_text, ""])
    calls = []
    native_get_text = pymupdf.Page.get_text
    marker = object()

    def mock_ocr(page, **kwargs):
        calls.append((page.number, kwargs))
        return marker

    def mock_get_text(page, *args, **kwargs):
        if kwargs.get("textpage") is marker:
            return "Recognized image text with scientific notation E = mc²."
        return native_get_text(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", mock_ocr)
    monkeypatch.setattr(pymupdf.Page, "get_text", mock_get_text)
    pages = list(iter_pdf_pages(
        pdf_path, "doc", "scan.pdf", ocr=True,
        ocr_language="eng+deu", tessdata_prefix=r"C:\OCR\tessdata",
    ))
    assert calls == [
        (index, {"language": "eng+deu", "full": True, "tessdata": r"C:\OCR\tessdata"})
        for index in [0, 2]
    ]
    assert "native" in pages[0].text and "E = mc²" in pages[0].text
    assert long_text in pages[1].text
    assert any("OCR was used" in warning for warning in pages[2].warnings)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("No tessdata specified and Tesseract is not installed"),
        RuntimeError("OCR initialisation failed"),
        OSError("Could not read eng.traineddata"),
    ],
)
def test_unavailable_ocr_is_explicit_and_preserves_selectable_text(pdf_path, monkeypatch, failure):
    write_pdf(pdf_path, ["selectable"])

    def fail_ocr(_page, **_kwargs):
        raise failure

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", fail_ocr)
    page, = iter_pdf_pages(pdf_path, "doc", "scan.pdf", ocr=True)
    assert "selectable" in page.text
    warnings = " ".join(page.warnings)
    assert "OCR is unavailable" in warnings
    assert "Install Tesseract" in warnings and "TESSDATA_PREFIX" in warnings
    assert "enable OCR and reindex" in warnings


def test_missing_ocr_api_is_explicit(pdf_path, monkeypatch):
    write_pdf(pdf_path, ["native"])
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", None)
    page, = iter_pdf_pages(pdf_path, "doc", "scan.pdf", ocr=True)
    assert "native" in page.text
    assert any("OCR is unavailable" in warning for warning in page.warnings)


@pytest.mark.parametrize("recovered", ["", "native\n"])
def test_empty_or_duplicate_ocr_keeps_selectable_text(pdf_path, monkeypatch, recovered):
    write_pdf(pdf_path, ["native"])
    marker = object()
    native_get_text = pymupdf.Page.get_text
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", lambda *_args, **_kwargs: marker)

    def get_text(page, *args, **kwargs):
        if kwargs.get("textpage") is marker:
            return recovered
        return native_get_text(page, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", get_text)
    page, = iter_pdf_pages(pdf_path, "doc", "scan.pdf", ocr=True)
    assert page.text.count("native") == 1
    if not recovered:
        assert any("OCR recovered no searchable text" in warning for warning in page.warnings)


def test_unrelated_ocr_errors_are_not_silenced(pdf_path, monkeypatch):
    write_pdf(pdf_path, ["native"])

    def fail(_page, **_kwargs):
        raise RuntimeError("unrelated application bug")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", fail)
    with pytest.raises(RuntimeError, match="unrelated application bug"):
        list(iter_pdf_pages(pdf_path, "doc", "scan.pdf", ocr=True))


def test_progress_errors_are_not_reported_as_corrupt_pdfs(pdf_path):
    write_pdf(pdf_path, ["text"])

    def fail(_fraction, _message):
        raise RuntimeError("UI callback failed")

    with pytest.raises(RuntimeError, match="UI callback"):
        list(iter_pdf_pages(pdf_path, "doc", "document.pdf", progress=fail))


def test_generator_close_releases_the_pdf(pdf_path, monkeypatch):
    write_pdf(pdf_path, ["one", "two"])
    opened = []
    original = pymupdf.open

    def capture(*args, **kwargs):
        pdf = original(*args, **kwargs)
        opened.append(pdf)
        return pdf

    monkeypatch.setattr(pymupdf, "open", capture)
    pages = iter_pdf_pages(pdf_path, "doc", "document.pdf")
    next(pages)
    assert not opened[0].is_closed
    pages.close()
    assert opened[0].is_closed
