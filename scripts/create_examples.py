import argparse
from pathlib import Path

import pymupdf

from config.settings import PROJECT_ROOT

DOCUMENTS = {
    "Equipment_Manual.pdf": [
        "FICTITIOUS EQUIPMENT MANUAL - test material only\n\n"
        "Under normal operating conditions, the recommended inspection interval is 12 months.",
        "Heavy operating conditions\n\n"
        "Under heavy operating conditions, inspect every 6 months.\n"
        "Error code E104 indicates an inlet blockage.",
    ],
    "Service_Update.pdf": [
        "FICTITIOUS SERVICE UPDATE - test material only\n\n"
        "This document states that the normal inspection interval is 9 months.\n"
        "It does not establish which document takes precedence over the Equipment Manual.",
    ],
}


def create_examples(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, pages in DOCUMENTS.items():
        path = output / filename
        with pymupdf.open() as pdf:
            for text in pages:
                page = pdf.new_page()
                page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), text, fontsize=12)
            pdf.save(path)
        paths.append(path)
    return paths


def create_visual_example(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as drawing:
        page = drawing.new_page(width=600, height=400)
        page.insert_text((40, 50), "SYNTHETIC TEST DIAGRAM", fontsize=20)
        page.draw_rect(pymupdf.Rect(40, 120, 240, 240), color=(0, 0, 0), fill=(0.8, 0.9, 1))
        page.insert_text((65, 175), "INLET", fontsize=24)
        page.insert_text((65, 215), "E104", fontsize=24)
        page.draw_line(pymupdf.Point(240, 180), pymupdf.Point(420, 180), width=3)
        page.draw_line(pymupdf.Point(400, 165), pymupdf.Point(420, 180), width=3)
        page.draw_line(pymupdf.Point(400, 195), pymupdf.Point(420, 180), width=3)
        page.insert_text((420, 175), "OUTLET", fontsize=20)
        image = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png")
    path = output / "Flow_Diagram.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=400)
        page.insert_image(page.rect, stream=image)
        pdf.save(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic test PDFs.")
    parser.add_argument("--visual", action="store_true", help="Also create an image-only flow diagram.")
    args = parser.parse_args()
    for path in create_examples(PROJECT_ROOT / "examples" / "generated"):
        print(path)
    if args.visual:
        print(create_visual_example(PROJECT_ROOT / "examples" / "generated"))
