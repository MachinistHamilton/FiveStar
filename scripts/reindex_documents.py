import argparse

from config.settings import Settings
from core.catalog import Catalog
from core.document_manager import DocumentManager
from core.embeddings import Embeddings
from core.text_chunker import TextChunker


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex saved PDFs using the configured local models.")
    parser.add_argument("--document-id", help="Reindex only this content-hash ID; otherwise all saved PDFs.")
    parser.add_argument("--ocr", action="store_true", help="Also use local OCR on low-text pages.")
    args = parser.parse_args()
    settings = Settings()
    catalog = Catalog(settings.data_dir)
    documents = (
        [catalog.get_document(args.document_id)]
        if args.document_id else catalog.list_documents()
    )
    if not documents:
        print("No saved PDFs. Upload and process a document in Streamlit first.")
        return
    embedder = Embeddings(settings)
    manager = DocumentManager(
        settings, embedder,
        TextChunker(embedder, settings.chunk_tokens, settings.chunk_overlap),
    )
    try:
        for document in documents:
            print(f"Reindexing {document['filename']} (vision={settings.vision_enabled})", flush=True)
            result = manager.reindex(
                document["id"], ocr=args.ocr,
                progress=lambda fraction, message: print(
                    f"{fraction:6.1%} {message}", flush=True
                ),
            )
            summary = catalog.index_summary(document["id"])
            print(
                f"Saved {result.chunk_count} passages; "
                f"{len(summary.visual_pages) if summary else 0} pages with visual interpretations.",
                flush=True,
            )
    finally:
        manager.store.close()


if __name__ == "__main__":
    main()
