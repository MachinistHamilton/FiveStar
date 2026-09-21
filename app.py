import logging
import sqlite3
from dataclasses import asdict

import pymupdf
import streamlit as st
from chromadb.errors import ChromaError
from pydantic import ValidationError

from config.settings import Settings
from core.catalog import Catalog
from core.creation import CreationPipeline
from core.document_manager import DocumentManager, clear_library, delete_document
from core.errors import AssistantError, LibraryBusyError
from core.keyword_retriever import KeywordRetriever, SelectedEvidenceRetriever
from core.library_lock import BUSY_MESSAGE, library_is_busy
from core.llm import OllamaClient
from core.page_analysis import analyze_saved_page
from core.rag_pipeline import RAGPipeline, validate_request
from core.text_chunker import keyword_chunker
from core.types import Answer, Draft, SearchHit

logger = logging.getLogger(__name__)
UI_ERRORS = (
    AssistantError, OSError, ValueError, RuntimeError, sqlite3.Error,
    ChromaError, pymupdf.mupdf.FzErrorBase,
)


@st.cache_resource(show_spinner=False, max_entries=2)
def get_manager(settings: Settings) -> DocumentManager:
    if library_is_busy(settings.data_dir):
        raise LibraryBusyError(BUSY_MESSAGE)
    text_settings = settings.model_copy(update={"vision_enabled": False})
    return DocumentManager(text_settings, None, keyword_chunker())


@st.cache_resource(show_spinner=False, max_entries=2)
def get_retriever(settings: Settings) -> KeywordRetriever:
    return KeywordRetriever(settings.data_dir, settings.candidate_count)


def report_error(action: str, error: Exception) -> None:
    if isinstance(error, LibraryBusyError):
        logger.info("%s deferred: %s", action, error)
        st.info(str(error))
        return
    logger.exception("%s failed", action)
    st.error(f"{action}: {error}")


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pop("viewer", None)


def advanced_settings(base: Settings) -> Settings:
    if "runtime" not in st.session_state:
        st.session_state.runtime = base
    current = st.session_state.runtime
    with st.expander("Search and optional AI settings"):
        with st.form("settings"):
            model = st.text_input("Ollama model", current.ollama_model)
            st.caption("AI is used only when you click Explain, Create or Analyze. Ordinary search needs no model.")
            context = st.number_input(
                "Context window (tokens)", 2048, 131072, current.context_window, step=1024
            )
            output = st.number_input(
                "Maximum answer tokens", 128, 8192, current.max_output_tokens, step=128
            )
            temperature = st.slider("Temperature", 0.0, 1.0, current.temperature, step=0.05)
            candidates = st.slider("Keyword result limit", 1, 100, current.candidate_count)
            evidence = st.slider("Passages for optional AI actions", 1, 20, current.evidence_count)
            threads = st.number_input("AI CPU threads", 1, 32, current.inference_threads)
            verify = st.checkbox("Verify claims against excerpts (slower)", current.verify_claims)
            if st.form_submit_button("Apply settings"):
                try:
                    st.session_state.runtime = Settings(**(
                        base.model_dump() | {
                            "ollama_model": model.strip(),
                            "context_window": context,
                            "max_output_tokens": output,
                            "temperature": temperature,
                            "candidate_count": candidates,
                            "evidence_count": evidence,
                            "inference_threads": threads,
                            "verify_claims": verify,
                        }
                    ))
                    st.success("Settings applied.")
                except ValidationError as exc:
                    st.error(str(exc))
        if st.button("Check Ollama connection"):
            client = OllamaClient(st.session_state.runtime)
            try:
                client.check_available()
                st.success("Selected local model is available.")
                st.write("Installed models:", ", ".join(client.list_models()))
            except UI_ERRORS as exc:
                report_error("Ollama connection", exc)
            finally:
                client.close()
        st.caption(
            "AI uses the configured CPU thread limit and unloads after each response. "
            "It may still be demanding; analyzing one chosen page never starts a whole-document job."
        )
    return st.session_state.runtime


def render_extraction_notes(warnings: list[str]) -> None:
    if not warnings:
        return
    image_notes = [note for note in warnings if "Contains images;" in note]
    layout_notes = [note for note in warnings if "PDF reading order" in note]
    visual_notes = [note for note in warnings if "Visual interpretation was indexed" in note]
    generic_notes = set(image_notes + layout_notes + visual_notes)
    other_notes = [note for note in warnings if note not in generic_notes]
    if image_notes:
        st.info(
            f"{len(image_notes)} PDF pages contain images. This does not mean text extraction "
            "failed: selectable text on those pages can still be indexed. Text inside "
            "screenshots and the meaning of diagrams may be missing."
        )
    if layout_notes:
        st.caption("Text extraction may change table, code and multi-column reading order. Compare with the original.")
    if visual_notes:
        st.info(
            f"Visual interpretation is saved for {len(visual_notes)} pages. These model-produced "
            "descriptions are searchable, but can misread images; verify against the original."
        )
    if other_notes:
        st.warning(
            "Some pages have little selectable text, OCR notices or visual-analysis uncertainties. "
            "Check the page-by-page notes below; low-text pages can also be covers or intentionally blank."
        )
    with st.expander(f"Page-by-page extraction notes ({len(warnings)})"):
        st.text("\n".join(other_notes + visual_notes + image_notes + layout_notes))


def sidebar(
    base: Settings, catalog: Catalog, *, busy: bool = False
) -> tuple[Settings, list[str] | None]:
    with st.sidebar:
        st.title("Local AI Research")
        st.caption("Your documents stay on this computer.")
        if busy:
            st.info(
                "A library operation is already running. Processing, reindexing and deletion "
                "are temporarily disabled. You do not need to upload the same PDF again."
            )
        uploads = st.file_uploader(
            "Choose PDFs", type=["pdf"], accept_multiple_files=True, disabled=busy
        )
        st.caption(
            "Processing extracts searchable text only: no embeddings, language model or vision "
            "model. Analyze a specific page later if its images matter."
        )
        use_ocr = st.checkbox(
            "OCR pages with little selectable text",
            help="Requires local Tesseract language data. Mixed PDFs are detected page by page.",
        )
        if st.button("Process documents", disabled=busy or not uploads, type="primary"):
            progress = st.progress(0.0, text="Extracting local PDF text...")
            for upload in uploads:
                try:
                    manager = get_manager(base)
                    result = manager.ingest(
                        upload.getvalue(), upload.name, ocr=use_ocr,
                        progress=lambda value, text, bar=progress: bar.progress(value, text=text),
                    )
                    if result.duplicate:
                        st.info(f"{upload.name}: text already indexed. Use Reindex to change OCR.")
                    else:
                        st.success(f"{upload.name}: indexed {result.chunk_count} passages.")
                    st.caption("Check indexed page coverage and extraction notes in the document library below.")
                    reset_chat()
                except UI_ERRORS as exc:
                    report_error(f"Processing {upload.name}", exc)
            progress.empty()

        documents = catalog.list_documents()
        st.subheader(f"Document library ({len(documents)})")
        for document in documents:
            with st.expander(document["filename"]):
                summary = catalog.index_summary(document["id"])
                st.caption(f"{document['page_count']} physical PDF pages")
                if document["deleting"]:
                    st.warning("Deletion was interrupted. Retry Delete to finish.")
                elif document["error"]:
                    st.error(document["error"])
                if summary:
                    st.success(
                        f"Searchable text: {len(summary.pages)} of {document['page_count']} pages "
                        f"| {summary.chunk_count} indexed passages"
                    )
                    st.caption("Coverage of the most recent successful index; not a measure of extraction accuracy.")
                    st.caption(
                        f"PDF/OCR text: {len(summary.text_pages)} pages | "
                        f"Visual interpretation: {len(summary.visual_pages)} pages"
                    )
                elif not document["deleting"]:
                    st.info("Not indexed for search.")
                saved_count = len(catalog.saved_visual_pages([document["id"]]))
                if saved_count:
                    st.caption(
                        f"Saved image analyses: {saved_count} pages. These completed analyses "
                        "are available to keyword search, even if a full visual reindex was paused."
                    )
                st.caption(f"Document ID: {document['id'][:16]}...")
                if st.button("Inspect extracted text", key=f"text_{document['id']}"):
                    st.session_state.viewer = (
                        document["id"], summary.pages[0] if summary and summary.pages else 1
                    )
                if st.button("View original", key=f"view_{document['id']}"):
                    st.session_state.viewer = (document["id"], 1)
                render_extraction_notes(document["warnings"])
                if st.button("Reindex", key=f"reindex_{document['id']}", disabled=busy):
                    progress = st.progress(0.0, text="Reindexing...")
                    try:
                        get_manager(base).reindex(
                            document["id"], ocr=use_ocr,
                            progress=lambda value, text, bar=progress: bar.progress(value, text=text),
                        )
                        reset_chat()
                        st.rerun()
                    except UI_ERRORS as exc:
                        report_error("Reindexing", exc)
                    finally:
                        progress.empty()
                confirm = st.checkbox(
                    "Confirm deletion", key=f"confirm_{document['id']}"
                )
                if st.button("Delete", key=f"delete_{document['id']}", disabled=busy or not confirm):
                    try:
                        delete_document(base.data_dir, document["id"])
                        reset_chat()
                        st.rerun()
                    except UI_ERRORS as exc:
                        report_error("Deleting document", exc)
        options = [document["id"] for document in documents if not document["deleting"]]
        names = {document["id"]: document["filename"] for document in documents}
        selected = st.multiselect(
            "Search only these documents (empty = all)",
            options, format_func=lambda identifier: names[identifier],
            key="document_filter",
            on_change=reset_chat,
        )
        runtime = advanced_settings(base)
        if st.button("Clear conversation"):
            reset_chat()
            st.rerun()
        with st.expander("Delete all local documents"):
            st.warning("Permanently removes originals, embeddings and metadata for this library.")
            confirmed = st.checkbox("I confirm permanent deletion of the entire library")
            if st.button("Delete all documents", disabled=busy or not confirmed):
                try:
                    clear_library(base.data_dir, confirmed=confirmed)
                    reset_chat()
                    st.rerun()
                except UI_ERRORS as exc:
                    report_error("Clearing library", exc)
        return runtime, selected or None


def render_answer(answer: Answer, key: str) -> None:
    st.markdown(answer.text)
    for warning in answer.warnings:
        st.warning(warning)
    if answer.sources:
        with st.expander(f"Supporting evidence ({len(answer.sources)} passages)"):
            st.caption(
                "Citations identify retrieved passages, not proof that every claim is true. "
                "Check the excerpts and originals. Scores are rankings, not confidence."
            )
            for source in answer.sources:
                hit, chunk = source.hit, source.hit.chunk
                st.text(
                    f"SOURCE {source.source_id} | {chunk.filename} | "
                    f"PDF page {chunk.page_number}"
                    + (f"-{chunk.page_end}" if chunk.page_end != chunk.page_number else "")
                )
                if chunk.page_label and chunk.page_label != str(chunk.page_number):
                    st.caption(f"PDF page label: {chunk.page_label} (not inferred from printed text)")
                if chunk.source_type == "visual":
                    st.warning(f"Saved visual interpretation by {chunk.source_model}; verify against the original.")
                st.text(chunk.text)
                scores = (
                    [f"Keyword score: {hit.keyword_score:.3f}"]
                    if hit.semantic_score is None and hit.keyword_score is not None
                    else [f"RRF: {hit.score:.4f}"]
                )
                if hit.rerank_score is not None:
                    scores.append(f"Reranker: {hit.rerank_score:.3f}")
                if hit.exact_match:
                    scores.append("Exact identifier / phrase / document-name match")
                st.caption(" | ".join(scores))
                if st.button("Inspect PDF page", key=f"{key}_{source.source_id}"):
                    st.session_state.viewer = (chunk.document_id, chunk.page_number)
                    st.rerun()


def render_viewer(catalog: Catalog, settings: Settings, *, busy: bool = False) -> None:
    if "viewer" not in st.session_state:
        return
    identifier, target_page = st.session_state.viewer
    try:
        document = catalog.get_document(identifier)
        path = catalog.document_path(identifier)
        with st.expander(f"Original PDF: {document['filename']}", expanded=True):
            with path.open("rb") as file:
                st.download_button(
                    "Download / open original PDF", file,
                    file_name=document["filename"], mime="application/pdf",
                )
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass or not len(pdf):
                    st.warning("This PDF cannot be previewed; it is encrypted or has no pages.")
                    return
                page = st.number_input(
                    "Physical PDF page", 1, len(pdf), min(target_page, len(pdf)),
                    key=f"page_{identifier}_{target_page}",
                )
                original, extracted = st.columns(2)
                with original:
                    st.subheader("Original page")
                    pixmap = pdf[int(page) - 1].get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2))
                    st.image(pixmap.tobytes("png"))
                    st.caption("Preview is rendered locally. Use the original PDF to inspect fine detail.")
                with extracted:
                    st.subheader("Text stored for search")
                    summary = catalog.index_summary(identifier)
                    passages = catalog.indexed_passages(identifier, summary.generation, int(page)) if summary else []
                    if passages:
                        st.caption(
                            f"{len(passages)} indexed passages on this page, from the most recent "
                            "successful index. Overlapping text is intentional. No AI model is "
                            "needed to inspect these passages."
                        )
                        for index, passage in enumerate(passages, start=1):
                            st.caption(f"Passage {index}")
                            if passage.source_type == "visual":
                                st.warning(
                                    f"Visual interpretation by {passage.source_model}; "
                                    "not verbatim or independently verified text."
                                )
                            else:
                                st.caption("Extracted PDF/OCR text")
                            st.text(passage.text)
                    else:
                        st.warning(
                            "No indexed text is stored for this page. It may be blank, scanned "
                            "or not yet indexed. Inspect the original; if it contains text, "
                            "enable OCR and reindex."
                        )
                    saved = [
                        item for item in catalog.saved_visual_pages([identifier])
                        if item.page_number == int(page)
                    ]
                    if saved:
                        st.subheader("Saved image analysis")
                        st.caption(f"Visual interpretation by {saved[0].model_identity}; verify against the original.")
                        st.text(saved[0].text)
                        for warning in saved[0].warnings:
                            st.warning(warning)
                st.caption(
                    "Optional AI image analysis runs only for the selected physical page and saves "
                    "the result for future keyword searches. It can still use significant CPU."
                )
                if st.button(
                    "Analyze this page's images (AI)",
                    key=f"analyze_{identifier}_{int(page)}", disabled=busy,
                ):
                    try:
                        with st.spinner("Analyzing only this page locally; not the whole PDF..."):
                            analyze_saved_page(settings, identifier, int(page))
                        st.session_state.analysis_saved = True
                        st.rerun()
                    except UI_ERRORS as exc:
                        report_error("Image analysis", exc)
                if st.session_state.pop("analysis_saved", False):
                    st.success("Page analysis saved. Search again to include its descriptions.")
    except UI_ERRORS as exc:
        report_error("Opening original PDF", exc)


def validated_evidence(
    settings: Settings, question: str, hits: list[SearchHit],
    document_ids: list[str] | None, selected_ids: list[str] | None = None,
) -> list[SearchHit]:
    displayed = {hit.chunk.chunk_id: hit for hit in hits}
    if selected_ids is None:
        selected_ids = [hit.chunk.chunk_id for hit in hits[:settings.evidence_count]]
    if (
        not selected_ids or len(selected_ids) > settings.evidence_count
        or len(set(selected_ids)) != len(selected_ids)
        or any(identifier not in displayed for identifier in selected_ids)
    ):
        raise ValueError(f"Select between 1 and {settings.evidence_count} displayed passages for the AI action.")
    current = {
        hit.chunk.chunk_id: hit for hit in get_retriever(settings).retrieve(question, document_ids)
    }
    if any(
        identifier not in current or current[identifier].chunk != displayed[identifier].chunk
        for identifier in selected_ids
    ):
        raise ValueError("These search results changed. Search again before requesting an AI action.")
    return [current[identifier] for identifier in selected_ids]


def explain_results(
    settings: Settings, question: str, hits: list[SearchHit],
    document_ids: list[str] | None,
) -> Answer:
    selected = validated_evidence(settings, question, hits, document_ids)
    client = OllamaClient(settings)
    try:
        client.check_available()
        pipeline = RAGPipeline(settings, SelectedEvidenceRetriever(selected), client)
        return pipeline.ask(question, document_ids=document_ids)
    finally:
        client.close()


def create_results(
    settings: Settings, request: str, question: str, hits: list[SearchHit],
    document_ids: list[str] | None, selected_ids: list[str],
) -> Draft:
    request = validate_request(request)
    selected = validated_evidence(settings, question, hits, document_ids, selected_ids)
    client = OllamaClient(settings)
    try:
        client.check_available()
        pipeline = CreationPipeline(settings, SelectedEvidenceRetriever(selected), client)
        return pipeline.create(request, document_ids=document_ids)
    finally:
        client.close()


def render_draft(draft: Draft, key: str) -> None:
    st.caption("Creation request")
    st.text(draft.request)
    if draft.content:
        st.warning(
            "AI-generated draft, not a verified solution. Code has not been run or tested. "
            "Review it and test in the target application using a copy of your project."
        )
        if draft.kind == "code":
            st.code(draft.content, language=None)
        else:
            st.text(draft.content)
        st.download_button(
            "Download draft (.txt)", draft.content,
            file_name="generated-draft.txt", mime="text/plain", key=f"{key}_download",
        )
    if draft.assumptions:
        st.caption("Assumptions and design choices")
        for assumption in draft.assumptions:
            st.text(assumption)
    if draft.questions:
        st.info("More details are needed. Edit the creation request with your answers, or search for better evidence.")
        for question in draft.questions:
            st.text(question)
    st.caption("Documented basis (not proof that the draft works)" if draft.content else "Draft status")
    render_answer(draft.basis, key)


def render_creation(
    message: dict, index: int, settings: Settings, document_ids: list[str] | None,
    *, busy: bool,
) -> None:
    hits: list[SearchHit] = message["hits"]
    with st.expander("Create from these documents - code, checklists and more"):
        st.caption(
            "Choose relevant passages and describe what to create. For PowerMill, include your "
            "version, goal and required inputs. This uses only selected search evidence, not "
            "the whole PDF or previous chat. Missing commands or details should trigger questions."
        )
        labels = {
            hit.chunk.chunk_id: f"{rank}. {hit.chunk.filename} - PDF page {hit.chunk.page_number}"
            + (" (visual interpretation)" if hit.chunk.source_type == "visual" else "")
            for rank, hit in enumerate(hits, 1)
        }
        with st.form(f"creation_{index}"):
            selected = st.multiselect(
                "Passages for this draft", list(labels),
                default=list(labels)[:settings.evidence_count], format_func=labels.__getitem__,
                max_selections=settings.evidence_count, key=f"draft_sources_{index}",
            )
            request = st.text_area(
                "What should I create?", value=message["question"], key=f"draft_request_{index}",
                help="For example: Draft a read-only PowerMill macro that lists toolpath names. "
                "Add the version and constraints. Maximum 2048 UTF-8 bytes.",
            )
            st.caption(
                "This loads the local model and may be slow. The app never executes generated "
                "code. Verification, when enabled, checks the documented premises, not the code."
            )
            submitted = st.form_submit_button("Create from these documents (AI)", disabled=busy)
        if submitted:
            try:
                with st.spinner("Drafting from the selected passages locally..."):
                    draft = create_results(
                        settings, request, message["question"], hits, document_ids, selected
                    )
                message.setdefault("drafts", []).append(draft)
                st.rerun()
            except UI_ERRORS as exc:
                report_error("Creating a draft", exc)
    for number, draft in enumerate(message.get("drafts", []), 1):
        st.subheader(f"Creation result {number}")
        render_draft(draft, f"draft_{index}_{number}")


def render_search_results(
    message: dict, index: int, settings: Settings, document_ids: list[str] | None,
    *, busy: bool,
) -> None:
    hits: list[SearchHit] = message["hits"]
    if not hits:
        st.info(
            "No matching keywords were found. Try an exact identifier, a quoted phrase, "
            "or different wording. Unanalyzed images are not searchable; open a relevant "
            "page from the library to inspect it or explicitly request image analysis."
        )
        return
    st.caption(f"{len(hits)} matching passages. Keyword search only; no AI ran for this search.")
    for rank, hit in enumerate(hits, 1):
        chunk = hit.chunk
        with st.expander(f"{rank}. {chunk.filename} - PDF page {chunk.page_number}", expanded=rank <= 3):
            if chunk.source_type == "visual":
                st.warning(f"Previously saved visual interpretation by {chunk.source_model}; verify the original.")
            st.text(chunk.text)
            st.caption(f"Keyword score: {hit.score:.3f} (ranking, not confidence)")
            if st.button("View page", key=f"result_{index}_{rank}"):
                st.session_state.viewer = (chunk.document_id, chunk.page_number)
                st.rerun()
    st.caption(
        f"Optional: ask the local AI to explain up to {settings.evidence_count} of these passages. "
        "This will load a model and may take time."
    )
    if st.button("Explain these results (AI)", key=f"explain_{index}", disabled=busy):
        try:
            with st.spinner("Generating a local AI explanation from the displayed results..."):
                message["answer"] = explain_results(
                    settings, message["question"], hits, document_ids
                )
            st.rerun()
        except UI_ERRORS as exc:
            report_error("AI explanation", exc)
    if message.get("answer") is not None:
        st.subheader("Optional AI explanation")
        render_answer(message["answer"], f"answer_{index}")
    render_creation(message, index, settings, document_ids, busy=busy)


def main() -> None:
    st.set_page_config(page_title="Local AI Document Research Assistant", page_icon=":books:", layout="wide")
    try:
        base = Settings()
        catalog = Catalog(base.data_dir)
        busy = library_is_busy(base.data_dir)
    except UI_ERRORS as exc:
        report_error("Loading configuration", exc)
        st.stop()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    runtime, document_ids = sidebar(base, catalog, busy=busy)
    st.title("Ask your documents")
    st.caption(
        "Type a question or keywords to find saved passages and physical PDF pages. "
        "No AI runs unless you explicitly request an explanation, a draft or image analysis."
    )
    if not catalog.list_documents():
        st.info("Choose PDFs in the sidebar, then select Process documents to get started.")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            render_search_results(message, index, runtime, document_ids, busy=busy)
    render_viewer(catalog, runtime, busy=busy)
    unfinished = catalog.unfinished_documents()
    if unfinished and busy:
        st.info(
            "Indexing has not finished for: " + ", ".join(unfinished)
            + ". Visual analysis can take a long time; completed page analyses are saved. "
            "Wait for the indexing process, then refresh below. No interrupted job "
            "will restart automatically."
        )
    elif unfinished:
        st.info(
            "An earlier indexing job did not finish. Keyword search still uses completed "
            "indexes and saved page analyses; it will not restart that job."
        )
    elif busy:
        st.info("The library is busy with another operation. Wait for it to finish, then refresh.")
    if unfinished or busy:
        if st.button("Refresh indexing status"):
            st.rerun()
    question = st.chat_input(
        "Type a question or keywords to search your PDFs", disabled=busy
    )
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching saved text..."):
                    hits = get_retriever(runtime).retrieve(question, document_ids)
                st.session_state.messages.append({
                    "question": question, "hits": hits, "answer": None, "drafts": [],
                })
                st.rerun()
            except UI_ERRORS as exc:
                report_error("Keyword search", exc)
    if st.session_state.messages:
        import json
        export = [
            {
                "question": item["question"],
                "results": [asdict(hit) for hit in item["hits"]],
                "answer": asdict(item["answer"]) if item["answer"] is not None else None,
                "drafts": [asdict(draft) for draft in item.get("drafts", [])],
            }
            for item in st.session_state.messages
        ]
        st.download_button(
            "Export conversation", json.dumps(export, indent=2),
            file_name="research-conversation.json", mime="application/json",
        )


if __name__ == "__main__":
    main()
