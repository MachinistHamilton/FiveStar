# Local AI Document Research Assistant

A local, keyword-first PDF research app. Upload PDFs, type a question or keywords,
and inspect matching passages and physical pages **without loading an AI model**.
Only explicit **Explain these results (AI)**, **Create from these documents (AI)**
or **Analyze this page's images (AI)** actions invoke local Ollama. No paid API or
cloud inference is used.

Normal processing uses PyMuPDF, SQLite and BM25. Sentence Transformers, Chroma
and hybrid retrieval remain available to advanced scripts, not ordinary UI search.

**Target machine:** Windows, 16 GB RAM, NVIDIA GPU. The development machine reports
a 4 GB RTX 3050 Ti Laptop GPU (the initial estimate was 6 GB). Whole-document
visual analysis overloaded this machine, so the normal workflow never starts it.
Keyword search needs no GPU or model downloads. Internet is needed for initial
installation and any optional model downloads, not normal operation.

This is a single-user research tool, not a guarantee of factual accuracy or a
replacement for reviewing original documents. A small local model can misread
evidence, miss a conflict, or incorrectly abstain. An existing citation is not
proof of entailment. Tables, equations, diagrams and OCR need particular care.

## 1. Windows setup

Use a 64-bit Python **3.11 or newer**; Python 3.12 is the tested version. Install
[Python](https://www.python.org/downloads/) (enable its PATH option),
[VS Code](https://code.visualstudio.com/) with the Python extension.
[Ollama for Windows](https://ollama.com/download/windows) is optional, needed only
for explicitly requested explanations, drafts or image analysis. It uses its own GPU
runtime; CUDA-enabled PyTorch is not required.

Open this project folder in VS Code. In a PowerShell terminal in the project root:

```powershell
# If starting from a fresh checkout:
git clone https://github.com/MachinistHamilton/FiveStar.git
Set-Location FiveStar

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

If you already have the project open, skip the clone and Set-Location commands.
For a runtime-only installation, use `requirements.txt` instead of
`requirements-dev.txt`. Dependency versions are pinned at the application level;
`python -m pip check` verifies the resolved installation.

If PowerShell blocks activation, **do not change system-wide policy**. Invoke the
virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In VS Code, run **Python: Select Interpreter** and select
`.venv\Scripts\python.exe`.

For optional AI, install Ollama and explicitly download the model you configure.
This laptop uses `qwen3-vl:4b-instruct` for explanations, drafts and page images:

```powershell
ollama pull qwen3-vl:4b-instruct
# If Ollama's desktop app has not already started its server:
ollama serve
```

Set `OLLAMA_MODEL=qwen3-vl:4b-instruct` and `VISION_MODEL=qwen3-vl:4b-instruct`
in `.env` and restart Streamlit. Keep `VISION_ENABLED=false`. No embedding or
reranker download is needed for this workflow. Do not start a second Ollama server
if one is already running.

Open `http://127.0.0.1:8501` (or your configured port).

1. Choose PDFs and click **Process documents**. This extracts text without
   embeddings or AI. Optional OCR must be selected explicitly.
2. Type a question or keywords. Results show saved passages, filenames and
   physical pages; submitting the question does **not** generate an AI answer.
3. Click **View page** to compare a passage with the original.
4. Optionally click **Explain these results (AI)** to explain only the displayed
   evidence, with citations.
5. If an image matters, select its physical page and explicitly click
   **Analyze this page's images (AI)**. This analyzes only that page. Search again
   to include its saved description.
6. To produce something new, open **Create from these documents - code, checklists
   and more**, select relevant passages, describe the desired output, then click
   **Create from these documents (AI)**. Merely opening the form or editing it
   does not run AI.

Existing completed indexes and saved image analyses are reused without another
upload. An interrupted bulk job is not restarted by searching.

In each library entry, **Searchable text** shows how many physical pages have
stored search passages. **Inspect extracted text** displays the original page
beside the exact passages stored in the most recent successful index; it works
without Ollama. Repeated text between passages is intentional chunk overlap.
Coverage is not an accuracy score: a page can contain searchable text while a
screenshot or figure on that same page remains unreadable. Generic image/layout
notices are summarized separately from low-text/OCR notices, with full details in
**Page-by-page extraction notes**.

### Model size and hardware

The optional text model in `.env.example` is Ollama's quantized `qwen3:8b` (normally Q4_K_M). Its weights
plus context/cache overhead exceed 4 GB VRAM and may exceed 6 GB. Ollama can offload part to RAM,
but this is slower. Close memory-heavy applications and begin with one document.
No GPT-3.5-level performance is promised.

For a lighter, explicitly selected alternative:

```powershell
ollama pull qwen3:4b
```

Set `OLLAMA_MODEL=qwen3:4b` in `.env` or enter it in the sidebar. There is **no silent
model substitution**. Models larger than 8B are not recommended on this machine.
Model downloads require several GB of disk space; keep extra room for Python,
embedding models, original PDFs and indexes.

You can explicitly use `qwen3-vl:4b-instruct` for **both** page images and requested
explanations by setting `OLLAMA_MODEL=qwen3-vl:4b-instruct` and `VISION_MODEL=qwen3-vl:4b-instruct`.
This avoids downloading/loading a second large model. This checkout's local
configuration uses that choice at the user's request; `.env.example` retains the
separate Qwen3-8B starting configuration for fresh installations.

AI requests default to `INFERENCE_THREADS=2` and `keep_alive=0` to request model
unloading after each response. This is **not a hard CPU-percentage limit**: model
loading, image encoding, OCR and generation can still be demanding. Remain in
keyword-only mode if responsiveness is a concern. Verification performs another
AI call and can be disabled in **Search and optional AI settings**.

### macOS / Linux differences

Install Python (and optionally Ollama) for your OS, then replace the Windows environment
commands with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
streamlit run app.py
```

All application paths resolve relative to this project, even if a script is
started from another directory. Launch Streamlit from the project root so its
local-only server configuration is loaded.
Headless startup skips Streamlit's first-run email prompt; open the local URL manually.
Automatic source-file watching is disabled because Streamlit's module inspection
can import optional Transformers vision modules unnecessarily. Restart Streamlit
after editing Python files; those optional vision packages are not needed here.

## 2. Screenshots, diagrams, scanned PDFs and OCR

### Local visual understanding

**OCR reads words; a vision model can also describe screenshots, diagrams and
visual relationships.** Open a relevant page, then click
**Analyze this page's images (AI)**. The ordinary processing/reindex buttons never
run vision, even if an advanced CLI configuration has `VISION_ENABLED=true`.

```powershell
ollama pull qwen3-vl:4b-instruct
python -m scripts.check_environment --vision
```

Qwen3-VL requires Ollama 0.12.7 or newer. `VISION_MODEL=qwen3-vl:4b-instruct` is the quality-first
default. `qwen3-vl:2b-instruct` is an explicitly configurable lighter alternative; there is
no automatic substitution. The 4B model may offload to RAM on a 4 GB GPU. Visual
analysis is substantially slower than text extraction, especially on a long manual.

Use the explicit **Instruct** tag. With Ollama 0.34.2, the unqualified `4b` tag
placed structured JSON entirely in the thinking field with an empty answer,
even with `think=false`. The application rejects empty answers rather than
silently using a model's reasoning field as source evidence.

For the one selected physical page, the app renders a bounded page image locally
and sends it only to the loopback Ollama server.
The model transcribes visible labels/code, describes diagrams and relationships,
records visibly demonstrated actions, and states uncertainties. Pages containing
both text and images can be selected too, independently of the OCR detector.

The resulting descriptions are **saved in the local SQLite catalog**, with
separate provenance and no embedding step. Later keyword searches can find
image-only terminology and diagrams. Citations, supporting excerpts and the
page preview identify **Visual interpretation** so model-generated observations
cannot be mistaken for verbatim PDF text. Claims based on them are explicitly
labeled and must be checked against the original.

Analysis is cached by document, physical page, rendered image, model digest and
configuration. Repeating the same page action reuses a matching result without
another inference; validating the installed model identity still requires Ollama.
Failures are reported without replacing good evidence. Completed analyses from
an interrupted bulk job are also searchable without resuming it. Changing vision
settings takes effect when you explicitly analyze a page again. Deleting a
document deletes its saved analyses.

### Advanced bulk/hybrid indexing (not the low-resource workflow)

The CLI retains semantic embedding and optional whole-document visual indexing.
It can be very CPU/memory intensive and is not recommended on this laptop.
Only run it if you deliberately want that work:

```powershell
python -m scripts.download_models
# Uses VISION_ENABLED and VISION_MODEL from .env; originals are already saved.
# VISION_ENABLED=true enables analysis of all image/vector-drawing pages.
python -m scripts.reindex_documents
# Or select one document hash (the PDF basename under storage/documents):
python -m scripts.reindex_documents --document-id <full-sha256-id>
```

Run only one indexing operation at a time. This command fails explicitly on an
error; run it again to reuse completed visual pages and retry the failed one.
While the library lock is held, uploads, processing, reindexing and deletion are
disabled. A competing request fails immediately with a helpful busy message
instead of waiting two minutes for a file-lock timeout. Do not delete
`library.lock` to bypass an active operation; the operating system releases the
lock when its owning process exits.
An actively held lock disables search and conflicting actions. A stale unfinished
generation **without an active lock does not block keyword search**. Use
**Refresh indexing status** after an operation finishes. The UI does not resume
bulk jobs; explicitly rerun the CLI if desired. UI **Reindex** extracts text only.

The page image defaults to a maximum edge of 1536 pixels. Tiny screenshot text,
complex charts, math and ambiguous arrows can still be misread. This is not
guaranteed visual comprehension or a structural CAD/table parser. Increasing
`VISION_MAX_IMAGE_EDGE` (up to 2048) costs memory and requires another explicit
page analysis to update a saved result; inspect
the original when a detail matters. A saved description is not exhaustive:
details it omitted cannot later be recovered by text-only answering.

`VISION_ENABLED=false` is the default. It controls the advanced CLI, not the
explicit page-analysis button. Keyword search reads the latest completed
generation per document across profiles plus the newest saved visual analysis per
page. No model profile or checkbox needs to match to search this saved text.

### Selectable text and optional OCR

Physical PDF pages start at **1**, including covers and front matter. PDF page
labels are shown separately when available; printed numbers inside an image or
body text are not inferred. Chunks stay within a single physical page.

Low-text pages generate explicit warnings. Without OCR, searchable pages are
indexed and unreadable ones remain visible in the document's warnings. An
entirely unreadable PDF is retained but not text-indexed. Enable OCR and
**Reindex**, or open a page and explicitly analyze its images. Uploading bytes
already indexed under the current processing profile does not redo processing.

For Windows OCR:

1. Install [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html);
   its documentation links Windows installers.
2. Install the required language data (`eng.traineddata` for English).
3. If auto-discovery fails, set this in `.env` and restart Streamlit:

   ```dotenv
   TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
   OCR_LANGUAGE=eng
   ```

4. Enable **OCR pages with little selectable text** and reindex.

On Ubuntu, install `tesseract-ocr` and `tesseract-ocr-eng` with your package manager;
on macOS use `brew install tesseract`. Other languages need their corresponding
traineddata and an `OCR_LANGUAGE` value such as `eng+deu`.

OCR runs locally through PyMuPDF/Tesseract, page by page. It is slower and less
reliable than selectable text. The low-text detector does not guarantee finding
every image-only region on a page containing substantial selectable text.
Explicit page image analysis works independently of that detector.
Dedicated OCR on a copy is also an option for improving verbatim recognition.
Complex layout, tables and diagrams are not structurally reconstructed; visual
descriptions remain interpretations. The original PDF remains the authority.

## 3. Search, answers and safety

1. SHA-256 of the uploaded bytes identifies the document and detects duplicates.
   Original filenames are display metadata; disk filenames use only validated hashes.
2. PyMuPDF extracts text incrementally. Overlapping chunks preserve paragraphs,
   sentences and physical pages. The UI writes SQLite text only, without creating
   Chroma vectors or loading a tokenizer model.
3. BM25 searches the latest complete index for each document and any saved page
   interpretations. Common question words are ignored. Exact identifiers such as
   `E104`, quoted phrases and explicitly named documents receive priority; use the
   document filter for strict scope.
4. Keyword search is lexical, not semantic: synonyms and omitted image details
   may not match. With no hits, try a different term or inspect a relevant page;
   the app never falls back to AI automatically.
5. Each question is independent. Include the relevant subject/identifier in a
   follow-up; there is no automatic model-based conversation rewrite.
6. An explicit explanation uses only the top displayed hits, revalidated against
   current search results. Whole excerpts fit a conservative context budget. Structured
   claims must reference retrieved source IDs; citations are generated from stored
   filenames/pages, not model-supplied page numbers.
7. Optional claim verification asks the local model to check support again.
   Unsupported/invalid claims are withheld with visible warnings. This heuristic
   uses a fallible model, not an independent factual oracle.
8. An optional AI explanation with insufficient evidence produces:
   **"I could not find sufficient information in the uploaded documents to answer
   this question."**

Retrieved text is explicitly treated as untrusted data, not instructions. Model
outputs are not executed, and no tools are exposed to the model. Only a loopback
Ollama endpoint is accepted; cloud-backed Ollama models are rejected. Do not
install a custom model/template or local proxy that forwards requests remotely.

### Creating new drafts from document evidence

Creation is a separate, opt-in synthesis mode: it can combine documented building
blocks into new code, a checklist or other plain-text output instead of merely
explaining excerpts. It uses the same configured Ollama text model; no new model
download, whole-document analysis or automatic query-rewriting call is introduced.

For a PowerMill macro:

1. Search for relevant concepts, such as toolpaths, iteration and output commands.
   The search question can describe the desired macro, but keyword retrieval does
   not automatically discover every dependency or infer synonyms.
2. Inspect the results and select the passages that document the required syntax
   in **Passages for this draft**. The default is the top few search results;
   review these rather than assuming they cover the task.
3. In **What should I create?**, describe the goal, PowerMill version, inputs and
   constraints. For example: "Draft a read-only macro that lists toolpath names
   in the current project." Supply the actual version you use.
4. Click **Create from these documents (AI)**. A result contains either a draft
   with a cited **Documented basis**, assumptions/design choices, or questions
   about missing information. For clarification, edit the request to include
   your answers and submit explicitly again. Prior drafts/chat are not sent as
   factual evidence.

The model is instructed to use documented PowerMill syntax rather than inventing
commands or substituting Python/VBA. This is a prompt constraint, **not a compiler
or a guarantee**. Missing evidence should cause a request for more information;
an unsupported draft can still slip through a fallible model.

Code is displayed separately from its citations so references do not corrupt the
artifact. Use the code block's copy control or **Download draft (.txt)**. Downloads
preserve the returned content; the app does not execute it, write it into a
PowerMill project, or publish it as searchable source evidence. Review and test
in the target application using a copy of the project before relying on it.

Citations support documented premises, **not proof that the newly assembled
draft works**. When claim verification is enabled, it checks those premises; if
any fail, the entire draft is withheld. Invalid citations, malformed responses
and output truncation are reported rather than returning partial code.
Visual-source interpretations remain explicitly labeled and must be checked
against the originals.

Only the selected, still-current passages are eligible for the prompt, up to
the configured evidence count and context budget. Whole passages that do not fit
are omitted with warnings. Deleted, changed or out-of-scope results must be
searched again before another AI action. Requests are limited to 2048 UTF-8 bytes.
The ordinary output-token limit also applies to drafts: start with a small task,
and adjust it explicitly if needed rather than automatically increasing laptop load.

Each creation attempt is kept with its search results in the browser session.
**Export conversation** includes the requests, drafts, assumptions, questions,
cited basis and warnings. Export before restarting or clearing the session;
drafts are not automatically persisted in the document library.

### Lightweight chunks versus advanced hybrid retrieval

The normal UI uses passages bounded to **2400 UTF-8 bytes**, with up to 160 bytes
of overlap. This needs no model tokenizer; bytes are not tokens.

The advanced hybrid path combines normalized embeddings and BM25 using Reciprocal
Rank Fusion (`k=60`), with optional cross-encoder reranking. Its requested chunk
target is 650 tokens, but `all-MiniLM-L6-v2` supports only **256
wordpieces including special tokens**. The chunker caps the actual size using the
loaded tokenizer and supported limit. It never silently embeds truncated chunks.
To use larger chunks, explicitly select an appropriate longer-context embedding
model, download it and reindex. A token limit is not a character limit.

### Context and answer limits

Default context is 8192 tokens with 1024 reserved for output. The client checks the
installed model's advertised context and budgets UTF-8 bytes with extra template
overhead rather than assuming a universal characters-per-token ratio. This is
deliberately conservative for the supported Qwen byte-tokenizer family; custom
models/templates require their own validation. Oversized input is rejected or
whole evidence passages/history turns are omitted with warnings; output truncation
is reported. The exact Ollama tokenizer is not invoked locally by the app.

BM25, reranker scores, cosine similarity and RRF are ranking signals, **not calibrated
confidence or an answerability threshold**. An unrelated nearest neighbor is not
evidence that the answer exists.

## 4. Configuration and document management

Copy `.env.example` to `.env` and restart after editing it. The sidebar exposes
model name, context, output length, temperature, keyword result/evidence counts,
AI CPU threads and claim verification. Defaults are 20 results and up to 6
AI evidence excerpts. Reranking is an advanced hybrid feature, not used by the UI.
You can restrict search to selected documents; changing the scope clears the
conversation to avoid carrying context from another document set.

- **Delete:** removes the original and its chunks, metadata and vectors across all
  model profiles. It requires a checkbox confirmation.
- **Reindex:** re-extracts the original as model-free text with the selected OCR setting.
  Failed reindexing leaves the previous successful generation searchable.
- **Delete all documents:** requires separate explicit confirmation and clears chat.
- **Clear conversation:** clears this browser session's chat, drafts and source excerpts.
  Chat is not persisted automatically; export it as JSON if desired.

For advanced hybrid use, model revision, normalized embedding configuration, chunk and vision configuration select
an isolated Chroma collection. New model profiles never search old vectors.
Reindex the documents for the new profile. Returning to an unchanged earlier
profile can reuse its index. Document deletion removes every profile.

Index updates stage a new generation, then atomically publish it in SQLite.
Partial generation chunks remain invisible; recovery removes them for the active
profile on manager initialization or indexing. Successful per-page visual cache
entries remain independently searchable. Interrupted deletions are hidden
and retryable. A file lock serializes writers and retrieval snapshots. This is
not a multi-user server: run one Streamlit process per library.

### Where local data lives

| Path | Contents |
| --- | --- |
| `storage/documents/` | Original PDFs, named by content hash |
| `storage/chroma/` | Advanced hybrid vector collections; not created by keyword-only ingestion |
| `storage/metadata/catalog.sqlite3` | Metadata, page chunks, saved visual analyses and index generation journal |
| `models/` | Hugging Face embedding/reranker cache |
| Ollama's own model directory | Quantized language model files |

These folders and `.env` are git-ignored. No uploaded documents or model weights
should be committed. Streamlit and Hugging Face telemetry are disabled and Chroma
uses local storage with anonymized telemetry disabled.

For a fresh library use the confirmed sidebar deletion. To remove all physical
database files too, stop Streamlit and delete **only this project's `storage`
folder** using Explorer; optionally delete this project's `models` cache.
Logical database deletion is not secure erasure of disk blocks, WALs or backups.
Exported chats may contain excerpts and must be deleted separately. The app has
no authentication or encryption at rest; use OS permissions/full-disk encryption
and do not expose its port or Ollama's port to a network.

## 5. Tests and repeatable evaluation

Basic tests generate their own small PDFs and mock neural/LLM responses; they
do not download large models or require an Ollama server. They use real PyMuPDF,
SQLite and Chroma, including a 501-page indexing test, duplicate/restart checks,
failure recovery, deletion, filters and real vector queries with deterministic
test embeddings. Keyword-first tests prohibit model/network calls during ordinary
search, verify text-only reindexing, and check explicit AI buttons and one-page
cache reuse. Creation tests cover selected evidence, code/text drafts, clarification,
strict response validation, withheld drafts, inert rendering and export using
mock responses. They do not establish real PowerMill syntax correctness or the
installed model's drafting quality. Live model tests are excluded by default.

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q app.py config core scripts tests
python -m pip check
```

Optional advanced checks (these load models and may stress a low-memory laptop):

```powershell
# Actual embedding + cross-encoder tests, with network connections blocked:
python -m pytest -m models tests\test_models.py

# Actual local vision on a generated diagram with known labels/direction:
# Requires a running Ollama with qwen3-vl:4b-instruct.
python -m pytest -m vision tests\test_live_vision.py

# Generate original synthetic examples you can upload in the UI:
python -m scripts.create_examples
# Include an image-only diagram to test visual indexing:
python -m scripts.create_examples --visual

# Isolated synthetic retrieval evaluation; does not touch your library:
python -m scripts.evaluate

# Also generate real answers; requires a running Ollama and the selected model:
python -m scripts.evaluate --answers
```

The generated examples live in `examples/generated/`. Expected questions, facts,
document names and physical pages are in `examples/questions.json`.
The optional `Flow_Diagram.pdf` contains no selectable text. Open its page and
explicitly analyze it, then search **"Which box contains the label E104?"** and
request an explanation: the answer should identify
the INLET box, cite physical PDF page 1, and label its source as a visual
interpretation. The arrow points from that box toward OUTLET.

| Check | Expected evidence / behavior |
| --- | --- |
| Normal interval in Equipment Manual | 12 months, Equipment_Manual.pdf page 1 |
| Meaning of E104 | Inlet blockage, Equipment_Manual.pdf page 2 |
| Heavy conditions interval | 6 months, Equipment_Manual.pdf page 2 |
| Compare normal intervals | 12 vs 9 months; cite manual page 1 and update page 1, acknowledge conflict |
| Warranty duration | No keyword results; optional AI must not invent a duration |
| Follow-up: "Inspection interval under heavy conditions?" | Search manual page 2 without conversation rewriting |

Inspect each answer's actual cited excerpts and check that its claims follow from
them. The evaluation prints fact-presence/abstention and expected-source recall;
it does **not** prove entailment or conflict handling just by finding a keyword.
For your own collection, add representative questions (including unknown answers,
near-matching codes, scanned pages and conflicting editions), record the expected
pages manually, and measure retrieval recall, claim support and abstention rate.
Run evaluation again when models, chunking, OCR or retrieval settings change.

### Verification

- Run the basic suite above for current keyword-first behavior, Streamlit
  search/explicit-AI actions, preview and export using controlled model responses.
- Earlier advanced-path validation: the separate actual embedding/reranker test passed with network connections
  blocked. The synthetic retrieval evaluation found all expected sources in 4/4
  answerable cases.
- Ruff, Python compilation and `pip check` passed.
- The local Streamlit server and its health endpoint were exercised in a browser.
- Earlier live image-only PDF validation passed with Ollama 0.34.2 and
  `qwen3-vl:4b-instruct`: actual visual analysis, persisted visual chunks, retrieval,
  and an answer identifying the INLET box with a physical-page citation.
- Live synthetic text evaluation passed 5/5 fact/abstention cases after adding
  constrained JSON schemas and explicit document-scope instructions. Correct
  handling of these small fixtures does not establish real-world factual accuracy.
- The unqualified `qwen3-vl:4b` model failed structured output and was explicitly
  replaced with the user-approved Instruct variant, then removed from this machine.
- OCR availability/error paths were tested with mocks; the Tesseract command was
  not available on PATH, so live OCR accuracy was **not** evaluated.

## 6. Implementation map and phase checks

| Phase | Modules | What to verify |
| --- | --- | --- |
| Foundation | `config/settings.py`, `core/types.py`, `core/errors.py` | `test_settings.py`, environment check |
| PDF processing | `core/pdf_processor.py`, `core/text_chunker.py` | Extraction, labels, low-text warnings, OCR mocks and token limits |
| Visual understanding | `core/page_analysis.py`, `core/vision.py`, `core/visual_cache.py` | Explicit one-page requests, strict observations, saved results, retries and provenance |
| Indexing | `core/embeddings.py`, `core/catalog.py`, `core/vector_store.py`, `core/document_manager.py` | Batched embeddings, atomic publication, restart/duplicate/delete tests |
| Search | `core/keyword_retriever.py`, `core/keyword_search.py`; advanced `core/retriever.py`, `core/reranker.py` | No-model search, cache invalidation, identifier priority, filters; advanced RRF |
| AI integration | `core/llm.py`, `core/rag_pipeline.py` | Mock Ollama transport, budgets, fresh follow-up retrieval |
| Creation | `core/creation.py`, creation controls in `app.py` | Explicit synthesis, documented premises, missing-input questions, no execution, export |
| Interface | `app.py`, `.streamlit/config.toml` | Streamlit AppTest, upload/chat/source inspection in browser |
| Accuracy controls | `core/citations.py`, claim verification in pipeline | Invalid/uncited claims withheld, conflicts and missing-information tests |
| Evaluation | `tests/`, `scripts/evaluate.py`, synthetic examples | Reproducible fixture facts and known physical page references |
| Setup | `scripts/check_environment.py`, `scripts/download_models.py` | Clear dependency, model and server diagnostics |

`Catalog` is authoritative for visible generations; the keyword index is rebuilt
in memory from persisted chunks only when the generation/visual-cache snapshot changes.
PyMuPDF extraction is page-streamed and neural encoding is batched, but BM25 keeps
the active text corpus in RAM and Streamlit buffers uploaded files. Multiple
500-page text PDFs are practical; a huge image-heavy archive may not be. Upload
in small batches and monitor Task Manager. The app caps uploads at 200 MB/file;
larger settings also require changing Streamlit's `server.maxUploadSize`.

The typed page/chunk/retrieval/model boundaries support future file readers and
model backends. Accounts, cloud inference, spreadsheet/Word support, advanced
structural table extraction and multi-user deployments are deliberately
not implemented.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| `streamlit` or modules not found | Select/activate `.venv`, or use its Python with `-m streamlit` |
| Ollama connection refused | Open Ollama or run `ollama serve`; use Check Ollama connection |
| Model missing | Run the exact suggested `ollama pull <model>` while online |
| Embedding/reranker unavailable offline | Run `python -m scripts.download_models` once online with the same `.env` and cache path |
| High CPU / GPU memory pressure | Stay in keyword-only mode; avoid bulk vision. For optional AI, reduce threads/context/output or disable verification; thread limits are not a CPU cap |
| Input/context too large | Shorten the question, reduce evidence/output, or choose a supported larger context after checking RAM |
| JSON/claim validation failure | Retry with a narrower question; inspect excerpts; a smaller model may be less reliable at structured output |
| Draft withheld / more details needed | Select passages documenting the required commands, supply missing details in the creation request, and explicitly retry |
| Draft output truncated | Start with a smaller task or explicitly increase Maximum answer tokens within your context/memory budget; partial code is not accepted |
| No keyword results | Try a specific term, identifier or quoted phrase. Unanalyzed images and synonyms may not match; embedding settings do not affect UI keyword search |
| Scanned PDF has no text | Install Tesseract language data, enable OCR and reindex |
| Wrong table order / missing figure detail | Inspect the original page; optionally analyze that page's images. Interpretations can be wrong |
| Slow visual analysis | Analyze only a relevant page; reuse its saved result. Consider explicitly selecting `qwen3-vl:2b-instruct` |
| Encrypted PDF | Export an unencrypted copy using an authorized password and upload that copy |
| Reindex or delete interrupted | Retry operation; originals and last complete generations are preserved where applicable |
| Library busy / lock cannot be acquired | Another operation is running. Do not process the PDF again; wait, then select Refresh indexing status. Never delete the lock file to bypass it |
| HF symlink warning on Windows | Harmless; caching uses more disk space. Developer Mode can enable symlinks |
| Port 8501 is in use | Use `streamlit run app.py --server.port 8502`; keep the loopback address |

### API references and licensing

Implementation targets the pinned releases in `requirements.txt`. APIs were
checked against the [Sentence Transformers reference](https://sbert.net/docs/package_reference/sentence_transformer/model.html),
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[Chroma persistent client documentation](https://cookbook.chromadb.dev/core/clients/),
[PyMuPDF page API](https://pymupdf.readthedocs.io/en/latest/page.html),
and [Ollama chat](https://docs.ollama.com/api/chat) /
[model-details API](https://docs.ollama.com/api-reference/show-model-details).
Review model licenses before redistribution. PyMuPDF is AGPL/commercial
dual-licensed; assess its licensing obligations before distributing a derivative
application. Example documents in this project are newly authored synthetic data.