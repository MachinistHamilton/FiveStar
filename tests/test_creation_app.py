import json

import pytest
import streamlit as st
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from config.settings import PROJECT_ROOT
from core.document_manager import DocumentManager, delete_document
from core.errors import GenerationError
from core.llm import OllamaClient
from core.text_chunker import keyword_chunker


@pytest.fixture
def creation_app(settings, pdf_bytes, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("VISION_ENABLED", "false")
    monkeypatch.setenv("VERIFY_CLAIMS", "false")
    manager = DocumentManager(settings, None, keyword_chunker())
    result = manager.ingest(
        pdf_bytes("PRINT displays a text message.", "PRINT displays text in another example."),
        "synthetic-macro-guide.pdf",
    )
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=30)
    yield app, manager, result.document_id
    manager.close()


def click_create(app):
    return next(
        button for button in app.button if button.label == "Create from these documents (AI)"
    ).click().run()


def payload(**changes):
    return {
        "sufficient": True, "kind": "code", "content": 'PRINT = "Hello"\n',
        "basis": [{"text": "PRINT displays a text message.", "sources": [1]}],
        "assumptions": ["Hello is the requested greeting."], "questions": [],
    } | changes


def test_explicit_creation_uses_chosen_evidence_and_exports_draft(creation_app, monkeypatch):
    app, _manager, _identifier = creation_app
    calls = []
    closed = []
    downloads = {}
    original_download = st.download_button
    original_close = OllamaClient.close

    def capture(label, data, *args, **kwargs):
        downloads[label] = data
        return original_download(label, data, *args, **kwargs)

    def close(client):
        closed.append(True)
        original_close(client)

    def respond(_self, messages, **kwargs):
        calls.append((messages, kwargs))
        return json.dumps(payload())

    monkeypatch.setattr(st, "download_button", capture)
    monkeypatch.setattr("core.llm.OllamaClient.check_available", lambda _self: None)
    monkeypatch.setattr("core.llm.OllamaClient.chat", respond)
    monkeypatch.setattr("core.llm.OllamaClient.close", close)
    app.chat_input[0].set_value("PRINT").run()
    assert not app.exception
    assert calls == []
    displayed = app.session_state["messages"][0]["hits"]
    selected = displayed[-1]
    app.multiselect(key="draft_sources_0").set_value([selected.chunk.chunk_id])
    app.text_area(key="draft_request_0").set_value("Draft a read-only greeting macro using PRINT.")
    click_create(app)
    assert not app.exception
    assert not app.error
    assert len(calls) == 1
    assert len(closed) == 1
    supplied = json.loads(calls[0][0][1]["content"])
    assert supplied["request"] == "Draft a read-only greeting macro using PRINT."
    assert len(supplied["evidence"]) == 1
    assert supplied["evidence"][0]["chunk_id"] == selected.chunk.chunk_id
    assert app.code[0].value == payload()["content"].rstrip()
    assert any("not been run or tested" in item.value for item in app.warning)
    assert any("physical PDF page" in item.value for item in app.markdown)
    assert downloads["Download draft (.txt)"] == payload()["content"]
    exported = json.loads(downloads["Export conversation"])
    assert exported[0]["question"] == "PRINT"
    assert exported[0]["answer"] is None
    draft = exported[0]["drafts"][0]
    assert draft["request"] == supplied["request"]
    assert draft["content"] == payload()["content"]
    assert draft["basis"]["sources"][0]["hit"]["chunk"]["chunk_id"] == selected.chunk.chunk_id
    next(button for button in app.button if button.label == "Inspect PDF page").click().run()
    assert not app.exception
    assert app.get("image")
    assert len(calls) == 1
    click_create(app)
    assert len(calls) == 2
    assert len(app.session_state["messages"][0]["drafts"]) == 2
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    assert not app.exception
    assert not app.code
    assert not app.session_state["messages"]


@pytest.mark.parametrize("kind", ["code", "text"])
def test_drafts_are_displayed_as_inert_content_not_executed(creation_app, monkeypatch, kind):
    app, _manager, _identifier = creation_app
    content = "raise RuntimeError('Generated content must not execute')"
    monkeypatch.setattr("core.llm.OllamaClient.check_available", lambda _self: None)
    monkeypatch.setattr(
        "core.llm.OllamaClient.chat",
        lambda *_args, **_kwargs: json.dumps(payload(kind=kind, content=content)),
    )
    app.chat_input[0].set_value("PRINT").run()
    click_create(app)
    assert not app.exception
    assert not app.error
    displayed = app.code if kind == "code" else app.text
    assert any(item.value == content for item in displayed)


def test_clarification_does_not_offer_an_incomplete_code_download(creation_app, monkeypatch):
    app, _manager, _identifier = creation_app
    monkeypatch.setattr("core.llm.OllamaClient.check_available", lambda _self: None)
    monkeypatch.setattr(
        "core.llm.OllamaClient.chat",
        lambda *_args, **_kwargs: json.dumps(payload(
            sufficient=False, content="", basis=[], assumptions=[],
            questions=["Which PowerMill version is required?"],
        )),
    )
    app.chat_input[0].set_value("PRINT").run()
    click_create(app)
    assert not app.exception
    assert any("Which PowerMill version" in item.value for item in app.text)
    assert any("Edit the creation request" in item.value for item in app.info)
    assert not app.code
    assert not any(item.label == "Download draft (.txt)" for item in app.get("download_button"))


@pytest.mark.parametrize("invalid", ["empty", "too-long", "no-passages", "deleted", "race"])
def test_invalid_or_stale_draft_request_never_initializes_ai(
    creation_app, settings, monkeypatch, invalid
):
    app, _manager, identifier = creation_app
    app.chat_input[0].set_value("PRINT").run()

    def no_ai(*_args, **_kwargs):
        pytest.fail("Invalid/stale draft request must not initialize AI.")

    monkeypatch.setattr("core.llm.OllamaClient.__init__", no_ai)
    if invalid == "empty":
        app.text_area(key="draft_request_0").set_value("")
    elif invalid == "too-long":
        app.text_area(key="draft_request_0").set_value("x" * 2049)
    elif invalid == "no-passages":
        app.multiselect(key="draft_sources_0").set_value([])
    elif invalid == "deleted":
        delete_document(settings.data_dir, identifier)
    if invalid == "race":
        monkeypatch.setattr("core.library_lock.library_is_busy", lambda _data: False)
        with FileLock(str(settings.data_dir / "metadata" / "library.lock")):
            click_create(app)
        assert any("busy" in item.value for item in app.info)
    else:
        click_create(app)
        assert app.error
    assert not app.exception
    assert not app.session_state["messages"][0]["drafts"]


def test_busy_library_disables_creation_and_keeps_existing_results(creation_app, settings):
    app, _manager, _identifier = creation_app
    app.chat_input[0].set_value("PRINT").run()
    with FileLock(str(settings.data_dir / "metadata" / "library.lock")):
        app.run()
        assert not app.exception
        assert next(
            button for button in app.button if button.label == "Create from these documents (AI)"
        ).disabled
        assert app.session_state["messages"][0]["hits"]


@pytest.mark.parametrize("stage", ["connection", "generation", "invalid-json", "truncated"])
def test_creation_failures_are_visible_and_not_saved_as_drafts(creation_app, monkeypatch, stage):
    app, _manager, _identifier = creation_app

    def fail(*_args, **_kwargs):
        raise GenerationError(
            "Ollama truncated the answer." if stage == "truncated" else "Ollama is not running."
        )

    monkeypatch.setattr(
        "core.llm.OllamaClient.check_available", fail if stage == "connection" else lambda _self: None
    )
    monkeypatch.setattr(
        "core.llm.OllamaClient.chat",
        (lambda *_args, **_kwargs: "{}") if stage == "invalid-json" else fail,
    )
    app.chat_input[0].set_value("PRINT").run()
    click_create(app)
    assert not app.exception
    assert any("Creating a draft" in error.value for error in app.error)
    assert not app.session_state["messages"][0]["drafts"]
    assert app.session_state["messages"][0]["hits"]
