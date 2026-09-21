import json

import httpx
import pytest

from config.settings import Settings
from core.errors import GenerationError
from core.llm import OllamaClient, prompt_cost


def configured(**changes):
    return Settings(_env_file=None, **changes)


def model_details(**changes):
    return {
        "capabilities": ["completion"],
        "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 32768},
        **changes,
    }


def completion(**changes):
    return {
        "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": "A local answer."},
        **changes,
    }


class OllamaStub:
    def __init__(self, tags=None, show=None, chat=None):
        self.tags = {"models": [{"name": "qwen3:8b"}]} if tags is None else tags
        self.show = model_details() if show is None else show
        self.chat = completion() if chat is None else chat
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        payload = {"/api/tags": self.tags, "/api/show": self.show, "/api/chat": self.chat}
        return httpx.Response(200, json=payload[request.url.path])

    def client(self, **settings):
        return OllamaClient(configured(**settings), transport=httpx.MockTransport(self))


def test_lists_installed_names_and_checks_architecture_context():
    stub = OllamaStub(tags={"models": [{"name": "qwen3:8b"}, {"model": "tiny:latest"}, {"name": "qwen3:8b"}]})
    client = stub.client()
    try:
        assert client.list_models() == ["qwen3:8b", "tiny:latest"]
        assert client.check_available() is None
        assert stub.requests[-1].url.path == "/api/show"
        assert json.loads(stub.requests[-1].content) == {"model": "qwen3:8b"}
    finally:
        client.close()
    assert client._client.is_closed


def test_chat_nonstreaming_nonthinking_with_full_options():
    stub = OllamaStub(chat=completion(message={"role": "assistant", "content": '{"ok":true}'}))
    client = stub.client()
    messages = [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hello"}]
    try:
        assert client.chat(messages, max_tokens=200, json_mode=True) == '{"ok":true}'
        body = json.loads(stub.requests[-1].content)
        assert body == {
            "model": "qwen3:8b", "messages": messages, "stream": False, "think": False,
            "format": "json", "keep_alive": 0,
            "options": {"num_ctx": 8192, "num_predict": 200, "temperature": 0.1, "num_thread": 2},
        }
        assert all(request.url.host == "127.0.0.1" for request in stub.requests)
        assert client._client.follow_redirects is False
        assert client._client._trust_env is False
    finally:
        client.close()


def test_chat_defaults_and_rechecks_local_model_each_call():
    stub = OllamaStub()
    client = stub.client()
    try:
        for _ in range(2):
            assert client.chat([{"role": "user", "content": "Hello"}]) == "A local answer."
        assert [request.url.path for request in stub.requests] == ["/api/tags", "/api/show", "/api/chat"] * 2
        body = json.loads(stub.requests[-1].content)
        assert "format" not in body
        assert body["options"]["num_predict"] == 1024
    finally:
        client.close()


def test_latest_tag_alias_and_localhost_are_resolved_locally():
    stub = OllamaStub(tags={"models": [{"name": "tiny:latest"}]})
    client = stub.client(ollama_model="tiny", ollama_url="http://localhost:11434")
    try:
        client.chat([{"role": "user", "content": "Hello"}])
        assert json.loads(stub.requests[-1].content)["model"] == "tiny:latest"
        assert all(request.url.host == "127.0.0.1" for request in stub.requests)
    finally:
        client.close()


def test_ipv6_loopback_is_supported():
    stub = OllamaStub()
    client = stub.client(ollama_url="http://[::1]:11434")
    try:
        client.check_available()
        assert all(request.url.host == "::1" for request in stub.requests)
    finally:
        client.close()


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:11434", "http://example.com", "http://127.0.0.1.evil.test",
    "http://user:pass@127.0.0.1", "http://127.0.0.1/api", "http://127.0.0.1?x=1",
])
def test_bypassed_settings_cannot_enable_remote_endpoint(url):
    settings = configured().model_copy(update={"ollama_url": url})
    with pytest.raises(GenerationError, match="loopback"):
        OllamaClient(settings)


@pytest.mark.parametrize("exception", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout])
def test_offline_and_timeouts_give_serve_instructions(exception):
    def fail(request):
        raise exception("network unavailable", request=request)
    client = OllamaClient(configured(), transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(GenerationError, match="ollama serve"):
            client.check_available()
    finally:
        client.close()


def test_missing_model_gives_pull_and_serve_instructions_without_chat():
    stub = OllamaStub(tags={"models": []})
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="ollama pull qwen3:8b") as error:
            client.chat([{"role": "user", "content": "Private document"}])
        assert "ollama serve" in str(error.value)
        assert [request.url.path for request in stub.requests] == ["/api/tags"]
    finally:
        client.close()


@pytest.mark.parametrize("name", ["qwen3:8b-cloud", "CLOUD-alias:latest", "https://model.example/x"])
def test_cloud_model_names_are_rejected_before_network(name):
    stub = OllamaStub()
    client = stub.client(ollama_model=name)
    try:
        with pytest.raises(GenerationError, match="Cloud/remote"):
            client.check_available()
        assert not stub.requests
    finally:
        client.close()


@pytest.mark.parametrize("metadata", [
    {"remote_model": "qwen3:8b"},
    {"remote_host": "https://ollama.com"},
    {"details": {"remote_model": "hidden"}},
    {"details": {"parent_model": "model:cloud"}},
    {"cloud": True},
    {"modelfile": "FROM model:cloud\nPARAMETER num_ctx 8192"},
    {"modelfile": "FROM https://server.example/model"},
    {"remote_model": []},
    {"remote_host": 0},
])
def test_remote_alias_show_metadata_never_receives_documents(metadata):
    stub = OllamaStub(show=model_details(**metadata))
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="remote"):
            client.chat([{"role": "user", "content": "PRIVATE"}])
        assert all(b"PRIVATE" not in request.content for request in stub.requests)
        assert [request.url.path for request in stub.requests] == ["/api/tags", "/api/show"]
    finally:
        client.close()


def test_remote_tags_metadata_is_rejected_before_show():
    stub = OllamaStub(tags={"models": [{"name": "qwen3:8b", "remote_model": "other"}]})
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="remote"):
            client.check_available()
        assert len(stub.requests) == 1
    finally:
        client.close()


@pytest.mark.parametrize("info,match", [
    ({}, "context length"),
    ({"general.architecture": "qwen3", "llama.context_length": 32768}, "context length"),
    ({"qwen3.context_length": "32768"}, "context length"),
    ({"qwen3.context_length": True}, "context length"),
    ({"qwen3.context_length": -1}, "context length"),
    ({"qwen3.context_length": 4096}, "exceeds"),
])
def test_missing_invalid_and_too_small_context_limits(info, match):
    stub = OllamaStub(show=model_details(model_info=info))
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match=match):
            client.check_available()
    finally:
        client.close()


def test_context_from_specific_architecture_not_unrelated_vision_field():
    stub = OllamaStub(show=model_details(model_info={
        "general.architecture": "gemma4", "gemma4.context_length": 32768,
        "gemma4.vision.context_length": 256,
    }))
    client = stub.client()
    try:
        client.check_available()
    finally:
        client.close()


def test_noncompletion_model_is_not_accepted():
    stub = OllamaStub(show=model_details(capabilities=["embedding"]))
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="completion"):
            client.check_available()
    finally:
        client.close()


@pytest.mark.parametrize("body", [[], {}, {"models": {}}, {"models": [None]}, {"models": [{"name": 7}]}])
def test_malformed_models_json_is_actionable(body):
    stub = OllamaStub(tags=body)
    client = stub.client()
    try:
        with pytest.raises(GenerationError):
            client.list_models()
    finally:
        client.close()


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
def test_http_errors_are_surfaced(status):
    client = OllamaClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(status, json={"error": "model could not load"})
    ))
    try:
        with pytest.raises(GenerationError, match=f"HTTP {status}") as error:
            client.check_available()
        assert "model could not load" in str(error.value)
    finally:
        client.close()


def test_redirects_are_not_followed():
    calls = []
    def redirect(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://remote.example/api/chat"})
    client = OllamaClient(configured(), transport=httpx.MockTransport(redirect))
    try:
        with pytest.raises(GenerationError, match="Redirects are disabled"):
            client.check_available()
        assert len(calls) == 1
    finally:
        client.close()


def test_malformed_wire_json_is_surfaced():
    client = OllamaClient(configured(), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=b'{"models":')
    ))
    try:
        with pytest.raises(GenerationError, match="malformed JSON"):
            client.check_available()
    finally:
        client.close()


@pytest.mark.parametrize("body,match", [
    ({"error": "runner failed"}, "runner failed"),
    (completion(done=False), "incomplete"),
    (completion(done_reason="length"), "truncated"),
    (completion(done_reason="max_tokens"), "truncated"),
    (completion(done_reason="unload"), "unexpectedly"),
    (completion(done_reason=[]), "unexpectedly"),
    (completion(message={"role": "assistant", "content": ""}), "empty or malformed"),
    (completion(message={"role": "assistant", "content": "  "}), "empty or malformed"),
    (completion(message={"role": "assistant", "content": 42}), "empty or malformed"),
    (completion(message={"role": "user", "content": "no"}), "empty or malformed"),
    (completion(message={"role": "assistant", "thinking": "secret"}), "empty or malformed"),
    (completion(message={"role": "assistant", "content": "x", "tool_calls": [{}]}), "empty or malformed"),
    (completion(done_reason=None, eval_count=1024), "truncated"),
])
def test_invalid_incomplete_or_truncated_chat_responses(body, match):
    stub = OllamaStub(chat=body)
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match=match):
            client.chat([{"role": "user", "content": "Hello"}])
    finally:
        client.close()


def test_json_mode_rejects_malformed_content():
    stub = OllamaStub()
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="malformed JSON content"):
            client.chat([{"role": "user", "content": "Hello"}], json_mode=True)
    finally:
        client.close()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_invalid_output_limit_makes_no_requests(value):
    stub = OllamaStub()
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="positive integer"):
            client.chat([{"role": "user", "content": "Hello"}], max_tokens=value)
        assert not stub.requests
    finally:
        client.close()


@pytest.mark.parametrize("text", ["a" * 8000, "\U0001f9e0" * 2000, "\u6f22" * 2700])
def test_conservative_budget_rejects_before_any_requests(text):
    stub = OllamaStub()
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="output reserve"):
            client.chat([{"role": "user", "content": text}])
        assert not stub.requests
    finally:
        client.close()


def test_prompt_cost_is_byte_based_and_accounts_for_message_template():
    ascii_message = [{"role": "user", "content": "a"}]
    unicode_message = [{"role": "user", "content": "\U0001f9e0"}]
    assert prompt_cost(unicode_message) == prompt_cost(ascii_message) + 3
    assert prompt_cost(ascii_message * 2) >= prompt_cost(ascii_message) + 128
    assert prompt_cost(ascii_message) >= 640


def test_large_reported_template_is_budgeted_before_document_request():
    stub = OllamaStub(show=model_details(template="x" * 8000))
    client = stub.client()
    try:
        with pytest.raises(GenerationError, match="output reserve"):
            client.chat([{"role": "user", "content": "PRIVATE"}])
        assert [request.url.path for request in stub.requests] == ["/api/tags", "/api/show"]
    finally:
        client.close()


def test_alias_changed_to_remote_between_generations_blocks_document_send():
    stub = OllamaStub()
    client = stub.client()
    try:
        client.chat([{"role": "user", "content": "First question"}])
        stub.show["remote_model"] = "remote-target"
        with pytest.raises(GenerationError, match="remote"):
            client.chat([{"role": "user", "content": "PRIVATE second question"}])
        assert sum(request.url.path == "/api/chat" for request in stub.requests) == 1
        assert all(b"PRIVATE" not in request.content for request in stub.requests)
    finally:
        client.close()


@pytest.mark.parametrize("endpoint", ["/api/show", "/api/chat"])
def test_show_and_chat_http_failures_are_not_abstentions(endpoint):
    stub = OllamaStub()
    def fail(request):
        if request.url.path == endpoint:
            return httpx.Response(500, json={"error": "inference runner crashed"})
        return stub(request)
    client = OllamaClient(configured(), transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(GenerationError, match="inference runner crashed"):
            client.chat([{"role": "user", "content": "Question"}])
    finally:
        client.close()


@pytest.mark.parametrize("messages", [
    [], [{"role": "tool", "content": "x"}], [{"role": "user", "content": 7}],
    [{"role": "user", "content": "x", "images": ["bad"]}],
    [{"role": "user", "content": "\ud800"}],
])
def test_invalid_messages_are_rejected(messages):
    with pytest.raises(GenerationError):
        prompt_cost(messages)
