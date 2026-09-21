"""Fail-closed, loopback-only access to an already installed Ollama model.

Prompt accounting deliberately uses UTF-8 bytes, not characters / four. This
overestimates ordinary byte-fallback tokenizers, with additional template space;
it is not a measurement from the model's tokenizer. No model is downloaded here.
"""

import json
from urllib.parse import urlsplit, urlunsplit

import httpx

from config.settings import Settings
from core.errors import GenerationError

_TEMPLATE_RESERVE = 512
_MESSAGE_RESERVE = 128


def prompt_cost(messages: list[dict[str, str]]) -> int:
    """Conservative input-token budget, excluding the requested output reserve."""
    if not isinstance(messages, list) or not messages:
        raise GenerationError("Provide at least one chat message.")
    total = _TEMPLATE_RESERVE
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
        ):
            raise GenerationError("Chat messages must contain a text role and content only.")
        try:
            total += _MESSAGE_RESERVE + len(message["role"].encode("utf-8"))
            total += len(message["content"].encode("utf-8"))
        except UnicodeError as exc:
            raise GenerationError("Chat text contains invalid Unicode; remove it and retry.") from exc
    return total


def _remote_metadata(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if "remote" in normalized or "cloud" in normalized:
                if item is not None and item is not False and item != "":
                    return True
            if normalized in {"name", "model", "parent_model", "from"}:
                if isinstance(item, str) and "cloud" in item.lower():
                    return True
            if normalized == "modelfile" and isinstance(item, str):
                for line in item.splitlines():
                    if line.strip().upper().startswith("FROM "):
                        reference = line.strip()[5:].strip().lower()
                        if "cloud" in reference or reference.startswith(("http://", "https://")):
                            return True
            if isinstance(item, (dict, list)) and _remote_metadata(item):
                return True
    elif isinstance(value, list):
        return any(_remote_metadata(item) for item in value)
    return False


def _canonical_model(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def parse_chat_response(result: dict, *, output_tokens: int, json_mode: bool = False) -> str:
    """Reject incomplete, truncated, or malformed nonstreaming Ollama answers."""
    if result.get("done_reason") in ("length", "max_tokens", "max_length"):
        raise GenerationError("Ollama truncated the answer at the output limit. Increase MAX_OUTPUT_TOKENS or ask a narrower question.")
    if result.get("done") is not True:
        raise GenerationError("Ollama returned an incomplete answer (done is not true). Retry the question.")
    reason = result.get("done_reason")
    if reason is not None and reason != "stop":
        raise GenerationError(f"Ollama stopped unexpectedly ({reason!r}); no answer was accepted.")
    if reason is None and type(result.get("eval_count")) is int and result["eval_count"] >= output_tokens:
        raise GenerationError("Ollama may have truncated the answer at the output limit. Increase MAX_OUTPUT_TOKENS.")
    message = result.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        or message.get("tool_calls")
    ):
        if isinstance(message, dict) and message.get("thinking") and not message.get("content"):
            raise GenerationError(
                "Ollama returned an empty or malformed text answer (reasoning-only output). "
                "Reasoning is not a final answer and was not accepted. "
                "Use a compatible local instruct model or update Ollama; "
                "some thinking models cannot return final answers with structured output."
            )
        raise GenerationError("Ollama returned an empty or malformed text answer.")
    content = message["content"].strip()
    if json_mode:
        try:
            json.loads(content)
        except ValueError as exc:
            raise GenerationError("Ollama returned malformed JSON content. Retry with a narrower question.") from exc
    return content


class OllamaClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        # Revalidate even Settings.model_copy()/model_construct() cannot bypass this boundary.
        endpoint = urlsplit(settings.ollama_url)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.username
            or endpoint.password
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
        ):
            raise GenerationError("Ollama must use an HTTP loopback URL, such as http://127.0.0.1:11434.")
        try:
            port = endpoint.port
        except ValueError as exc:
            raise GenerationError("The local Ollama URL has an invalid port.") from exc
        # Pin localhost to a literal loopback address, avoiding DNS/proxy configuration.
        host = "[::1]" if endpoint.hostname == "::1" else "127.0.0.1"
        base_url = urlunsplit(("http", f"{host}:{port}" if port else host, "", "", ""))
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(120.0, connect=5.0),
        )
        self._template_extra = 0
        self._model_name = settings.ollama_model
        self._model_digest = ""
        self._model_capabilities: tuple[str, ...] = ()

    @property
    def model_name(self) -> str:
        """Installed tag selected by the latest availability check."""
        return self._model_name

    @property
    def model_digest(self) -> str:
        """Digest reported by the locally verified tag, if advertised."""
        return self._model_digest

    @property
    def model_capabilities(self) -> tuple[str, ...]:
        """Capabilities from the locally verified model's /show metadata."""
        return self._model_capabilities

    def close(self) -> None:
        self._client.close()

    def prompt_cost(self, messages: list[dict[str, str]]) -> int:
        """Include unusually large model templates/system defaults reported by /show."""
        return prompt_cost(messages) + self._template_extra

    def _request(self, method: str, path: str, **kwargs: object) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise GenerationError(
                "Local Ollama timed out. Check `ollama serve`, or select a smaller installed model."
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationError(
                "Cannot reach local Ollama. Start `ollama serve` and check OLLAMA_URL."
            ) from exc
        if response.is_redirect:
            raise GenerationError("Ollama returned a redirect. Redirects are disabled to keep documents local.")
        if response.is_error:
            detail = ""
            try:
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                    detail = f": {payload['error'][:300]}"
            except ValueError:
                pass
            hint = ""
            if response.status_code == 404:
                hint = f" Install the local model with `ollama pull {self.settings.ollama_model}`."
            raise GenerationError(f"Ollama {path} returned HTTP {response.status_code}{detail}.{hint}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationError(f"Ollama {path} returned malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise GenerationError(f"Ollama {path} returned a malformed JSON object.")
        if payload.get("error"):
            raise GenerationError(f"Ollama {path} error: {str(payload['error'])[:300]}")
        return payload

    def _models(self) -> list[dict]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise GenerationError("Ollama /api/tags did not return a models list.")
        for model in models:
            if not isinstance(model, dict):
                raise GenerationError("Ollama /api/tags returned malformed model metadata.")
            name = model.get("name", model.get("model"))
            if not isinstance(name, str) or not name.strip():
                raise GenerationError("Ollama /api/tags returned a model without a valid name.")
        return models

    def list_models(self) -> list[str]:
        """List installed tags; check_available separately rejects remote-backed tags."""
        return list(dict.fromkeys(model.get("name", model.get("model")) for model in self._models()))

    def check_available(self) -> None:
        self._model_digest = ""
        self._model_capabilities = ()
        name = self.settings.ollama_model
        if not isinstance(name, str) or not name.strip() or any(c.isspace() for c in name):
            raise GenerationError("Choose a valid installed Ollama model name.")
        if "cloud" in name.lower() or "://" in name:
            raise GenerationError("Cloud/remote Ollama models are prohibited. Select and pull a local model.")
        matching = [
            item for item in self._models()
            if _canonical_model(item.get("name", item.get("model"))) == _canonical_model(name)
        ]
        if not matching:
            raise GenerationError(
                f"Model {name!r} is not installed locally. Run `ollama pull {name}` "
                "during setup, then start `ollama serve`."
            )
        if any(_remote_metadata(item) for item in matching):
            raise GenerationError("The selected Ollama tag is cloud/remote-backed; choose a local model.")
        self._model_name = matching[0].get("name", matching[0].get("model"))
        details = self._request("POST", "/api/show", json={"model": self._model_name})
        if _remote_metadata(details):
            raise GenerationError("Ollama /api/show identifies a cloud/remote-backed model; documents were not sent.")
        capabilities = details.get("capabilities")
        if capabilities is not None and (
            not isinstance(capabilities, list) or "completion" not in capabilities
        ):
            raise GenerationError("The installed model does not advertise text completion. Choose a chat model.")
        info = details.get("model_info")
        if not isinstance(info, dict):
            raise GenerationError("Ollama did not advertise model context limits. Update Ollama or choose another local model.")
        architecture = info.get("general.architecture")
        if isinstance(architecture, str) and architecture:
            limits = [info.get(f"{architecture}.context_length")]
        else:
            limits = [value for key, value in info.items() if key.endswith(".context_length")]
        if not limits or any(type(limit) is not int or limit <= 0 for limit in limits):
            raise GenerationError("Ollama did not advertise a valid architecture context length. Update Ollama or choose another local model.")
        supported = min(limits)
        if self.settings.context_window > supported:
            raise GenerationError(
                f"CONTEXT_WINDOW={self.settings.context_window} exceeds this model's advertised "
                f"context limit ({supported}). Lower CONTEXT_WINDOW or choose a larger-context local model."
            )
        template = details.get("template", "")
        system = details.get("system", "")
        if not isinstance(template, str) or not isinstance(system, str):
            raise GenerationError("Ollama /api/show returned malformed template metadata.")
        try:
            self._template_extra = max(0, len(template.encode("utf-8")) - _TEMPLATE_RESERVE)
            self._template_extra += len(system.encode("utf-8"))
        except UnicodeError as exc:
            raise GenerationError("Ollama /api/show returned invalid Unicode in template metadata.") from exc
        digest = matching[0].get("digest")
        self._model_digest = digest if isinstance(digest, str) else ""
        if isinstance(capabilities, list):
            self._model_capabilities = tuple(item for item in capabilities if isinstance(item, str))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        """Generate locally; json_schema overrides json_mode's generic JSON format."""
        output_tokens = self.settings.max_output_tokens if max_tokens is None else max_tokens
        if type(output_tokens) is not int or output_tokens <= 0:
            raise GenerationError("max_tokens must be a positive integer.")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise GenerationError("json_schema must be a JSON schema object.")

        def check_budget() -> None:
            if self.prompt_cost(messages) + output_tokens > self.settings.context_window:
                raise GenerationError(
                    "Prompt plus output reserve exceeds CONTEXT_WINDOW. Shorten the question/history "
                    "or use fewer/smaller excerpts; alternatively select a supported larger context."
                )

        check_budget()
        # Recheck on every generation, including rewrites and verification. Local aliases
        # can change; never send documents merely because an earlier check succeeded.
        self.check_available()
        check_budget()
        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": 0,
            "options": {
                "num_ctx": self.settings.context_window,
                "num_predict": output_tokens,
                "temperature": self.settings.temperature,
                "num_thread": self.settings.inference_threads,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"
        result = self._request("POST", "/api/chat", json=payload)
        return parse_chat_response(
            result, output_tokens=output_tokens,
            json_mode=json_mode or json_schema is not None,
        )
