"""Model-provider adapters. No provider owns the scientific workflow."""

from __future__ import annotations

import hashlib
import json
import http.client
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .contracts import ModelResponse, ModelUsage, ToolCall


class ProviderError(RuntimeError):
    pass


class RetryableProviderError(ProviderError):
    """A transient provider response that is safe to retry unchanged."""


class ModelProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse: ...

    def audit_config(self) -> dict[str, Any]: ...


class ScriptedProvider:
    """Deterministic provider used for offline runtime tests."""

    name = "scripted"

    def __init__(self, responses: Sequence[ModelResponse], model: str = "scripted-v1") -> None:
        self.model = model
        self._responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        if not self._responses:
            raise ProviderError("Scripted provider has no remaining responses")
        return self._responses.popleft()

    def audit_config(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "paid": False}


class DatasetProfilerMockProvider:
    """Offline control provider that exercises the real tool loop.

    It is intentionally not an intelligence substitute. It asks for the dataset
    profile tool once, then reports that the deterministic artifact is ready.
    """

    name = "dataset-profiler-mock"
    model = "dataset-profiler-mock-v1"

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        del tools
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            user = next(
                (message for message in reversed(messages) if message.get("role") == "user"),
                {},
            )
            try:
                request = json.loads(str(user.get("content", "{}")))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Mock provider expected a JSON user request: {exc}") from exc
            return ModelResponse(
                model=self.model,
                tool_calls=(
                    ToolCall(
                        id="mock-inspect-1",
                        name="inspect_dataset",
                        arguments={
                            "dataset_root": request["dataset_root"],
                            "validation_path": request["validation_path"],
                        },
                    ),
                ),
            )
        last_tool = tool_messages[-1]
        result = json.loads(str(last_tool["content"]))
        if result.get("selected_adapter_id"):
            user = next(
                (message for message in reversed(messages) if message.get("role") == "user"),
                {},
            )
            request = json.loads(str(user.get("content", "{}")))
            return ModelResponse(
                model=self.model,
                tool_calls=(
                    ToolCall(
                        id="mock-profile-1",
                        name="profile_dataset",
                        arguments={
                            "dataset_id": request["dataset_id"],
                            "adapter_id": result["selected_adapter_id"],
                            "dataset_root": request["dataset_root"],
                            "validation_path": request["validation_path"],
                        },
                    ),
                ),
            )
        identity = result.get("dataset", {})
        volume = result.get("volume", {})
        return ModelResponse(
            model=self.model,
            content=(
                f"Dataset profiling completed: {identity.get('id', 'unknown')}; "
                f"{volume.get('subjects', '?')} subjects, "
                f"{volume.get('runs', '?')} runs, and {volume.get('trials', '?')} trials."
            ),
        )

    def audit_config(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "paid": False}


class SearchSpaceBuilderMockProvider:
    """Offline control provider for the search-space builder tool loop."""

    name = "search-space-builder-mock"
    model = "search-space-builder-mock-v1"

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        del tools
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if not tool_messages:
            user = next(
                (message for message in reversed(messages) if message.get("role") == "user"),
                {},
            )
            try:
                request = json.loads(str(user.get("content", "{}")))
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Mock provider expected a JSON user request: {exc}") from exc
            return ModelResponse(
                model=self.model,
                tool_calls=(
                    ToolCall(
                        id="mock-search-space-1",
                        name="build_search_space_draft",
                        arguments={
                            "dataset_profile_path": request["dataset_profile_path"],
                            "component_registry_path": request["component_registry_path"],
                        },
                    ),
                ),
            )
        draft = json.loads(str(tool_messages[-1]["content"]))
        dimensions = (draft.get("canonical_space") or {}).get("dimensions") or draft.get(
            "dimensions", {}
        )
        component_count = sum(len(items) for items in dimensions.values())
        return ModelResponse(
            model=self.model,
            content=(
                f"Search-space draft completed for {draft.get('dataset_id', 'unknown')}: "
                f"{len(dimensions)} dimensions and {component_count} applicable components. "
                "Frontier network discovery and an independent Dataset Critic are required next."
            ),
        )

    def audit_config(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "paid": False}


@dataclass
class OpenAICompatibleProvider:
    """Small stdlib adapter for compatible Chat Completions endpoints."""

    name: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    stream: bool = False
    progress_callback: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def deepseek(
        cls,
        *,
        model: str = "deepseek-v4-flash",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_output_tokens: int = 1024,
    ) -> "OpenAICompatibleProvider":
        return cls(
            name="deepseek",
            model=model,
            base_url="https://api.deepseek.com",
            api_key_env=api_key_env,
            extra_body={"max_tokens": max_output_tokens},
        )

    @classmethod
    def kimi(
        cls,
        *,
        model: str = "kimi-k3",
        api_key_env: str = "MOONSHOT_API_KEY",
        max_output_tokens: int | None = None,
        reasoning_effort: str = "low",
    ) -> "OpenAICompatibleProvider":
        if model.startswith("kimi-k3"):
            request_parameters: dict[str, Any] = {
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_output_tokens or 1024,
            }
        else:
            # K2.7 Code is always-thinking and uses max_tokens. Sending K3-only
            # reasoning_effort/max_completion_tokens parameters may be rejected.
            request_parameters = {"max_tokens": max_output_tokens or 16384}
        return cls(
            name="kimi",
            model=model,
            base_url="https://api.moonshot.cn/v1",
            api_key_env=api_key_env,
            extra_body=request_parameters,
            # Kimi tool calls support SSE across current model families. Always
            # stream so long reasoning/tool-argument responses do not depend on
            # one idle non-streaming HTTP connection surviving until completion.
            stream=True,
        )

    def audit_config(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "stream": self.stream,
            "request_parameters": dict(self.extra_body),
            "paid": True,
        }

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"Missing API credential in environment variable {self.api_key_env}"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "stream": self.stream,
        }
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self.extra_body)
        url = self.base_url.rstrip("/") + "/chat/completions"
        request_data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "bci-autodiscovery/0.1",
        }
        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._progress(
                "provider_attempt_started",
                {"attempt": attempt, "maximum_attempts": self.max_attempts},
            )
            try:
                # urllib Request/connection state is not reusable after a remote
                # disconnect on Windows. Build an independent request for every
                # retry so recovery never inherits a half-closed socket.
                request = urllib.request.Request(
                    url,
                    data=request_data,
                    headers=request_headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = (
                        self._read_stream(response)
                        if self.stream
                        else json.loads(response.read().decode("utf-8"))
                    )
                break
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise ProviderError(f"Provider HTTP {exc.code}: {details}") from exc
                last_error = ProviderError(f"Provider HTTP {exc.code}: {details}")
            except (
                RetryableProviderError,
                urllib.error.URLError,
                TimeoutError,
                ssl.SSLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
                OSError,
            ) as exc:
                last_error = exc
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Provider returned invalid JSON: {exc}") from exc
            if attempt < self.max_attempts:
                self._progress(
                    "provider_retry_scheduled",
                    {
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "error_type": type(last_error).__name__,
                    },
                )
                time.sleep(self.retry_backoff_seconds * attempt)
        if payload is None:
            raise ProviderError(
                f"Provider request failed after {self.max_attempts} attempts: {last_error}"
            ) from last_error

        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider response did not contain choices[0].message") from exc
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                arguments = {
                    "__provider_invalid_json_arguments__": {
                        "message": (
                            "The provider returned malformed tool arguments. Retry this "
                            "tool call with exactly one JSON object matching its schema."
                        ),
                        "raw_length": len(raw_arguments),
                        "raw_sha256": hashlib.sha256(
                            raw_arguments.encode("utf-8", errors="replace")
                        ).hexdigest(),
                        "json_error": exc.msg,
                    }
                }
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call-{len(calls)}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )
        return ModelResponse(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=tuple(calls),
            usage=ModelUsage.from_mapping(payload.get("usage")),
            model=payload.get("model") or self.model,
            provider_response_id=payload.get("id"),
        )

    def _progress(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_callback is not None:
            try:
                self.progress_callback(event, payload or {})
            except OSError:
                # Progress is observational only. A detached/expired terminal can
                # close stdout while the paid provider request is still valid;
                # losing that display channel must not fail the research run.
                self.progress_callback = None

    def _read_stream(self, response: Any) -> dict[str, Any]:
        """Aggregate OpenAI-compatible SSE, including fragmented parallel tool calls."""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        response_id: str | None = None
        response_model: str | None = None
        first_chunk = True
        done_received = False
        last_activity_notice = time.monotonic()
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done_received = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Provider stream returned invalid JSON: {exc}") from exc
            if first_chunk:
                self._progress("stream_opened", {"model": chunk.get("model") or self.model})
                first_chunk = False
            response_id = chunk.get("id") or response_id
            response_model = chunk.get("model") or response_model
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(str(delta["content"]))
                if delta.get("reasoning_content"):
                    reasoning_parts.append(str(delta["reasoning_content"]))
                for piece in delta.get("tool_calls") or []:
                    index = int(piece.get("index", 0))
                    current = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if piece.get("id"):
                        current["id"] = str(piece["id"])
                    if piece.get("type"):
                        current["type"] = str(piece["type"])
                    function = piece.get("function") or {}
                    if function.get("name"):
                        current["function"]["name"] += str(function["name"])
                    if function.get("arguments"):
                        current["function"]["arguments"] += str(function["arguments"])
            now = time.monotonic()
            if now - last_activity_notice >= 5.0:
                self._progress(
                    "stream_activity",
                    {
                        "reasoning_chars": sum(map(len, reasoning_parts)),
                        "content_chars": sum(map(len, content_parts)),
                        "tool_call_fragments": len(tool_calls),
                    },
                )
                last_activity_notice = now
        if first_chunk:
            raise RetryableProviderError("Provider stream ended before any data chunk")
        if not done_received:
            raise RetryableProviderError(
                "Provider stream ended before the [DONE] marker"
            )
        message: dict[str, Any] = {
            "content": "".join(content_parts),
            "reasoning_content": "".join(reasoning_parts) or None,
        }
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        self._progress(
            "stream_completed",
            {
                "content_chars": len(message["content"]),
                "reasoning_chars": len(message.get("reasoning_content") or ""),
                "tool_calls": len(tool_calls),
            },
        )
        return {
            "id": response_id,
            "model": response_model or self.model,
            "choices": [{"message": message}],
            "usage": usage,
        }
