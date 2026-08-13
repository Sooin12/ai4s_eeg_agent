from __future__ import annotations

import json
import io
import ssl
import urllib.error
from pathlib import Path

import pytest

from bci_autodiscovery.agents.audit import JsonlAuditSink
from bci_autodiscovery.agents.providers import OpenAICompatibleProvider, ProviderError


class _ProviderResponse:
    def __init__(self, value: object) -> None:
        self.buffer = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.buffer.read()


class _StreamingResponse:
    def __init__(self, chunks: list[dict], *, include_done: bool = True) -> None:
        lines = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
        if include_done:
            lines.append(b"data: [DONE]\n\n")
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_provider_fails_before_network_when_api_key_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = OpenAICompatibleProvider.deepseek()

    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        provider.complete(messages=[], tools=[])


def test_kimi_smoke_configuration_is_bounded_and_auditable() -> None:
    provider = OpenAICompatibleProvider.kimi(
        max_output_tokens=1024,
        reasoning_effort="low",
    )

    config = provider.audit_config()
    assert config["paid"] is True
    assert config["api_key_env"] == "MOONSHOT_API_KEY"
    assert config["request_parameters"] == {
        "reasoning_effort": "low",
        "max_completion_tokens": 1024,
    }
    assert config["stream"] is True
    assert config["max_attempts"] == 3


def test_provider_ignores_closed_progress_channel() -> None:
    notices = 0

    def closed_progress(_event, _payload) -> None:
        nonlocal notices
        notices += 1
        raise OSError(22, "Invalid argument")

    provider = OpenAICompatibleProvider.kimi(model="kimi-k3")
    provider.progress_callback = closed_progress

    provider._progress("provider_attempt_started", {"attempt": 1})
    provider._progress("provider_attempt_started", {"attempt": 2})

    assert notices == 1
    assert provider.progress_callback is None


def test_kimi_k27_uses_its_supported_always_thinking_parameters() -> None:
    provider = OpenAICompatibleProvider.kimi(model="kimi-k2.7-code")

    assert provider.audit_config()["request_parameters"] == {"max_tokens": 16384}


def test_provider_retries_transient_tls_eof(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary-test-key")
    attempts = 0

    def fake_open(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError(ssl.SSLError("unexpected EOF"))
        return _ProviderResponse(
            {
                "id": "ok",
                "model": "kimi-k2.7-code",
                "choices": [{"message": {"content": "ready"}}],
                "usage": {},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    provider = OpenAICompatibleProvider.kimi(model="kimi-k2.7-code")
    provider.stream = False
    provider.retry_backoff_seconds = 0
    response = provider.complete(messages=[], tools=[])
    assert attempts == 3
    assert response.content == "ready"


def test_provider_rebuilds_request_after_windows_socket_error(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary-test-key")
    attempts = 0
    request_ids: list[int] = []

    def fake_open(request, **_kwargs):
        nonlocal attempts
        attempts += 1
        request_ids.append(id(request))
        if attempts == 1:
            raise OSError(22, "Invalid argument")
        return _StreamingResponse(
            [{
                "id": "recovered",
                "model": "kimi-k3",
                "choices": [{"delta": {"content": "ready"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }]
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    provider = OpenAICompatibleProvider.kimi(model="kimi-k3")
    provider.retry_backoff_seconds = 0
    response = provider.complete(messages=[], tools=[])

    assert attempts == 2
    assert request_ids[0] != request_ids[1]
    assert response.content == "ready"


def test_kimi_stream_assembles_reasoning_and_fragmented_tool_calls(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary-test-key")
    chunks = [
        {
            "id": "stream-id",
            "model": "kimi-k2.7-code",
            "choices": [{"delta": {"reasoning_content": "checking "}}],
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_scholarly_metadata",
                                    "arguments": '{"query_id":',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"q1"}'},
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: _StreamingResponse(chunks)
    )
    provider = OpenAICompatibleProvider.kimi(model="kimi-k2.7-code")
    response = provider.complete(messages=[], tools=[])
    assert response.reasoning_content == "checking "
    assert response.tool_calls[0].name == "search_scholarly_metadata"
    assert response.tool_calls[0].arguments == {"query_id": "q1"}
    assert response.usage.total_tokens == 12


def test_kimi_retries_stream_that_ends_before_done_marker(monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary-test-key")
    incomplete_chunks = [
        {
            "id": "partial",
            "model": "kimi-k2.7-code",
            "choices": [{"delta": {"reasoning_content": "partial"}}],
        }
    ]
    complete_chunks = [
        {
            "id": "complete",
            "model": "kimi-k2.7-code",
            "choices": [{"delta": {"content": "ready"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    ]
    responses = [
        _StreamingResponse(incomplete_chunks, include_done=False),
        _StreamingResponse(incomplete_chunks, include_done=False),
        _StreamingResponse(complete_chunks),
    ]
    attempts = 0

    def fake_open(*_args, **_kwargs):
        nonlocal attempts
        response = responses[attempts]
        attempts += 1
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    provider = OpenAICompatibleProvider.kimi(model="kimi-k2.7-code")
    provider.retry_backoff_seconds = 0

    response = provider.complete(messages=[], tools=[])

    assert attempts == 3
    assert response.content == "ready"
    assert response.usage.total_tokens == 4


def test_provider_turns_malformed_tool_arguments_into_recoverable_call(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "temporary-test-key")
    chunks = [
        {
            "id": "malformed-tool",
            "model": "kimi-k2.7-code",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "bad-call",
                                "type": "function",
                                "function": {
                                    "name": "search_scholarly_metadata",
                                    "arguments": (
                                        '{"query_id":"q1"}'
                                        '{"source_name":"crossref"}'
                                    ),
                                },
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: _StreamingResponse(chunks),
    )
    provider = OpenAICompatibleProvider.kimi(model="kimi-k2.7-code")

    response = provider.complete(messages=[], tools=[])

    call = response.tool_calls[0]
    assert call.name == "search_scholarly_metadata"
    error = call.arguments["__provider_invalid_json_arguments__"]
    assert error["raw_length"] > 0
    assert len(error["raw_sha256"]) == 64
    assert "Retry" in error["message"]


def test_jsonl_audit_redacts_nested_credentials(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path, run_id="redaction")
    sink.record(
        "credential_test",
        {
            "api_key": "secret-value",
            "nested": {"Authorization": "Bearer secret", "safe": "visible"},
        },
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["payload"]["api_key"] == "<redacted>"
    assert event["payload"]["nested"]["Authorization"] == "<redacted>"
    assert event["payload"]["nested"]["safe"] == "visible"
