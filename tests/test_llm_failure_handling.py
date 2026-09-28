"""The fallback provider is a dependency: a failure must be visible, not disguised.

Before this, the Groq client caught every transport error and returned an apology
sentence, so an expired key, a rate limit and a bad model name all looked like a
successful reply with a normal latency.
"""

import json
from pathlib import Path

import pytest
import requests

from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import (
    LLM_UNAVAILABLE_REPLY,
    GroqLLMClient,
    LLMUnavailableError,
    MockLLMClient,
)
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor

MODEL_PATH = "models/baseline_intent.keras"
LABEL_MAP_PATH = Path("data/processed/label_map.json")
OOD_TEXT = "What is the best recipe for baking chocolate brownies?"


def _class_names() -> list:
    idx_to_label = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["idx_to_label"]
    return [idx_to_label[str(i)] for i in range(len(idx_to_label))]


def _router(llm_client) -> HybridRouter:
    return HybridRouter(
        model_path=MODEL_PATH,
        class_names=_class_names(),
        tool_executor=ToolExecutor(BankingBackendSimulator()),
        llm_client=llm_client,
    )


class AlwaysFailingClient(GroqLLMClient):
    """Transport always fails, without touching the network or waiting."""

    def __init__(self):
        super().__init__(api_key="test-key-not-used", retry_delay_s=0.0)

    def generate(self, user_text, conversation_history=None) -> str:
        raise LLMUnavailableError("simulated transport failure")


def test_client_raises_typed_error_and_logs_the_cause(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", boom)
    client = GroqLLMClient(api_key="test-key-not-used", retry_delay_s=0.0)

    with caplog.at_level("ERROR"):
        with pytest.raises(LLMUnavailableError):
            client.generate("hello")

    failures = [r for r in caplog.records if r.msg == "llm_call_failed"]
    assert failures, "the provider failure was not logged"
    assert failures[0].error_type == "ConnectionError"


def test_http_error_status_is_raised_not_disguised(monkeypatch):
    class FakeResponse:
        status_code = 401
        text = '{"error":{"message":"Invalid API Key"}}'
        headers = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("401 Client Error", response=self)

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse())
    client = GroqLLMClient(api_key="test-key-not-used", retry_delay_s=0.0)

    with pytest.raises(LLMUnavailableError):
        client.generate("hello")


def test_rate_limit_is_retried_then_reported(monkeypatch, caplog):
    """A 429 gets another attempt, and exhausting the retries is not silent."""
    calls = {"n": 0}

    class LimitedResponse:
        status_code = 429
        text = '{"error":{"message":"Rate limit reached on output tokens per minute"}}'
        headers = {"retry-after": "0"}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("429 Too Many Requests", response=self)

    def counting_post(*args, **kwargs):
        calls["n"] += 1
        return LimitedResponse()

    monkeypatch.setattr(requests, "post", counting_post)
    client = GroqLLMClient(api_key="test-key-not-used", retry_delay_s=0.0)

    with caplog.at_level("WARNING"):
        with pytest.raises(LLMUnavailableError):
            client.generate("hello")

    assert calls["n"] == GroqLLMClient.MAX_ATTEMPTS
    retries = [r for r in caplog.records if r.msg == "llm_call_retry"]
    assert len(retries) == GroqLLMClient.MAX_ATTEMPTS - 1
    assert any(r.msg == "llm_call_failed" for r in caplog.records)


def test_router_reports_provider_failure_as_its_own_route():
    router = _router(AlwaysFailingClient())
    response = router.process_message("cust_1", "sess_fail", OOD_TEXT)

    assert response.route_taken == "LLM_UNAVAILABLE"
    assert response.reply == LLM_UNAVAILABLE_REPLY


def test_router_failure_is_logged_as_an_error(caplog):
    router = _router(AlwaysFailingClient())
    with caplog.at_level("ERROR"):
        router.process_message("cust_1", "sess_fail_2", OOD_TEXT)

    assert any(r.msg == "llm_unavailable" for r in caplog.records)


def test_working_provider_path_is_unchanged():
    router = _router(MockLLMClient())
    response = router.process_message("cust_1", "sess_ok", OOD_TEXT)

    assert response.route_taken == "LLM_FALLBACK"
    assert response.reply != LLM_UNAVAILABLE_REPLY
