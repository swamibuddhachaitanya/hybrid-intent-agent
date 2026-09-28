"""Every routing decision must be traceable, and customer text must not be logged.

The API record carries the outcome (route, latency, session fingerprint) and the
router record carries the decision inputs (energy score, threshold, is_mutation),
so a wrong action can be traced back to the score that allowed it.
"""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.observability.logger import JsonFormatter
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import MockLLMClient
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor

MODEL_PATH = "models/baseline_intent.keras"
LABEL_MAP_PATH = Path("data/processed/label_map.json")

TOOL_TEXT = "What is my current bill balance?"
OOD_TEXT = "What is the best recipe for baking chocolate brownies?"

# Distinctive enough that a leak into a log record is unambiguous.
SENTINEL_TEXT = "please freeze my visa ending in 4242 right now"


@pytest.fixture(scope="module")
def client():
    idx_to_label = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["idx_to_label"]
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    router = HybridRouter(
        model_path=MODEL_PATH,
        class_names=class_names,
        tool_executor=ToolExecutor(BankingBackendSimulator()),
        llm_client=MockLLMClient(),
    )
    with TestClient(create_app(router=router)) as test_client:
        yield test_client


def test_each_route_emits_a_request_record(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/chat", json={"user_id": "c1", "session_id": "log_tool", "message": TOOL_TEXT})
        client.post("/chat", json={"user_id": "c1", "session_id": "log_llm", "message": OOD_TEXT})

    requests = [r for r in caplog.records if r.msg == "chat_request"]
    assert len(requests) == 2
    routes = {r.route_taken for r in requests}
    assert routes == {"TOOL", "LLM_FALLBACK"}
    for record in requests:
        assert isinstance(record.latency_ms, float)
        assert len(record.session) == 12  # fingerprint, not the raw session id
        assert record.text_len > 0


def test_router_records_the_decision_inputs(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/chat", json={"user_id": "c1", "session_id": "log_dec", "message": TOOL_TEXT})

    decisions = [r for r in caplog.records if r.msg == "routing_decision"]
    assert decisions, "no routing decision was logged"
    record = decisions[0]
    assert record.route_taken == "TOOL"
    assert isinstance(record.energy_score, float)
    assert isinstance(record.threshold_used, float)
    assert isinstance(record.is_mutation, bool)


def test_raw_customer_text_is_not_logged_above_debug(client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/chat", json={"user_id": "c1", "session_id": "log_pii", "message": SENTINEL_TEXT})

    for record in caplog.records:
        if record.levelno < logging.INFO:
            continue
        serialized = json.dumps(record.__dict__, default=str)
        assert SENTINEL_TEXT not in record.getMessage()
        assert SENTINEL_TEXT not in serialized


def test_json_formatter_emits_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
    record.route_taken = "TOOL"
    record.energy_score = 12.5

    payload = json.loads(formatter.format(record))

    assert payload["msg"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["route_taken"] == "TOOL"
    assert payload["energy_score"] == 12.5
