"""FastAPI integration tests for /health, /chat, and /session/reset endpoints."""

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.api.app import app_state, create_app
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import MockLLMClient
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor

MODEL_PATH = "models/baseline_intent.keras"
LABEL_MAP_PATH = Path("data/processed/label_map.json")


@pytest.fixture(scope="module")
def test_client():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    backend = BankingBackendSimulator()
    executor = ToolExecutor(backend)
    mock_llm = MockLLMClient()

    router = HybridRouter(
        model_path=MODEL_PATH,
        class_names=class_names,
        tool_executor=executor,
        llm_client=mock_llm,
    )

    app = create_app(router=router)
    with TestClient(app) as client:
        yield client


def test_health_check_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["encoder_name"] == "all-MiniLM-L6-v2"
    assert "uptime_seconds" in data


def test_chat_tool_execution_with_latency(test_client):
    payload = {
        "user_id": "cust_1001",
        "session_id": "sess_api_tool_1",
        "message": "What is my current bill balance?",
    }
    response = test_client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["route_taken"] == "TOOL"
    assert data["intent"] == "bill_balance"
    assert "Your credit balance is" in data["reply"]
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0
    assert "energy_score" in data
    assert "confidence_score" in data


def test_chat_multi_turn_conversation(test_client):
    session_id = "sess_api_multiturn_1"

    # Turn 1: User says "I need to pay my bill" (needs amount slot)
    payload_turn1 = {
        "user_id": "cust_1001",
        "session_id": session_id,
        "message": "I need to pay my bill",
    }
    res1 = test_client.post("/chat", json=payload_turn1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["route_taken"] == "CLARIFICATION"
    assert "How much would you like to pay" in data1["reply"]

    # Turn 2: Follow up with amount
    payload_turn2 = {
        "user_id": "cust_1001",
        "session_id": session_id,
        "message": "Pay $75.00 please",
    }
    res2 = test_client.post("/chat", json=payload_turn2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["route_taken"] == "TOOL"
    assert data2["intent"] == "pay_bill"
    assert "Successfully processed payment" in data2["reply"]


def test_chat_out_of_domain_llm_fallback(test_client):
    payload = {
        "user_id": "cust_1001",
        "session_id": "sess_api_ood_1",
        "message": "What is the best recipe for baking chocolate brownies?",
    }
    response = test_client.post("/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["route_taken"] == "LLM_FALLBACK"
    assert any(w in data["reply"].lower() for w in ["banking", "assistant", "account"])


def test_chat_validation_error_on_empty_fields(test_client):
    # Missing required message and empty user_id
    invalid_payload = {
        "user_id": "",
        "session_id": "sess_invalid",
    }
    response = test_client.post("/chat", json=invalid_payload)
    assert response.status_code == 422


def test_session_reset_endpoint(test_client):
    # First start a conversation that gets into NEEDS_CLARIFICATION
    session_id = "sess_to_reset"
    test_client.post(
        "/chat",
        json={
            "user_id": "cust_1001",
            "session_id": session_id,
            "message": "I need to pay my bill",
        },
    )

    # Now reset session
    reset_res = test_client.post(
        "/session/reset",
        json={"session_id": session_id},
    )
    assert reset_res.status_code == 200
    data = reset_res.json()
    assert data["status"] == "ok"
    assert data["session_id"] == session_id


def test_serve_ui_endpoint(test_client):
    res = test_client.get("/")
    assert res.status_code == 200
    assert "Hybrid Intent Support AI" in res.text
    assert "Decision Telemetry" in res.text
