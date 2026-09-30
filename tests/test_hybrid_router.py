import json
from pathlib import Path
import pytest
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import MockLLMClient
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor

MODEL_PATH = "models/baseline_intent.keras"
LABEL_MAP_PATH = Path("data/processed/label_map.json")


@pytest.fixture
def class_names():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    return [idx_to_label[str(i)] for i in range(len(idx_to_label))]


@pytest.fixture
def router(class_names):
    backend = BankingBackendSimulator()
    executor = ToolExecutor(backend)
    mock_llm = MockLLMClient()
    return HybridRouter(
        model_path=MODEL_PATH,
        class_names=class_names,
        tool_executor=executor,
        llm_client=mock_llm,
    )


def test_direct_tool_execution_flow(router):
    # User 1001 checks bill balance directly in a single turn
    response = router.process_message(
        user_id="cust_1001",
        session_id="session_tool_1",
        text="What is my current bill balance?",
    )

    assert response.route_taken == "TOOL"
    assert response.intent == "bill_balance"
    assert response.tool_result is not None
    assert response.tool_result.success is True
    assert "Your credit balance is" in response.reply


def test_multi_turn_clarification_flow(router):
    session_id = "session_clarify_1"

    # Turn 1: User expresses intent without dollar amount
    res1 = router.process_message(
        user_id="cust_1001",
        session_id=session_id,
        text="I need to pay my bill",
    )
    assert res1.route_taken == "CLARIFICATION"
    assert res1.intent == "pay_bill"
    assert "How much would you like to pay" in res1.reply

    # Turn 2: User provides amount
    res2 = router.process_message(
        user_id="cust_1001",
        session_id=session_id,
        text="Pay $75.00 please",
    )
    assert res2.route_taken == "TOOL"
    assert res2.intent == "pay_bill"
    assert res2.tool_result.success is True
    assert "Successfully processed payment" in res2.reply


def test_out_of_domain_routes_to_llm_fallback(router):
    # Tier A query (recipes)
    response = router.process_message(
        user_id="cust_1001",
        session_id="session_ood_1",
        text="What is the best recipe for baking chocolate brownies?",
    )

    assert response.route_taken == "LLM_FALLBACK"
    assert "banking assistant" in response.reply.lower()
    assert response.tool_result is None


def test_unsupported_banking_routes_to_llm_fallback(router):
    # Tier B query (mortgage refinance)
    response = router.process_message(
        user_id="cust_1001",
        session_id="session_ood_2",
        text="Can I apply for a 30 year fixed mortgage refinance?",
    )

    assert response.route_taken == "LLM_FALLBACK"
    assert "mortgage" in response.reply.lower() or "1-800" in response.reply
