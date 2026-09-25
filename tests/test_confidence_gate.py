import numpy as np
import pytest
from src.routing.confidence_gate import ConfidenceGate, RoutingDecision

CLASS_NAMES = [
    "bill_balance",  # Read
    "card_declined",  # Read
    "freeze_account",  # Write
    "order_status",  # Read
    "pay_bill",  # Write
    "pin_change",  # Write
]


@pytest.fixture
def gate():
    return ConfidenceGate(
        class_names=CLASS_NAMES,
        tau_read=10.27,
        tau_write=11.19,
    )


def test_high_energy_read_action_routes_to_tool(gate):
    # Class 0: bill_balance (Read action, threshold 10.27)
    # Large logit gives energy ~ 11.5
    logits = np.array([11.5, 0.0, -1.0, 0.5, -2.0, -1.0])
    decision = gate.decide(logits)

    assert decision.route == "TOOL"
    assert decision.intent == "bill_balance"
    assert decision.is_mutation is False
    assert decision.threshold_used == 10.27
    assert decision.energy_score > 10.27
    assert decision.reason == "sufficient_energy_in_distribution"


def test_high_energy_write_action_routes_to_tool(gate):
    # Class 4: pay_bill (Write action, threshold 11.19)
    logits = np.array([0.0, -1.0, -1.0, 0.0, 12.0, -2.0])
    decision = gate.decide(logits)

    assert decision.route == "TOOL"
    assert decision.intent == "pay_bill"
    assert decision.is_mutation is True
    assert decision.threshold_used == 11.19
    assert decision.energy_score > 11.19


def test_low_energy_query_routes_to_llm_fallback(gate):
    # Uniform / low confidence out-of-scope query
    logits = np.array([1.0, 0.8, 1.2, 0.9, 0.5, 0.4])
    decision = gate.decide(logits)

    assert decision.route == "LLM_FALLBACK"
    assert decision.reason == "insufficient_confidence_routed_to_fallback"


def test_asymmetric_boundary_read_accepted_write_rejected(gate):
    """
    Critical Test:
    Energy score is 10.70.
    10.27 (read threshold) <= 10.70 < 11.19 (write threshold).
    - If intent is Read ('order_status'), it MUST be accepted to TOOL.
    - If intent is Write ('freeze_account'), it MUST be rejected to LLM_FALLBACK.
    """
    # 1. Read intent with energy ~ 10.70
    logits_read = np.array([0.0, 0.0, 0.0, 10.70, 0.0, 0.0])
    decision_read = gate.decide(logits_read)
    assert decision_read.intent == "order_status"
    assert decision_read.is_mutation is False
    assert decision_read.route == "TOOL"

    # 2. Write intent with identical energy ~ 10.70
    logits_write = np.array([0.0, 0.0, 10.70, 0.0, 0.0, 0.0])
    decision_write = gate.decide(logits_write)
    assert decision_write.intent == "freeze_account"
    assert decision_write.is_mutation is True
    assert decision_write.route == "LLM_FALLBACK"


def test_batch_decide(gate):
    batch_logits = np.array(
        [
            [11.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Read -> TOOL
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Ambiguous -> LLM_FALLBACK
        ]
    )
    decisions = gate.decide_batch(batch_logits)
    assert len(decisions) == 2
    assert decisions[0].route == "TOOL"
    assert decisions[1].route == "LLM_FALLBACK"


def test_invalid_shape_raises_error(gate):
    with pytest.raises(ValueError):
        gate.decide(np.array([1.0, 2.0]))  # Too short
