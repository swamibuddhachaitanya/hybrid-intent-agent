import pytest
from src.dialogue.state_tracker import DialogueStateTracker, DialogueStatus
from src.extraction.entity_extractor import EntityExtractor


@pytest.fixture
def tracker():
    return DialogueStateTracker()


@pytest.fixture
def extractor():
    return EntityExtractor()


def test_one_shot_pay_bill_ready_immediately(tracker, extractor):
    session_id = "test_user_1"
    text = "Please pay $100 towards my visa"
    entities = extractor.extract(text)

    state = tracker.update(session_id, predicted_intent="pay_bill", entities=entities)

    assert state.status == DialogueStatus.READY_TO_EXECUTE
    assert state.active_intent == "pay_bill"
    assert state.slots["amount"] == 100.0
    assert state.slots["card_brand"] == "visa"
    assert state.clarification_prompt is None


def test_multi_turn_pay_bill_clarification(tracker, extractor):
    session_id = "test_user_2"

    # Turn 1: User expresses intent without amount
    turn1_text = "I would like to pay my bill"
    entities1 = extractor.extract(turn1_text)
    state1 = tracker.update(session_id, predicted_intent="pay_bill", entities=entities1)

    assert state1.status == DialogueStatus.NEEDS_CLARIFICATION
    assert state1.active_intent == "pay_bill"
    assert state1.missing_slot == "amount"
    assert "How much would you like to pay" in state1.clarification_prompt

    # Turn 2: User provides amount in follow-up
    turn2_text = "Pay $45.50 please"
    entities2 = extractor.extract(turn2_text)
    # Notice predicted_intent can be None or unchanged on follow-up turn
    state2 = tracker.update(session_id, predicted_intent=None, entities=entities2)

    assert state2.status == DialogueStatus.READY_TO_EXECUTE
    assert state2.active_intent == "pay_bill"
    assert state2.slots["amount"] == 45.50
    assert state2.missing_slot is None


def test_freeze_account_requires_identifier(tracker, extractor):
    session_id = "test_user_3"

    # Turn 1: Freeze without stating which card
    turn1_text = "Please freeze my card immediately"
    entities1 = extractor.extract(turn1_text)
    state1 = tracker.update(
        session_id, predicted_intent="freeze_account", entities=entities1
    )

    assert state1.status == DialogueStatus.NEEDS_CLARIFICATION
    assert "Which card or account" in state1.clarification_prompt

    # Turn 2: User specifies card ending digits
    turn2_text = "The one ending in 9876"
    entities2 = extractor.extract(turn2_text)
    state2 = tracker.update(session_id, predicted_intent=None, entities=entities2)

    assert state2.status == DialogueStatus.READY_TO_EXECUTE
    assert state2.slots["last_four"] == "9876"


def test_zero_slot_intent_executes_immediately(tracker, extractor):
    session_id = "test_user_4"
    text = "What is my current bill balance?"
    entities = extractor.extract(text)

    state = tracker.update(
        session_id, predicted_intent="bill_balance", entities=entities
    )

    assert state.status == DialogueStatus.READY_TO_EXECUTE
    assert state.active_intent == "bill_balance"


def test_order_status_requires_order_id(tracker, extractor):
    session_id = "test_user_5"
    text = "Where is my order?"
    entities = extractor.extract(text)

    state = tracker.update(
        session_id, predicted_intent="order_status", entities=entities
    )

    assert state.status == DialogueStatus.NEEDS_CLARIFICATION
    assert state.missing_slot == "order_id"
    assert "order number or tracking ID" in state.clarification_prompt


def test_explicit_cancellation_aborts_flow(tracker, extractor):
    session_id = "test_cancel_user"

    # Turn 1: User starts pay_bill
    tracker.update(
        session_id,
        predicted_intent="pay_bill",
        entities=extractor.extract("I want to pay my bill"),
        user_text="I want to pay my bill",
    )

    # Turn 2: User aborts mid-flow
    turn2_text = "Actually never mind, cancel that"
    state2 = tracker.update(
        session_id,
        predicted_intent=None,
        entities=extractor.extract(turn2_text),
        user_text=turn2_text,
    )

    assert state2.status == DialogueStatus.CANCELLED
    assert state2.active_intent is None
    assert state2.slots == {}
    assert "cancelled that request" in state2.clarification_prompt


def test_intent_switching_mid_flow(tracker, extractor):
    session_id = "test_switch_user"

    # Turn 1: User starts pay_bill (missing amount)
    state1 = tracker.update(
        session_id,
        predicted_intent="pay_bill",
        entities=extractor.extract("I need to pay my bill"),
        user_text="I need to pay my bill",
    )
    assert state1.active_intent == "pay_bill"
    assert state1.status == DialogueStatus.NEEDS_CLARIFICATION

    # Turn 2: User changes their mind and switches to freeze_account
    switch_text = "Wait, forget that, please freeze my visa ending 4412"
    entities2 = extractor.extract(switch_text)
    state2 = tracker.update(
        session_id,
        predicted_intent="freeze_account",  # Classifier detected freeze_account!
        entities=entities2,
        user_text=switch_text,
    )

    # Active intent must cleanly switch to freeze_account, discarding pay_bill
    assert state2.active_intent == "freeze_account"
    assert state2.slots["card_brand"] == "visa"
    assert state2.slots["last_four"] == "4412"
    assert "amount" not in state2.slots
    assert state2.status == DialogueStatus.READY_TO_EXECUTE

