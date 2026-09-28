"""Dialogue State Tracker and Slot Filling Manager.

Manages multi-turn conversation state, accumulates extracted entities,
enforces intent-specific required slots, generates targeted clarification prompts,
and handles edge cases: intent switching, explicit cancellations, and state resets.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Dict, List, Optional, Set
from src.extraction.entity_extractor import ExtractedEntities


class DialogueStatus(str, Enum):
    READY_TO_EXECUTE = "READY_TO_EXECUTE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"


@dataclass
class IntentSlotSchema:
    intent: str
    required_slots: Set[str]
    any_of_slots: Optional[List[Set[str]]] = None
    clarification_prompts: Dict[str, str] = field(default_factory=dict)


INTENT_SCHEMAS: Dict[str, IntentSlotSchema] = {
    "pay_bill": IntentSlotSchema(
        intent="pay_bill",
        required_slots={"amount"},
        clarification_prompts={
            "amount": "How much would you like to pay towards your bill?",
        },
    ),
    "freeze_account": IntentSlotSchema(
        intent="freeze_account",
        required_slots=set(),
        any_of_slots=[{"card_brand", "last_four", "account_type"}],
        clarification_prompts={
            "any_of": "Which card or account would you like to freeze? (e.g., your Visa or card ending in 1234)",
        },
    ),
    "pin_change": IntentSlotSchema(
        intent="pin_change",
        required_slots={"pin"},
        clarification_prompts={
            "pin": "Please provide the new 4-digit PIN you would like to set.",
        },
    ),
    "order_status": IntentSlotSchema(
        intent="order_status",
        required_slots={"order_id"},
        clarification_prompts={
            "order_id": "Please provide your order number or tracking ID (e.g., #ORD-12345).",
        },
    ),
    "bill_balance": IntentSlotSchema(
        intent="bill_balance",
        required_slots=set(),
    ),
    "card_declined": IntentSlotSchema(
        intent="card_declined",
        required_slots=set(),
    ),
}


@dataclass
class SessionState:
    session_id: str
    active_intent: Optional[str] = None
    slots: Dict[str, any] = field(default_factory=dict)
    status: DialogueStatus = DialogueStatus.NEEDS_CLARIFICATION
    missing_slot: Optional[str] = None
    clarification_prompt: Optional[str] = None
    turn_count: int = 0


class DialogueStateTracker:
    # Explicit cancellation phrases that abort the active flow
    CANCEL_PATTERN = re.compile(
        r"\b(?:cancel|never\s*mind|stop|abort|forget\s*it|nevermind|don'?t\s*do\s*(?:that|it))\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def reset_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def update(
        self,
        session_id: str,
        predicted_intent: Optional[str],
        entities: ExtractedEntities,
        user_text: str = "",
    ) -> SessionState:
        """Update dialogue state with new turn input and check slot completeness."""
        session = self.get_or_create_session(session_id)
        session.turn_count += 1

        # 1. Edge Case: Explicit User Cancellation ("never mind", "cancel")
        # Only treat as a pure cancellation if there's no new predicted intent being switched to
        if user_text and self.CANCEL_PATTERN.search(user_text) and not predicted_intent:
            session.active_intent = None
            session.slots = {}
            session.status = DialogueStatus.CANCELLED
            session.missing_slot = None
            session.clarification_prompt = "Understood, I've cancelled that request. What else can I help you with?"
            return session

        # 2. Edge Case: Intent Switching
        # If user expresses a new supported intent with confidence, switch context cleanly
        if predicted_intent and predicted_intent in INTENT_SCHEMAS:
            if session.active_intent != predicted_intent:
                session.active_intent = predicted_intent
                session.slots = {}  # Discard previous intent's uncommitted slots

        if not session.active_intent:
            session.status = DialogueStatus.NEEDS_CLARIFICATION
            session.clarification_prompt = "How can I help you today?"
            return session

        schema = INTENT_SCHEMAS[session.active_intent]

        # 3. Accumulate extracted entities into session slots
        entity_dict = entities.masked_summary()
        if entities.pin:
            entity_dict["pin"] = entities.pin

        for key, val in entity_dict.items():
            if val is not None:
                session.slots[key] = val

        # 4. Check slot completeness against schema
        missing_required = schema.required_slots - set(session.slots.keys())
        if missing_required:
            missing = sorted(list(missing_required))[0]
            session.status = DialogueStatus.NEEDS_CLARIFICATION
            session.missing_slot = missing
            session.clarification_prompt = schema.clarification_prompts.get(
                missing, f"Please provide the {missing}."
            )
            return session

        # Check 'any-of' slot groups (e.g. at least one card identifier needed for freeze_account)
        if schema.any_of_slots:
            for group in schema.any_of_slots:
                if not (group & set(session.slots.keys())):
                    session.status = DialogueStatus.NEEDS_CLARIFICATION
                    session.missing_slot = "any_of"
                    session.clarification_prompt = schema.clarification_prompts.get(
                        "any_of", "Please provide account or card details."
                    )
                    return session

        # All slots satisfied!
        session.status = DialogueStatus.READY_TO_EXECUTE
        session.missing_slot = None
        session.clarification_prompt = None
        return session
