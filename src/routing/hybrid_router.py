"""Master Hybrid Router combining Local NLU, Confidence Gating, Entity Slots, Deterministic Tools, and LLM Fallback."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import tensorflow as tf

from src.dialogue.state_tracker import DialogueStateTracker, DialogueStatus
from src.extraction.entity_extractor import EntityExtractor
from src.observability.logger import get_logger, session_fingerprint
from src.preprocessing.embedding_preprocessor import EmbeddingPreprocessor
from src.routing.confidence_gate import ConfidenceGate
from src.routing.llm_fallback import (
    LLM_UNAVAILABLE_REPLY,
    BaseLLMClient,
    LLMUnavailableError,
    MockLLMClient,
)
from src.tools.tool_executor import ToolExecutor, ToolResult

logger = get_logger(__name__)


@dataclass
class HybridResponse:
    reply: str
    route_taken: str  # "TOOL", "CLARIFICATION", "LLM_FALLBACK"
    intent: Optional[str]
    confidence_score: float
    energy_score: float
    slots: Dict[str, Any]
    tool_result: Optional[ToolResult] = None


class HybridRouter:
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        tool_executor: ToolExecutor,
        llm_client: Optional[BaseLLMClient] = None,
        tau_read: float = 10.2707,
        tau_write: float = 11.1946,
    ):
        self.class_names = class_names
        self.preprocessor = EmbeddingPreprocessor()
        self.model = tf.keras.models.load_model(model_path)
        self.gate = ConfidenceGate(
            class_names=class_names, tau_read=tau_read, tau_write=tau_write
        )
        self.extractor = EntityExtractor()
        self.tracker = DialogueStateTracker()
        self.executor = tool_executor
        self.llm = llm_client or MockLLMClient()

        # Cache final layer weights for logit extraction: z = h @ W + b
        self.final_dense = self.model.layers[-1]
        self.W, self.b = self.final_dense.get_weights()

    def _extract_logits(self, embedding: np.ndarray) -> np.ndarray:
        """Extract pre-softmax logits from single 1D embedding."""
        curr = tf.convert_to_tensor(embedding.reshape(1, -1), dtype=tf.float32)
        for layer in self.model.layers[:-1]:
            curr = layer(curr, training=False)
        features = curr.numpy()
        logits = np.dot(features, self.W) + self.b
        return logits.flatten()

    def _log_decision(self, gate_decision, session_id: str, text: str, route: str, **extra) -> None:
        """Record the inputs behind a routing decision.

        Raw text stays at DEBUG. The INFO record carries its length and a session
        fingerprint, which is enough to trace a decision later without storing what
        the customer typed.
        """
        fingerprint = session_fingerprint(session_id)
        logger.info(
            "routing_decision",
            extra={
                "route_taken": route,
                "intent": gate_decision.intent,
                "energy_score": round(gate_decision.energy_score, 4),
                "softmax_confidence": round(gate_decision.softmax_confidence, 4),
                "threshold_used": gate_decision.threshold_used,
                "is_mutation": gate_decision.is_mutation,
                "session": fingerprint,
                "text_len": len(text),
                **extra,
            },
        )
        logger.debug("routing_decision_text", extra={"session": fingerprint, "text": text})

    def process_message(
        self,
        user_id: str,
        session_id: str,
        text: str,
    ) -> HybridResponse:
        """Process incoming user turn through the complete hybrid architecture."""

        # 1. Embed text (384-dim)
        emb = self.preprocessor.transform([text], batch_size=1)[0]

        # 2. Extract logits and route via Confidence Gate
        logits = self._extract_logits(emb)
        gate_decision = self.gate.decide(logits)

        # Retrieve existing session if already in an active flow
        existing_session = self.tracker.get_or_create_session(session_id)
        has_active_intent = existing_session.active_intent is not None

        # 3. Branch: If Gate routes to LLM_FALLBACK
        # Exception: If user is in an active slot-filling session and provides slot entities without switching intents
        entities = self.extractor.extract(text)
        has_slot_entities = bool(entities.masked_summary())

        if gate_decision.route == "LLM_FALLBACK" and not (has_active_intent and has_slot_entities):
            self._log_decision(gate_decision, session_id, text, route="LLM_FALLBACK")
            try:
                llm_reply = self.llm.generate(text)
            except LLMUnavailableError:
                # The fallback provider is a dependency. Its failure is its own route so
                # it can be counted, rather than looking like a fast successful reply.
                logger.error(
                    "llm_unavailable",
                    extra={
                        "session": session_fingerprint(session_id),
                        "intent": gate_decision.intent,
                        "text_len": len(text),
                    },
                )
                return HybridResponse(
                    reply=LLM_UNAVAILABLE_REPLY,
                    route_taken="LLM_UNAVAILABLE",
                    intent=gate_decision.intent,
                    confidence_score=gate_decision.softmax_confidence,
                    energy_score=gate_decision.energy_score,
                    slots={},
                )
            return HybridResponse(
                reply=llm_reply,
                route_taken="LLM_FALLBACK",
                intent=gate_decision.intent,
                confidence_score=gate_decision.softmax_confidence,
                energy_score=gate_decision.energy_score,
                slots={},
            )

        # 4. Gate accepted (or ongoing slot-filling turn): Update state
        predicted_intent = (
            gate_decision.intent if gate_decision.route == "TOOL" else existing_session.active_intent
        )

        session_state = self.tracker.update(
            session_id=session_id,
            predicted_intent=predicted_intent,
            entities=entities,
            user_text=text,
        )

        # If user cancelled
        if session_state.status == DialogueStatus.CANCELLED:
            self._log_decision(gate_decision, session_id, text, route="CLARIFICATION", reason="cancelled")
            return HybridResponse(
                reply=session_state.clarification_prompt,
                route_taken="CLARIFICATION",
                intent=None,
                confidence_score=gate_decision.softmax_confidence,
                energy_score=gate_decision.energy_score,
                slots={},
            )

        # If missing required slots: Prompt user for clarification
        if session_state.status == DialogueStatus.NEEDS_CLARIFICATION:
            self._log_decision(
                gate_decision,
                session_id,
                text,
                route="CLARIFICATION",
                pending_intent=session_state.active_intent,
            )
            return HybridResponse(
                reply=session_state.clarification_prompt,
                route_taken="CLARIFICATION",
                intent=session_state.active_intent,
                confidence_score=gate_decision.softmax_confidence,
                energy_score=gate_decision.energy_score,
                slots=session_state.slots,
            )

        # 6. Status is READY_TO_EXECUTE: Execute Deterministic Tool
        slots = session_state.slots
        intent = session_state.active_intent
        tool_res: Optional[ToolResult] = None

        if intent == "pay_bill":
            tool_res = self.executor.tool_pay_bill(
                user_id=user_id,
                amount=slots.get("amount"),
                card_brand=slots.get("card_brand"),
                last_four=slots.get("last_four"),
            )
        elif intent == "bill_balance":
            tool_res = self.executor.tool_bill_balance(
                user_id=user_id,
                account_type=slots.get("account_type"),
            )
        elif intent == "freeze_account":
            tool_res = self.executor.tool_freeze_account(
                user_id=user_id,
                card_brand=slots.get("card_brand"),
                last_four=slots.get("last_four"),
                account_type=slots.get("account_type"),
            )
        elif intent == "pin_change":
            tool_res = self.executor.tool_pin_change(
                user_id=user_id,
                new_pin=slots.get("pin"),
                card_brand=slots.get("card_brand"),
                last_four=slots.get("last_four"),
            )
        elif intent == "card_declined":
            tool_res = self.executor.tool_card_declined(user_id=user_id)
        elif intent == "order_status":
            tool_res = self.executor.tool_order_status(
                user_id=user_id,
                order_id=slots.get("order_id"),
            )

        self._log_decision(
            gate_decision,
            session_id,
            text,
            route="TOOL",
            executed_intent=intent,
            tool_success=tool_res.success if tool_res else None,
        )

        # Once executed, reset session for next transaction
        self.tracker.reset_session(session_id)

        reply_text = (
            tool_res.message
            if tool_res
            else "Action completed, but no response details were generated."
        )

        return HybridResponse(
            reply=reply_text,
            route_taken="TOOL",
            intent=intent,
            confidence_score=gate_decision.softmax_confidence,
            energy_score=gate_decision.energy_score,
            slots=slots,
            tool_result=tool_res,
        )
