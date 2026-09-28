"""Pydantic V2 request and response schemas for the Hybrid Intent Chatbot API."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique customer ID",
        examples=["cust_1001"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation session ID",
        examples=["sess_abc123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Raw message text from the user",
        examples=["What is my current bill balance?"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "cust_1001",
                "session_id": "sess_abc123",
                "message": "What is my current bill balance?",
            }
        }
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational reply or tool execution output")
    route_taken: str = Field(
        ...,
        description="Routing branch taken: TOOL, CLARIFICATION, or LLM_FALLBACK",
        examples=["TOOL"],
    )
    intent: Optional[str] = Field(
        None,
        description="Classified or active conversational intent",
        examples=["bill_balance"],
    )
    confidence_score: float = Field(
        ...,
        description="Softmax probability confidence for the top intent",
        examples=[0.9982],
    )
    energy_score: float = Field(
        ...,
        description="Free energy score S_energy(z) = logsumexp(z)",
        examples=[11.45],
    )
    slots: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted and accumulated entity slots",
        examples=[{"account_type": "credit"}],
    )
    latency_ms: float = Field(
        ...,
        description="Inference and routing latency in milliseconds",
        examples=[14.8],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply": "Your credit balance is $450.25. Payment due date: 2026-10-15.",
                "route_taken": "TOOL",
                "intent": "bill_balance",
                "confidence_score": 0.9982,
                "energy_score": 11.45,
                "slots": {"account_type": "credit"},
                "latency_ms": 14.8,
            }
        }
    )


class ResetSessionRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        description="Session ID to clear state and uncommitted slots for",
        examples=["sess_abc123"],
    )


class ResetSessionResponse(BaseModel):
    status: str = Field(default="ok")
    session_id: str
    message: str


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    model_loaded: bool
    encoder_name: str
    uptime_seconds: float
