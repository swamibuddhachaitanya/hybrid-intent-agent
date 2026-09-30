"""LLM Fallback Client supporting Groq API and Mock Fallback for tests.

PCI-DSS / Safety Rule:
The LLM is strictly used for general assistance, out-of-scope deflection,
and conversational clarification. It NEVER directly executes database mutations.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv

from src.observability.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Returned to the customer when the fallback provider cannot be reached. Kept here
# so the router and the benchmark assert against the same string.
LLM_UNAVAILABLE_REPLY = (
    "I'm having trouble connecting to my knowledge base right now. "
    "Please try again or reach our support team."
)


class LLMUnavailableError(RuntimeError):
    """Raised when the fallback provider cannot be reached or rejects the call.

    The router treats this as a routing outcome, not a crash: the customer still
    gets a reply, and the failure shows up in logs and metrics as its own route.
    """


SYSTEM_PROMPT = """You are a polite, helpful customer support assistant for a retail bank.
Your primary role is to assist customers with banking questions.

Rules:
1. If a customer asks a question outside of banking (like cooking, weather, trivia, or sports), politely inform them that you can only assist with banking and account-related topics.
2. If a customer asks about a banking service we do not support in this automated chat (like mortgage applications, commercial loans, or investment brokerage), guide them to call customer service at 1-800-555-0199 or visit bank.com.
3. NEVER make up account balances, card numbers, or transaction IDs.
4. Keep your responses concise (2 to 3 sentences maximum).
"""


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self, user_text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate conversational reply given user text and optional turn history."""
        pass


class MockLLMClient(BaseLLMClient):
    """Deterministic mock provider for offline tests and CI/CD."""

    def generate(
        self, user_text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        text_lower = user_text.lower()

        # Out of domain (recipes, weather, general trivia)
        if any(
            w in text_lower
            for w in ["recipe", "cake", "weather", "trivia", "capital of", "cook", "brownie"]
        ):
            return "I am a banking assistant and can only help with account-related questions. How may I assist you with your banking today?"

        # Unsupported banking (mortgages, loans, investments)
        if any(
            w in text_lower
            for w in ["mortgage", "loan", "invest", "stock", "crypto", "refinance"]
        ):
            return "We currently don't support mortgage or investment services via this chat. Please contact our lending team at 1-800-555-0199."

        # Generic conversational fallback
        return "I'm not completely sure I understood your request. Could you please clarify if you'd like help with checking balances, card freezes, payments, or order status?"


class GroqLLMClient(BaseLLMClient):
    """Live LLM fallback client using the ultra-fast Groq API."""

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3.8-27b"
    # 429 and the 5xx family are worth another attempt; 4xx auth/config errors are not.
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    MAX_ATTEMPTS = 3
    MAX_RETRY_WAIT_S = 6.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        retry_delay_s: float = 1.0,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.retry_delay_s = retry_delay_s
        # Populated after a successful call so callers can measure real token usage
        # instead of assuming it.
        self.last_usage: Optional[Dict[str, int]] = None
        # Attempts used by the most recent call, so a caller can tell a first-try
        # success from one that survived a rate limit.
        self.last_attempts: int = 1
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable or pass explicitly."
            )

    def _retry_wait(self, exc: Exception, attempt: int) -> float:
        """Honour Retry-After when the provider sends it, else back off linearly."""
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "headers", {}) or {}
        try:
            suggested = float(retry_after.get("retry-after", 0))
        except (TypeError, ValueError):
            suggested = 0.0
        return min(max(suggested, self.retry_delay_s * attempt), self.MAX_RETRY_WAIT_S)

    def generate(
        self, user_text: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_text})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 150,
        }

        started = time.perf_counter()
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            self.last_attempts = attempt
            try:
                response = requests.post(
                    self.GROQ_API_URL, headers=headers, json=payload, timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") or {}
                self.last_usage = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                logger.info(
                    "llm_call_ok",
                    extra={
                        "model": self.model,
                        "attempt": attempt,
                        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    },
                )
                return data["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status in self.RETRYABLE_STATUS or isinstance(
                    exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
                )
                if retryable and attempt < self.MAX_ATTEMPTS:
                    wait = self._retry_wait(exc, attempt)
                    logger.warning(
                        "llm_call_retry",
                        extra={
                            "model": self.model,
                            "attempt": attempt,
                            "status": status,
                            "wait_s": round(wait, 2),
                            "error_type": type(exc).__name__,
                        },
                    )
                    time.sleep(wait)
                    continue
                break

        # Every attempt failed. A stale key, a rate limit and a bad model name all arrive
        # here. Log the real cause and fail loudly to the caller; do not disguise it as a
        # reply, because that makes an error look like a fast successful generation.
        body = getattr(getattr(last_exc, "response", None), "text", "")
        logger.error(
            "llm_call_failed",
            extra={
                "model": self.model,
                "attempts": self.last_attempts,
                "error_type": type(last_exc).__name__,
                "error": str(last_exc),
                "status": getattr(getattr(last_exc, "response", None), "status_code", None),
                "body": body[:300],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
        )
        raise LLMUnavailableError(
            f"LLM fallback unavailable after {self.last_attempts} attempt(s): "
            f"{type(last_exc).__name__}"
        ) from last_exc
