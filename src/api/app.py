"""FastAPI application for Hybrid Intent Chatbot service.

Features:
- Lifespan context manager loading Keras model and MiniLM encoder once on startup
- CORS middleware
- Latency instrumentation (ms)
- Typed REST endpoints with Pydantic V2 schemas
"""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResetSessionRequest,
    ResetSessionResponse,
)
from src.observability.logger import configure_logging, get_logger, session_fingerprint
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import BaseLLMClient, GroqLLMClient, MockLLMClient
from src.tools.banking_backend import BankingBackendSimulator, backend_db
from src.tools.tool_executor import ToolExecutor

logger = get_logger(__name__)

# Filepaths
MODEL_PATH = Path("models/baseline_intent.keras")
LABEL_MAP_PATH = Path("data/processed/label_map.json")

# Server state container
app_state = {
    "router": None,
    "start_time": time.time(),
}


def build_router(
    model_path: Path = MODEL_PATH,
    label_map_path: Path = LABEL_MAP_PATH,
    backend: Optional[BankingBackendSimulator] = None,
    llm_client: Optional[BaseLLMClient] = None,
) -> HybridRouter:
    """Instantiate the complete hybrid router stack."""
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found at {label_map_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    db = backend or backend_db
    executor = ToolExecutor(db)

    # Use Groq if API key is present in environment, otherwise MockLLMClient
    if llm_client is None:
        groq_key = os.environ.get("GROQ_API_KEY")
        llm = GroqLLMClient(api_key=groq_key) if groq_key else MockLLMClient()
    else:
        llm = llm_client

    return HybridRouter(
        model_path=str(model_path),
        class_names=class_names,
        tool_executor=executor,
        llm_client=llm,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, embeddings, and router once at server startup if not already injected."""
    app_state["start_time"] = time.time()
    if app_state["router"] is None:
        try:
            app_state["router"] = build_router()
        except Exception as exc:
            print(f"Warning: Failed to initialize router on startup: {exc}")
    yield
    # Shutdown logic
    if not app.extra.get("testing"):
        app_state["router"] = None


def create_app(router: Optional[HybridRouter] = None) -> FastAPI:
    """App factory function enabling dependency injection for testing."""
    configure_logging()

    application = FastAPI(
        title="Hybrid Intent Chatbot API",
        version="1.0.0",
        description="High-reliability conversational banking API with Confidence Gated NLU, deterministic tools, and LLM fallback.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    STATIC_INDEX_PATH = Path("src/api/static/index.html")

    @application.get(
        "/",
        summary="Web UI - Interactive Hybrid Router Inspector",
        tags=["UI"],
    )
    async def serve_ui():
        if STATIC_INDEX_PATH.exists():
            with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        return HTMLResponse(
            content="<h1>Hybrid Customer Support API running. Visit /docs for OpenAPI specs.</h1>",
            status_code=200,
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        tags=["System"],
    )
    async def health():
        current_router = app_state.get("router")
        uptime = time.time() - app_state["start_time"]
        return HealthResponse(
            status="healthy" if current_router is not None else "degraded",
            version="1.0.0",
            model_loaded=current_router is not None,
            encoder_name="all-MiniLM-L6-v2",
            uptime_seconds=round(uptime, 2),
        )

    @application.post(
        "/chat",
        response_model=ChatResponse,
        summary="Process Customer Chat Message",
        tags=["Conversation"],
    )
    async def chat(request: ChatRequest):
        current_router = app_state.get("router")
        if current_router is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference model is not initialized or still loading.",
            )

        start_t = time.perf_counter()
        result = current_router.process_message(
            user_id=request.user_id,
            session_id=request.session_id,
            text=request.message,
        )
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        # Request-level record. The router logs the decision inputs; this logs the
        # outcome, so a wrong action can be traced to the score that allowed it.
        # Customer text stays out of the INFO record by design.
        logger.info(
            "chat_request",
            extra={
                "route_taken": result.route_taken,
                "intent": result.intent,
                "energy_score": round(result.energy_score, 4),
                "latency_ms": latency_ms,
                "session": session_fingerprint(request.session_id),
                "slots": result.slots,
                "text_len": len(request.message),
            },
        )

        return ChatResponse(
            reply=result.reply,
            route_taken=result.route_taken,
            intent=result.intent,
            confidence_score=round(result.confidence_score, 4),
            energy_score=round(result.energy_score, 4),
            slots=result.slots,
            latency_ms=latency_ms,
        )

    @application.post(
        "/session/reset",
        response_model=ResetSessionResponse,
        summary="Reset Dialogue Session State",
        tags=["Conversation"],
    )
    async def reset_session(request: ResetSessionRequest):
        current_router = app_state.get("router")
        if current_router is not None and hasattr(current_router, "tracker"):
            current_router.tracker.reset_session(request.session_id)
        return ResetSessionResponse(
            status="ok",
            session_id=request.session_id,
            message=f"Session {request.session_id} has been reset successfully.",
        )

    return application


app = create_app()
