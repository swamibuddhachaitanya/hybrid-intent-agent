"""Benchmark latency and calculate cost savings for the hybrid agentic architecture."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Dict, List
from dotenv import load_dotenv
import numpy as np

from src.observability.logger import configure_logging
from src.routing.hybrid_router import HybridRouter
from src.routing.llm_fallback import (
    LLM_UNAVAILABLE_REPLY,
    GroqLLMClient,
    MockLLMClient,
)
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor

load_dotenv()

MODEL_PATH = Path("models/baseline_intent.keras")
LABEL_MAP_PATH = Path("data/processed/label_map.json")
EVAL_DIR = Path("evaluation")
OUTPUT_PATH = EVAL_DIR / "latency_benchmark.json"

TOOL_QUERY = "What is my current bill balance?"
OOD_QUERY = "What is the best recipe for baking chocolate brownies?"

TOOL_ITERATIONS = 100
# The first requests in a process pay one-time costs: encoder graph setup, tokenizer
# caches, allocator growth. Measured on this machine, two warmups inflated the tool p50
# by about 40% (28.20ms against 20.66ms with ten). Warm properly and drop the first
# samples so the reported percentiles describe steady state.
WARMUP_ITERATIONS = 20
DISCARD_FIRST_N = 10
# Sample count for the LLM path is bounded by the provider, not by preference: this
# model's tier caps output at 1000 tokens per minute, so an unpaced 100-call burst
# starts returning 429s partway through. 30 paced samples fit the budget.
LLM_ITERATIONS = 30
INTER_CALL_SLEEP_S = 2.5


def calculate_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def main():
    print("=" * 70)
    print("HYBRID INTENT CHATBOT: LATENCY & COST BENCHMARK (T8.1)")
    print("=" * 70)
    configure_logging()

    # 1. Initialize HybridRouter with live Groq client if key is available
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    backend = BankingBackendSimulator()
    executor = ToolExecutor(backend)

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            llm_client = GroqLLMClient(api_key=groq_key)
            llm_provider = f"Groq API ({llm_client.model})"
        except Exception as e:
            print(f"Warning: Failed to init Groq client ({e}), falling back to MockLLMClient.")
            llm_client = MockLLMClient()
            llm_provider = "MockLLMClient"
    else:
        llm_client = MockLLMClient()
        llm_provider = "MockLLMClient"

    print(f"LLM Provider: {llm_provider}")

    router = HybridRouter(
        model_path=str(MODEL_PATH),
        class_names=class_names,
        tool_executor=executor,
        llm_client=llm_client,
    )

    # Warmup pass
    print(f"\nWarming up router ({WARMUP_ITERATIONS} requests)...")
    for i in range(WARMUP_ITERATIONS):
        router.process_message("cust_1001", f"warmup_{i}", TOOL_QUERY)
    router.process_message("cust_1001", "warmup_ood", OOD_QUERY)

    # 2. Benchmark Tool Path (100 iterations)
    print(f"\n1. Benchmarking Tool Path ({TOOL_ITERATIONS} iterations): '{TOOL_QUERY}'...")
    tool_latencies: List[float] = []
    for i in range(TOOL_ITERATIONS):
        sess_id = f"bench_tool_{i}"
        t0 = time.perf_counter()
        resp = router.process_message("cust_1001", sess_id, TOOL_QUERY)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.route_taken == "TOOL"
        tool_latencies.append(elapsed_ms)

    tool_stats = calculate_percentiles(tool_latencies[DISCARD_FIRST_N:])
    print(
        f"   Mean: {tool_stats['mean_ms']}ms | "
        f"p50: {tool_stats['median_p50_ms']}ms | "
        f"p95: {tool_stats['p95_ms']}ms | "
        f"p99: {tool_stats['p99_ms']}ms"
    )

    # 3. Benchmark LLM Fallback Path (10 iterations)
    print(f"\n2. Benchmarking LLM Fallback Path ({LLM_ITERATIONS} iterations): '{OOD_QUERY}'...")
    llm_latencies: List[float] = []
    prompt_tokens: List[int] = []
    completion_tokens: List[int] = []
    attempts_used: List[int] = []
    for i in range(LLM_ITERATIONS):
        sess_id = f"bench_llm_{i}"
        t0 = time.perf_counter()
        resp = router.process_message("cust_1001", sess_id, OOD_QUERY)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # A provider failure must never be recorded as a fast successful call.
        assert resp.route_taken == "LLM_FALLBACK", f"unexpected route: {resp.route_taken}"
        assert resp.reply != LLM_UNAVAILABLE_REPLY, "provider returned the unavailable notice"
        llm_latencies.append(elapsed_ms)
        attempts_used.append(getattr(llm_client, "last_attempts", 1))
        usage = getattr(llm_client, "last_usage", None) or {}
        if usage.get("prompt_tokens"):
            prompt_tokens.append(int(usage["prompt_tokens"]))
        if usage.get("completion_tokens"):
            completion_tokens.append(int(usage["completion_tokens"]))
        if i == 0 or (i + 1) % 10 == 0:
            print(f"   Iteration {i+1}/{LLM_ITERATIONS}: {elapsed_ms:.1f}ms")
        # Pace the calls. The token-per-minute cap is the binding constraint, and pacing
        # keeps rate-limit waits out of the measured latency distribution.
        if i + 1 < LLM_ITERATIONS:
            time.sleep(INTER_CALL_SLEEP_S)

    llm_stats = calculate_percentiles(llm_latencies)
    print(
        f"\n   LLM Mean: {llm_stats['mean_ms']}ms | "
        f"p50: {llm_stats['median_p50_ms']}ms | "
        f"p95: {llm_stats['p95_ms']}ms | "
        f"p99: {llm_stats['p99_ms']}ms"
    )

    # 4. Cost Modeling Analysis (100,000 queries)
    # Market rates for standard enterprise LLMs (e.g. GPT-4o-mini / Claude 3.5 Haiku / Groq Llama 3)
    # ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
    # Average banking user query: ~35 input tokens; system prompt + history: ~150 tokens = 185 tokens
    # Average response: ~60 output tokens
    total_queries = 100_000
    # Token counts come from the provider's usage field on the calls made above.
    # The previous 185 / 60 were estimates, which made the cost model arithmetic on
    # assumptions rather than on a measurement.
    avg_input_tokens = round(float(np.mean(prompt_tokens)), 1) if prompt_tokens else 185
    avg_output_tokens = round(float(np.mean(completion_tokens)), 1) if completion_tokens else 60
    tokens_measured = bool(prompt_tokens)

    cost_per_input_token = 0.15 / 1_000_000
    cost_per_output_token = 0.60 / 1_000_000

    cost_per_llm_query = (avg_input_tokens * cost_per_input_token) + (
        avg_output_tokens * cost_per_output_token
    )

    # Pure LLM Baseline (100% of queries hitting LLM)
    pure_llm_cost = round(total_queries * cost_per_llm_query, 2)

    # Hybrid Architecture:
    # 45.8% automated locally via local classifier & tools ($0.00 model API cost)
    # 54.2% routed to LLM fallback
    # NOTE: this is the READ-intent regime (tau=10.2707). Write intents use the
    # stricter tau=11.1946 and automate less, so blended traffic lands between the two.
    local_automated_rate = 0.458
    llm_fallback_rate = 0.542

    hybrid_llm_queries = total_queries * llm_fallback_rate
    hybrid_cost = round(hybrid_llm_queries * cost_per_llm_query, 2)
    savings_amount = round(pure_llm_cost - hybrid_cost, 2)
    savings_percentage = round((savings_amount / pure_llm_cost) * 100, 2)

    print("\n" + "=" * 70)
    print("COST MODELING & SAVINGS ANALYSIS (100,000 Queries)")
    print("=" * 70)
    print(f"Pure LLM Baseline Cost:   ${pure_llm_cost:.2f}")
    print(f"Hybrid System Cost:       ${hybrid_cost:.2f}")
    print(f"Net Cloud Savings:        ${savings_amount:.2f} ({savings_percentage}%)")
    print(f"Local Automation Rate:    {local_automated_rate * 100:.1f}%")
    print("=" * 70)

    # 5. Save audited benchmark artifact
    benchmark_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "Local CPU Inference (Intel / AMD)",
        "llm_provider": llm_provider,
        "sample_counts": {
            "tool_iterations": TOOL_ITERATIONS,
            "tool_warmup_iterations": WARMUP_ITERATIONS,
            "tool_samples_discarded": DISCARD_FIRST_N,
            "llm_iterations": LLM_ITERATIONS,
            "inter_call_sleep_s": INTER_CALL_SLEEP_S,
        },
        "environment_note": (
            "Measured on a shared laptop that also runs Docker Desktop. Across repeated runs "
            "the tool p50 ranged from about 16ms to 32ms depending on warmup and background "
            "load, so treat these percentiles as an order of magnitude, not a specification."
        ),
        "rate_limit_note": (
            "The provider caps this model at 1000 output tokens per minute, which bounds "
            "benchmark throughput regardless of the sample count requested. Calls are paced by "
            f"{INTER_CALL_SLEEP_S}s so rate-limit waits do not enter the latency distribution."
        ),
        "latency_metrics": {
            "tool_path": tool_stats,
            "llm_fallback_path": llm_stats,
            "llm_calls_that_retried": sum(1 for a in attempts_used if a > 1),
            "latency_speedup_factor": round(
                llm_stats["median_p50_ms"] / max(0.001, tool_stats["median_p50_ms"]), 1
            ),
        },
        "cost_analysis_100k_queries": {
            "total_queries": total_queries,
            "assumptions": {
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "token_source": "measured_provider_usage" if tokens_measured else "estimated",
                "input_token_rate_per_million": 0.15,
                "output_token_rate_per_million": 0.60,
                "cost_per_query_usd": round(cost_per_llm_query, 6),
            },
            "pure_llm_baseline_usd": pure_llm_cost,
            "hybrid_system_cost_usd": hybrid_cost,
            "net_savings_usd": savings_amount,
            "savings_percentage": savings_percentage,
            "read_regime_local_automated_pct": local_automated_rate * 100.0,
            "llm_fallback_pct": llm_fallback_rate * 100.0,
            "regime_note": (
                "Automation rate is the read-intent cutoff (tau=10.2707). Write intents use "
                "tau=11.1946, so blended traffic sits between 33% and 46%. The savings "
                "percentage equals the automation rate by construction, because the local "
                "path is priced at zero."
            ),
        },
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    print(f"\nWrote benchmark results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
