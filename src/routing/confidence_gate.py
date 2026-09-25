"""Production Confidence Gate for routing customer queries between deterministic tools and LLM fallback.

Derivation & Provenance:
- Uses an Energy-based scoring function: S_energy(z) = logsumexp(z)
- Calibrated using cost-sensitive asymmetric risk optimization on the validation set:
  - Read actions (R=10): tau = 10.2707
  - Write/Mutation actions (R=50): tau = 11.1946
"""

from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
from scipy.special import logsumexp, softmax


RouteType = Literal["TOOL", "LLM_FALLBACK"]


@dataclass(frozen=True)
class RoutingDecision:
    route: RouteType
    intent: str
    energy_score: float
    softmax_confidence: float
    is_mutation: bool
    threshold_used: float
    reason: str


class ConfidenceGate:
    # State-mutating actions carry asymmetric penalty if misrouted (financial movement, credential locks)
    WRITE_INTENTS = {"pay_bill", "freeze_account", "pin_change"}
    READ_INTENTS = {"order_status", "bill_balance", "card_declined"}

    # Empirically optimized cutoffs from validation set sweep
    DEFAULT_TAU_READ = 10.2707    # R = 10 (Balanced regime)
    DEFAULT_TAU_WRITE = 11.1946   # R = 50 (Safety-critical write regime)

    def __init__(
        self,
        class_names: list[str],
        tau_read: float = DEFAULT_TAU_READ,
        tau_write: float = DEFAULT_TAU_WRITE,
    ):
        self.class_names = class_names
        self.tau_read = tau_read
        self.tau_write = tau_write

    def decide(self, logits: np.ndarray) -> RoutingDecision:
        """Evaluate a 1D vector of pre-softmax logits and return a routing decision.

        Args:
            logits: (K,) array of unnormalized logits from the classifier head.

        Returns:
            RoutingDecision specifying whether to invoke the tool or fallback to LLM.
        """
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1 or len(logits) != len(self.class_names):
            raise ValueError(
                f"Expected 1D logits of length {len(self.class_names)}, got shape {logits.shape}"
            )

        # 1. Compute predictions and scores
        pred_idx = int(np.argmax(logits))
        intent = self.class_names[pred_idx]

        probs = softmax(logits)
        softmax_conf = float(probs[pred_idx])
        energy = float(logsumexp(logits))

        # 2. Determine risk regime
        is_mutation = intent in self.WRITE_INTENTS
        threshold = self.tau_write if is_mutation else self.tau_read

        # 3. Apply the gate
        if energy >= threshold:
            return RoutingDecision(
                route="TOOL",
                intent=intent,
                energy_score=energy,
                softmax_confidence=softmax_conf,
                is_mutation=is_mutation,
                threshold_used=threshold,
                reason="sufficient_energy_in_distribution",
            )
        else:
            return RoutingDecision(
                route="LLM_FALLBACK",
                intent=intent,
                energy_score=energy,
                softmax_confidence=softmax_conf,
                is_mutation=is_mutation,
                threshold_used=threshold,
                reason="insufficient_confidence_routed_to_fallback",
            )

    def decide_batch(self, logits_batch: np.ndarray) -> list[RoutingDecision]:
        """Vectorized evaluation over a 2D batch of logits (N, K)."""
        logits_batch = np.asarray(logits_batch, dtype=np.float32)
        if logits_batch.ndim != 2 or logits_batch.shape[1] != len(self.class_names):
            raise ValueError(
                f"Expected 2D logits of shape (N, {len(self.class_names)}), got {logits_batch.shape}"
            )
        return [self.decide(row) for row in logits_batch]