import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline_intent.keras")
EVAL_DIR = Path("evaluation")
OUTPUT_FILE = EVAL_DIR / "optimal_threshold_selection.json"

COST_RATIOS = [10.0, 50.0]  # Cost of False Accept relative to False Reject (C_FR = 1.0)


def extract_logits_and_probs(model: tf.keras.Model, X: np.ndarray):
    """Keras 3 sequential extractor for probabilities and pre-softmax logits."""
    batch_size = 128
    logits_list = []
    final_dense = model.layers[-1]
    W, b = final_dense.get_weights()

    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        curr = tf.convert_to_tensor(batch, dtype=tf.float32)
        for layer in model.layers[:-1]:
            curr = layer(curr, training=False)
        features = curr.numpy()
        logits_batch = np.dot(features, W) + b
        logits_list.append(logits_batch)

    logits = np.vstack(logits_list)
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    return logits, probs


def find_optimal_threshold(
    in_scores: np.ndarray,
    in_correct_mask: np.ndarray,
    ood_scores: np.ndarray,
    cost_ratio: float,
    num_grid_points: int = 500,
):
    """
    Find threshold tau that minimizes:
      Total Cost = C_FR * N_FR + C_FA * N_FA
    Where:
      - False Reject (FR): A correctly classified in-dist query whose score < tau (unnecessarily sent to LLM)
      - False Accept (FA): A misclassified in-dist query OR any OOD query whose score >= tau (leaked into tool)
    """
    min_val = min(np.min(in_scores), np.min(ood_scores))
    max_val = max(np.max(in_scores), np.max(ood_scores))
    thresholds = np.linspace(min_val, max_val, num_grid_points)

    best_threshold = None
    min_cost = float("inf")
    best_stats = {}

    n_in = len(in_scores)
    n_ood = len(ood_scores)

    for tau in thresholds:
        # In-dist decisions
        passed_in = in_scores >= tau
        # False Reject: was correct in-distribution, but rejected by threshold
        false_rejects = np.sum((~passed_in) & in_correct_mask)
        # False Accept from in-dist: was misclassified, but passed threshold
        in_dist_false_accepts = np.sum(passed_in & (~in_correct_mask))

        # False Accept from OOD: any OOD query that passed threshold
        ood_false_accepts = np.sum(ood_scores >= tau)

        total_false_accepts = in_dist_false_accepts + ood_false_accepts

        # Normalized cost per total traffic
        cost = (1.0 * false_rejects) + (cost_ratio * total_false_accepts)

        if cost < min_cost:
            min_cost = cost
            best_threshold = float(tau)
            best_stats = {
                "threshold": float(tau),
                "total_cost": float(cost),
                "false_reject_count": int(false_rejects),
                "false_reject_rate": float(false_rejects / n_in),
                "false_accept_count": int(total_false_accepts),
                "false_accept_rate": float(total_false_accepts / n_ood),
            }

    return best_stats


def evaluate_threshold_on_test(
    tau: float,
    in_scores: np.ndarray,
    in_correct_mask: np.ndarray,
    ood_scores_dict: dict,
):
    """Evaluate generalization of selected threshold tau on held-out test data."""
    passed_in = in_scores >= tau
    retained_in_pct = float(np.mean(passed_in)) * 100.0

    # In-dist misclassifications that penetrated the gate
    leaked_in_errors = int(np.sum(passed_in & (~in_correct_mask)))

    ood_leakages = {}
    for name, scores in ood_scores_dict.items():
        leaked = int(np.sum(scores >= tau))
        rate = float(np.mean(scores >= tau)) * 100.0
        ood_leakages[name] = {"count": leaked, "rate_pct": rate}

    return {
        "tau": tau,
        "in_distribution_automated_pct": retained_in_pct,
        "in_distribution_error_leaks": leaked_in_errors,
        "ood_leakages": ood_leakages,
    }


def main():
    # 1. Load data
    val_data = np.load(PROCESSED_DIR / "val_features.npz")
    test_data = np.load(PROCESSED_DIR / "test_features.npz")
    ood_data = np.load(PROCESSED_DIR / "ood_features.npz")

    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    X_ood, ood_tiers = ood_data["embeddings"], ood_data["tiers"]

    # 2. Model inference
    model = tf.keras.models.load_model(MODEL_PATH)

    logits_val, probs_val = extract_logits_and_probs(model, X_val)
    logits_test, probs_test = extract_logits_and_probs(model, X_test)
    logits_ood, probs_ood = extract_logits_and_probs(model, X_ood)

    # In-dist accuracy flags
    val_preds = np.argmax(probs_val, axis=1)
    val_true = np.argmax(y_val, axis=1)
    val_correct_mask = val_preds == val_true

    test_preds = np.argmax(probs_test, axis=1)
    test_true = np.argmax(y_test, axis=1)
    test_correct_mask = test_preds == test_true

    # Scores
    msp_val = np.max(probs_val, axis=1)
    msp_test = np.max(probs_test, axis=1)
    msp_ood = np.max(probs_ood, axis=1)

    energy_val = logsumexp(logits_val, axis=1)
    energy_test = logsumexp(logits_test, axis=1)
    energy_ood = logsumexp(logits_ood, axis=1)

    # OOD subsets for test audit
    ood_msp_subsets = {
        "Tier A": msp_ood[ood_tiers == "A"],
        "Tier B": msp_ood[ood_tiers == "B"],
        "Tier C": msp_ood[ood_tiers == "C"],
    }
    ood_energy_subsets = {
        "Tier A": energy_ood[ood_tiers == "A"],
        "Tier B": energy_ood[ood_tiers == "B"],
        "Tier C": energy_ood[ood_tiers == "C"],
    }

    # 3. Optimize thresholds on Validation + OOD
    results = {}

    print("=" * 80)
    print("COST-SENSITIVE THRESHOLD OPTIMIZATION (Tuned strictly on Validation set)")
    print("=" * 80)
    print(
        f"{'Cost Ratio (R)':<15} | {'Scorer':<12} | {'Optimal Tau':<12} | {'Val FR Rate':<12} | {'Val FA Rate'}"
    )
    print("-" * 80)

    for R in COST_RATIOS:
        results[f"R_{int(R)}"] = {}

        # Softmax optimization
        opt_msp = find_optimal_threshold(
            msp_val, val_correct_mask, msp_ood, cost_ratio=R
        )
        test_eval_msp = evaluate_threshold_on_test(
            opt_msp["threshold"], msp_test, test_correct_mask, ood_msp_subsets
        )
        results[f"R_{int(R)}"]["Softmax"] = {
            "validation_tuning": opt_msp,
            "test_evaluation": test_eval_msp,
        }
        print(
            f"R = {int(R):<11} | {'Softmax':<12} | {opt_msp['threshold']:<12.4f} | "
            f"{opt_msp['false_reject_rate'] * 100:>10.2f}% | {opt_msp['false_accept_rate'] * 100:>10.2f}%"
        )

        # Energy optimization
        opt_energy = find_optimal_threshold(
            energy_val, val_correct_mask, energy_ood, cost_ratio=R
        )
        test_eval_energy = evaluate_threshold_on_test(
            opt_energy["threshold"], energy_test, test_correct_mask, ood_energy_subsets
        )
        results[f"R_{int(R)}"]["Energy"] = {
            "validation_tuning": opt_energy,
            "test_evaluation": test_eval_energy,
        }
        print(
            f"R = {int(R):<11} | {'Energy':<12} | {opt_energy['threshold']:<12.4f} | "
            f"{opt_energy['false_reject_rate'] * 100:>10.2f}% | {opt_energy['false_accept_rate'] * 100:>10.2f}%"
        )
        print("-" * 80)

    # 4. Print Held-out Test Audit Table
    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET AUDIT AT OPTIMAL OPERATING POINTS")
    print("=" * 80)
    print(
        f"{'Regime':<12} | {'Scorer':<8} | {'Tau':<8} | {'In-Dist Auto':<12} | {'In-Dist Err':<12} | {'Tier A Leak':<11} | {'Tier B Leak':<11} | {'Tier C Leak'}"
    )
    print("-" * 80)
    for R in COST_RATIOS:
        regime = f"R={int(R)}"
        for scorer in ["Softmax", "Energy"]:
            t_data = results[f"R_{int(R)}"][scorer]["test_evaluation"]
            tau = t_data["tau"]
            auto_pct = t_data["in_distribution_automated_pct"]
            err_count = t_data["in_distribution_error_leaks"]
            a_leak = t_data["ood_leakages"]["Tier A"]["rate_pct"]
            b_leak = t_data["ood_leakages"]["Tier B"]["rate_pct"]
            c_leak = t_data["ood_leakages"]["Tier C"]["rate_pct"]
            print(
                f"{regime:<12} | {scorer:<8} | {tau:<8.4f} | {auto_pct:>10.2f}% | {err_count:>12} | "
                f"{a_leak:>9.2f}% | {b_leak:>9.2f}% | {c_leak:>9.2f}%"
            )
        print("-" * 80)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved cost-sensitive audit to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
