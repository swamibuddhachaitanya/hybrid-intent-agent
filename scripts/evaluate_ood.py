import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.preprocessing.embedding_preprocessor import EmbeddingPreprocessor

OOD_TIERS_PATH = Path("data/processed/ood_tiers.json")
OOD_FEATURES_PATH = Path("data/processed/ood_features.npz")
IN_DIST_TEST_PATH = Path("data/processed/test_features.npz")
MODEL_PATH = Path("models/baseline_intent.keras")
LABEL_MAP_PATH = Path("data/processed/label_map.json")
EVAL_DIR = Path("evaluation")
OUTPUT_METRICS_PATH = EVAL_DIR / "ood_baseline_evaluation.json"

THRESHOLDS = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def get_or_compute_embeddings(records: list[dict]) -> np.ndarray:
    if OOD_FEATURES_PATH.exists():
        print(f"[cache] Loading cached OOD embeddings from {OOD_FEATURES_PATH}...")
        data = np.load(OOD_FEATURES_PATH)
        return data["embeddings"]

    print(f"[embeddings] Generating embeddings for {len(records)} texts...")
    preprocessor = EmbeddingPreprocessor()
    texts = [r["text"] for r in records]
    embeddings = preprocessor.transform(texts, batch_size=64)

    OOD_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOD_FEATURES_PATH,
        embeddings=embeddings,
        tiers=np.array([r["tier"] for r in records]),
        source_tags=np.array([r["source_tag"] for r in records]),
    )
    print(f"[cache] Saved OOD embeddings to {OOD_FEATURES_PATH}")
    return embeddings


def compute_distribution_stats(p_max_values: np.ndarray) -> dict:
    if len(p_max_values) == 0:
        return {}
    return {
        "count": int(len(p_max_values)),
        "mean_p_max": float(np.mean(p_max_values)),
        "std_p_max": float(np.std(p_max_values)),
        "min_p_max": float(np.min(p_max_values)),
        "p50": float(np.percentile(p_max_values, 50)),
        "p90": float(np.percentile(p_max_values, 90)),
        "p95": float(np.percentile(p_max_values, 95)),
        "p99": float(np.percentile(p_max_values, 99)),
        "max_p_max": float(np.max(p_max_values)),
    }


def compute_leakage_table(p_max_values: np.ndarray) -> dict:
    n = len(p_max_values)
    if n == 0:
        return {}
    return {
        f"tau_{t:.2f}": {
            "leaked_count": int(np.sum(p_max_values >= t)),
            "leakage_rate": float(np.mean(p_max_values >= t)),
        }
        for t in THRESHOLDS
    }


def main():
    with open(OOD_TIERS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}

    # 1. Feature Extraction / Cache
    ood_embeddings = get_or_compute_embeddings(records)

    # 2. Model Inference
    print(f"[inference] Loading baseline model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(ood_embeddings, batch_size=128, verbose=1)
    p_max = np.max(probs, axis=1)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [idx_to_label[i] for i in pred_idx]

    # Load In-Distribution Test set for direct comparison
    in_dist_data = np.load(IN_DIST_TEST_PATH)
    in_dist_x = in_dist_data["X"]
    in_dist_probs = model.predict(in_dist_x, batch_size=128, verbose=0)
    in_dist_p_max = np.max(in_dist_probs, axis=1)

    # 3. Partition by Tier
    tiers = np.array([r["tier"] for r in records])
    tier_a_mask = tiers == "A"
    tier_b_mask = tiers == "B"
    tier_c_mask = tiers == "C"

    # 4. Analyze Tier C Intent-by-Intent Collisions
    tier_c_records = [
        (i, records[i]) for i in range(len(records)) if records[i]["tier"] == "C"
    ]
    tier_c_analysis = {}

    for idx, r in tier_c_records:
        intent = r["source_tag"]
        hypothesized = r["collides_with"]
        predicted = pred_labels[idx]
        conf = float(p_max[idx])

        if intent not in tier_c_analysis:
            tier_c_analysis[intent] = {
                "hypothesized_collision": hypothesized,
                "total": 0,
                "matched_hypothesis_count": 0,
                "p_max_values": [],
                "predicted_distribution": {},
            }

        tier_c_analysis[intent]["total"] += 1
        tier_c_analysis[intent]["p_max_values"].append(conf)
        if predicted == hypothesized:
            tier_c_analysis[intent]["matched_hypothesis_count"] += 1

        tier_c_analysis[intent]["predicted_distribution"][predicted] = (
            tier_c_analysis[intent]["predicted_distribution"].get(predicted, 0) + 1
        )

    for intent, data in tier_c_analysis.items():
        arr = np.array(data["p_max_values"])
        data["mean_p_max"] = float(np.mean(arr))
        data["p95"] = float(np.percentile(arr, 95))
        data["hypothesis_match_rate"] = float(
            data["matched_hypothesis_count"] / data["total"]
        )
        data.pop("p_max_values")

    # 5. Compile Artifact
    report = {
        "summary": {
            "total_evaluated": len(records),
            "in_distribution_test_count": len(in_dist_p_max),
        },
        "distributions": {
            "in_distribution_test": compute_distribution_stats(in_dist_p_max),
            "tier_a_out_of_domain": compute_distribution_stats(p_max[tier_a_mask]),
            "tier_b_in_domain_unsupported": compute_distribution_stats(
                p_max[tier_b_mask]
            ),
            "tier_c_hard_boundary": compute_distribution_stats(p_max[tier_c_mask]),
        },
        "leakage_tables": {
            "in_distribution_test_retention": compute_leakage_table(in_dist_p_max),
            "tier_a_leakage": compute_leakage_table(p_max[tier_a_mask]),
            "tier_b_leakage": compute_leakage_table(p_max[tier_b_mask]),
            "tier_c_leakage": compute_leakage_table(p_max[tier_c_mask]),
        },
        "tier_c_breakdown": tier_c_analysis,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 6. Console Output
    print("\n" + "=" * 70)
    print("BASELINE OOD EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Subset':<30} | {'Mean p_max':<10} | {'p50':<8} | {'p90':<8} | {'p95':<8}")
    print("-" * 70)
    for name, key in [
        ("In-Dist Test (Supported)", "in_distribution_test"),
        ("Tier A (Out of Domain)", "tier_a_out_of_domain"),
        ("Tier B (In-Domain Unsup)", "tier_b_in_domain_unsupported"),
        ("Tier C (Hard Boundary)", "tier_c_hard_boundary"),
    ]:
        d = report["distributions"][key]
        print(
            f"{name:<30} | {d['mean_p_max']:<10.4f} | {d['p50']:<8.4f} | {d['p90']:<8.4f} | {d['p95']:<8.4f}"
        )

    print("\n" + "=" * 70)
    print("LEAKAGE RATE AT CANDIDATE THRESHOLDS (Fraction of queries >= Threshold)")
    print("=" * 70)
    print(
        f"{'Threshold (tau)':<16} | {'In-Dist Retained':<16} | {'Tier A Leak':<12} | {'Tier B Leak':<12} | {'Tier C Leak':<12}"
    )
    print("-" * 70)
    for t in THRESHOLDS:
        k = f"tau_{t:.2f}"
        id_ret = (
            report["leakage_tables"]["in_distribution_test_retention"][k][
                "leakage_rate"
            ]
            * 100
        )
        a_leak = report["leakage_tables"]["tier_a_leakage"][k]["leakage_rate"] * 100
        b_leak = report["leakage_tables"]["tier_b_leakage"][k]["leakage_rate"] * 100
        c_leak = report["leakage_tables"]["tier_c_leakage"][k]["leakage_rate"] * 100
        print(
            f"tau >= {t:<9.2f} | {id_ret:>14.2f}% | {a_leak:>10.2f}% | {b_leak:>10.2f}% | {c_leak:>10.2f}%"
        )

    print(f"\nSaved evaluation artifact to: {OUTPUT_METRICS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
