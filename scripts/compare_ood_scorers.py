import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from scipy.special import logsumexp

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline_intent.keras")
EVAL_DIR = Path("evaluation")
OUTPUT_FILE = EVAL_DIR / "ood_scorers_comparison.json"


def compute_centroids(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Compute unit-normalized centroids for each class in training set."""
    labels = np.argmax(y_train, axis=1)
    num_classes = y_train.shape[1]
    centroids = []
    for k in range(num_classes):
        class_embs = X_train[labels == k]
        mean_vec = np.mean(class_embs, axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids.append(mean_vec / norm if norm > 0 else mean_vec)
    return np.array(centroids)  # Shape: (num_classes, 384)


def extract_penultimate_and_logits(model: tf.keras.Model, X: np.ndarray):
    """
    Keras 3 sequential forward pass:
    Pass X sequentially through each layer up to the penultimate layer (layer -2),
    then compute pre-softmax logits using final Dense weights (W, b).
    """
    # Process in batches to keep memory usage low
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

    return np.vstack(logits_list)


def compute_metrics_auroc_fpr95(in_scores: np.ndarray, ood_scores: np.ndarray):
    """Compute AUROC and FPR at 95% TPR (in-distribution is positive class)."""
    y_true = np.concatenate([np.ones_like(in_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([in_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_scores)

    # Threshold where 95% of in-dist scores are >= threshold
    target_tpr = 0.95
    sorted_in_scores = np.sort(in_scores)
    cutoff_idx = int(np.floor((1.0 - target_tpr) * len(in_scores)))
    threshold_at_95_tpr = sorted_in_scores[cutoff_idx]

    fpr_at_95 = float(np.mean(ood_scores >= threshold_at_95_tpr))

    return float(auroc), float(fpr_at_95), float(threshold_at_95_tpr)


def main():
    # 1. Load data
    train_data = np.load(PROCESSED_DIR / "train_features.npz")
    test_data = np.load(PROCESSED_DIR / "test_features.npz")
    ood_data = np.load(PROCESSED_DIR / "ood_features.npz")

    X_train, y_train = train_data["X"], train_data["y"]
    X_test = test_data["X"]
    X_ood = ood_data["embeddings"]
    ood_tiers = ood_data["tiers"]

    # 2. Compute Centroids
    centroids = compute_centroids(X_train, y_train)

    # 3. Model setup
    model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Compute In-Distribution Scores
    probs_in = model.predict(X_test, batch_size=128, verbose=0)
    logits_in = extract_penultimate_and_logits(model, X_test)

    msp_in = np.max(probs_in, axis=1)
    energy_in = logsumexp(logits_in, axis=1)  # Free energy: logsumexp(logits)
    centroid_sim_in = np.max(np.dot(X_test, centroids.T), axis=1)

    # 5. Compute OOD Scores
    probs_ood = model.predict(X_ood, batch_size=128, verbose=0)
    logits_ood = extract_penultimate_and_logits(model, X_ood)

    msp_ood = np.max(probs_ood, axis=1)
    energy_ood = logsumexp(logits_ood, axis=1)
    centroid_sim_ood = np.max(np.dot(X_ood, centroids.T), axis=1)

    # 6. Evaluate by Tier
    tier_masks = {
        "Tier A (Out of Domain)": ood_tiers == "A",
        "Tier B (In-Domain Unsupported)": ood_tiers == "B",
        "Tier C (Hard Boundary)": ood_tiers == "C",
        "All OOD Combined": np.ones_like(ood_tiers, dtype=bool),
    }

    results = {}

    print("=" * 80)
    print("OOD SCORER COMPARISON: AUROC (higher is better) & FPR@95% (lower is better)")
    print("=" * 80)
    print(
        f"{'Subset':<28} | {'Scorer':<18} | {'AUROC':<10} | {'FPR@95% TPR':<12} | {'Thresh@95'}"
    )
    print("-" * 80)

    for tier_name, mask in tier_masks.items():
        results[tier_name] = {}
        for scorer_name, in_s, ood_s in [
            ("Max Softmax (MSP)", msp_in, msp_ood[mask]),
            ("Energy Score", energy_in, energy_ood[mask]),
            ("Centroid Cosine Sim", centroid_sim_in, centroid_sim_ood[mask]),
        ]:
            auroc, fpr95, thresh95 = compute_metrics_auroc_fpr95(in_s, ood_s)
            results[tier_name][scorer_name] = {
                "auroc": auroc,
                "fpr_at_95_tpr": fpr95,
                "threshold_at_95_tpr": thresh95,
            }
            print(
                f"{tier_name:<28} | {scorer_name:<18} | {auroc:<10.4f} | {fpr95 * 100:>10.2f}% | {thresh95:.4f}"
            )
        print("-" * 80)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed comparisons to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
