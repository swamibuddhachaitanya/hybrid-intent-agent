"""Pre-training data integrity check.

Answers one question before we train anything: is the test split a fair
measure of generalization?

Checks:
  1. exact duplicate strings inside a split
  2. the same string carrying two different labels
  3. strings that appear in both train and test
  4. near-duplicate pairs across splits, via embedding cosine similarity
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed")
NEAR_DUP_THRESHOLD = 0.95
MAX_EXAMPLES = 5


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def load_split(name: str):
    with open(PROCESSED_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [r["text"] for r in rows], [r["intent"] for r in rows]


def load_features(name: str):
    data = np.load(PROCESSED_DIR / f"{name}_features.npz")
    return data["X"], data["y"]


def load_class_names():
    with open(PROCESSED_DIR / "label_map.json", "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    return [idx_to_label[str(i)] for i in range(len(idx_to_label))]


def exact_duplicates(texts, labels, split_name: str):
    counts = Counter()
    labels_by_text = defaultdict(set)
    for text, label in zip(texts, labels):
        key = normalize(text)
        counts[key] += 1
        labels_by_text[key].add(label)

    dupes = {k: c for k, c in counts.items() if c > 1}
    conflicts = {k: v for k, v in labels_by_text.items() if len(v) > 1}

    print(f"\n[{split_name}] rows={len(texts)} unique_texts={len(counts)}")
    print(f"[{split_name}] texts appearing more than once: {len(dupes)}")
    for key, count in sorted(dupes.items(), key=lambda kv: -kv[1])[:MAX_EXAMPLES]:
        print(f"    x{count}  {key!r}")

    print(f"[{split_name}] same text, different intents: {len(conflicts)}")
    for key, intent_set in list(conflicts.items())[:MAX_EXAMPLES]:
        print(f"    {key!r} -> {sorted(intent_set)}")

    label_counts = Counter(labels)
    print(f"[{split_name}] per-intent counts: {dict(sorted(label_counts.items()))}")
    return len(dupes), len(conflicts)


def cross_split_overlap(train_texts, test_texts) -> int:
    train_norm = {normalize(t) for t in train_texts}
    hits = [normalize(t) for t in test_texts if normalize(t) in train_norm]

    print(f"\n[exact] test rows that also exist in train: {len(hits)}")
    for hit in hits[:MAX_EXAMPLES]:
        print(f"    {hit!r}")
    return len(hits)


def near_duplicates(
    X_train, y_train, X_eval, y_eval, texts_train, texts_eval, class_names, split_name: str
):
    # embeddings are unit length, so the dot product IS cosine similarity
    sims = X_eval @ X_train.T
    pairs = np.argwhere(sims > NEAR_DUP_THRESHOLD)

    scored = sorted(
        ((float(sims[i, j]), int(i), int(j)) for i, j in pairs),
        key=lambda t: -t[0],
    )

    print(f"\n[near-dup] {split_name}-vs-train pairs above {NEAR_DUP_THRESHOLD}: {len(scored)}")

    disagreeing = 0
    for score, i, j in scored:
        eval_intent = class_names[int(y_eval[i].argmax())]
        train_intent = class_names[int(y_train[j].argmax())]
        if eval_intent != train_intent:
            disagreeing += 1
        if (
            len([1 for s, _, _ in scored[:MAX_EXAMPLES]])
            and scored.index((score, i, j)) < MAX_EXAMPLES
        ):
            flag = "same intent" if eval_intent == train_intent else "DIFFERENT INTENT"
            print(f"    {score:.3f}  ({flag})")
            print(f"        {split_name} : [{eval_intent}] {texts_eval[i]!r}")
            print(f"        train: [{train_intent}] {texts_train[j]!r}")

    print(f"[near-dup] pairs whose intents disagree: {disagreeing}")

    # Per-row nearest-neighbor metrics against train: sims.max(axis=1)
    max_sims_per_row = sims.max(axis=1)
    above_thresh = int((max_sims_per_row > NEAR_DUP_THRESHOLD).sum())
    mean_max = float(max_sims_per_row.mean())
    highest_max = float(max_sims_per_row.max())

    print(f"\n[per-row max similarity to train] ({split_name})")
    print(f"    Rows with max sim > {NEAR_DUP_THRESHOLD}: {above_thresh} / {len(texts_eval)} ({above_thresh / len(texts_eval):.1%})")
    print(f"    Mean of max similarity: {mean_max:.4f}")
    print(f"    Max of max similarity : {highest_max:.4f}")

    return len(scored), disagreeing, above_thresh


def main():
    class_names = load_class_names()

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    exact_duplicates(train_texts, train_labels, "train")
    exact_duplicates(val_texts, val_labels, "val")
    exact_duplicates(test_texts, test_labels, "test")

    print("\n" + "-" * 50)
    print("CROSS-SPLIT EXACT OVERLAP")
    print("-" * 50)
    print("val against train:")
    exact_val_cross = cross_split_overlap(train_texts, val_texts)
    print("\ntest against train:")
    exact_test_cross = cross_split_overlap(train_texts, test_texts)

    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    X_test, y_test = load_features("test")
    print(f"\nembeddings: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    print("\n" + "-" * 50)
    print("VAL vs TRAIN NEAR-DUPLICATES & MAX SIMILARITY")
    print("-" * 50)
    near_val_cross, near_val_disagree, _ = near_duplicates(
        X_train, y_train, X_val, y_val, train_texts, val_texts, class_names, split_name="val"
    )

    print("\n" + "-" * 50)
    print("TEST vs TRAIN NEAR-DUPLICATES & MAX SIMILARITY")
    print("-" * 50)
    near_test_cross, near_test_disagree, _ = near_duplicates(
        X_train, y_train, X_test, y_test, train_texts, test_texts, class_names, split_name="test"
    )

    print("\n" + "=" * 60)
    total_exact = exact_val_cross + exact_test_cross
    total_disagree = near_val_disagree + near_test_disagree
    if total_exact == 0 and total_disagree == 0:
        print("VERDICT: clean")
    else:
        print(
            f"VERDICT: {exact_test_cross} exact test-train duplicates, "
            f"{exact_val_cross} exact val-train duplicates, "
            f"{total_disagree} near-duplicate pairs with disagreeing intents."
        )


if __name__ == "__main__":
    main()
