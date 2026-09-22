import json
from pathlib import Path

import numpy as np

# pyrefly: ignore [missing-import]
from src.preprocessing.embedding_preprocessor import EmbeddingPreprocessor

PROCESSED_DIR = Path("data/processed")


def one_hot_encode(labels, label_to_idx):
    num_classes = len(label_to_idx)
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        idx = label_to_idx[label]
        encoded[i, idx] = 1.0
    return encoded


def main():
    # 1. Load raw processed JSON files
    with open(PROCESSED_DIR / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(PROCESSED_DIR / "val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(PROCESSED_DIR / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 2. Build deterministic label map from training set
    unique_intents = sorted(list({item["intent"] for item in train_data}))
    label_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
    idx_to_label = {i: intent for i, intent in enumerate(unique_intents)}

    with open(PROCESSED_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2
        )

    print(f"Label map saved with {len(label_to_idx)} classes")

    for k, v in label_to_idx.items():
        print(f"{k}: {v}")

    # compute embeddings
    print("\nComputing embeddings with all-MiniLM-L6-v2...")
    preprocessor = EmbeddingPreprocessor()

    for split_name, data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        texts = [item["text"] for item in data]
        labels = [item["intent"] for item in data]

        X = preprocessor.transform(texts)
        Y = one_hot_encode(labels, label_to_idx)

        out_path = PROCESSED_DIR / f"{split_name}_features.npz"
        np.savez_compressed(out_path, X=X, y=Y)
        print(
            f"Saved {split_name}_features.npz -> X shape: {X.shape}, y shape: {Y.shape}"
        )


if __name__ == "__main__":
    main()
