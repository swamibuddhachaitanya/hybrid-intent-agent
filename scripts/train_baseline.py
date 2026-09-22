"""Train the Phase 1 intent classifier on cached MiniLM embeddings.

Design decisions worth defending:
  - encoder is frozen, so embeddings are precomputed (see prepare_features.py)
  - checkpoint is selected by val_loss, not val_accuracy
  - test is evaluated once, after selection
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation")

SEED = 42
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20
EMBEDDING_DIM = 384
ENCODER_NAME = "all-MiniLM-L6-v2"
MODEL_PATH = MODELS_DIR / "baseline_intent.keras"


def load_features(split_name: str):
    data = np.load(PROCESSED_DIR / f"{split_name}_features.npz")
    X, y = data["X"], data["y"]
    assert X.ndim == 2 and X.shape[1] == EMBEDDING_DIM, (
        f"{split_name}: bad X shape {X.shape}"
    )
    assert X.shape[0] == y.shape[0], f"{split_name}: X/y row mismatch"
    return X, y


def load_class_names():
    with open(PROCESSED_DIR / "label_map.json", "r", encoding="utf-8") as f:
        idx_to_label = json.load(f)["idx_to_label"]
    return [idx_to_label[str(i)] for i in range(len(idx_to_label))]


def get_encoder_revision():
    """Commit hash of the cached encoder snapshot, or None if not found."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = (
        hf_home / "hub" / f"models--sentence-transformers--{ENCODER_NAME}" / "snapshots"
    )
    if not snapshots.exists():
        return None
    commits = sorted(p.name for p in snapshots.iterdir() if p.is_dir())
    return commits[0] if commits else None


def build_model(input_dim: int, num_classes: int):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def main():
    # must run before any layer is built, or initial weights are unseeded
    keras.utils.set_random_seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    X_test, y_test = load_features("test")

    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"classes ({len(class_names)}): {class_names}")

    model = build_model(X_train.shape[1], y_train.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    val_losses = history.history["val_loss"]
    val_accs = history.history["val_accuracy"]
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(val_losses)) + 1
    final_val_loss = float(val_losses[-1])

    print(
        f"\nepochs run        : {epochs_run} (early stop fired: {epochs_run < EPOCHS})"
    )
    print(f"final val loss    : {final_val_loss:.6f}")
    print(f"best epoch        : {best_epoch}")
    print(f"best val loss     : {float(val_losses[best_epoch - 1]):.6f}")
    print(f"best val accuracy : {float(val_accs[best_epoch - 1]):.6f}")

    # test is touched here, once, after selection
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\ntest loss        : {float(test_loss):.6f}")
    print(f"test accuracy    : {float(test_acc):.6f}")
    print(f"test macro F1    : {float(report['macro avg']['f1-score']):.4f}")
    print(f"test weighted F1 : {float(report['weighted avg']['f1-score']):.4f}")

    print("\nper class:")
    for name in class_names:
        r = report[name]
        print(
            f"  {name:<18} P {float(r['precision']):.4f}  "
            f"R {float(r['recall']):.4f}  F1 {float(r['f1-score']):.4f}  "
            f"n {int(r['support'])}"
        )

    print("\nconfusion matrix (rows=true, cols=pred):")
    print("  " + " ".join(f"{n[:8]:>9}" for n in class_names))
    for name, row in zip(class_names, cm):
        print(f"  {name[:12]:<12} " + " ".join(f"{v:>9}" for v in row))

    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "encoder": {"name": ENCODER_NAME, "revision": get_encoder_revision()},
        "dataset": {
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
            "classes": class_names,
            "provenance": {
                "raw_test_rows": 180,
                "exact_train_test_duplicates_removed": 1,
                "exact_duplicate_text": "i need my account frozen",
                "near_duplicate_cosine_threshold": 0.95,
                "test_rows_with_near_twin_in_train": 24,
                "test_rows_with_near_twin_in_train_pct": 0.1341,
            },
        },
        "model": {
            "architecture": "384 -> 128 -> 64 -> 6 dense, dropout 0.3",
            "params": int(model.count_params()),
            "max_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "training": {
            "epochs_run": epochs_run,
            "early_stopped": epochs_run < EPOCHS,
            "best_epoch": best_epoch,
            "final_val_loss": final_val_loss,
            "best_val_loss": float(val_losses[best_epoch - 1]),
            "best_val_accuracy": float(val_accs[best_epoch - 1]),
        },
        "test": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "per_class": {
                name: {
                    "precision": float(report[name]["precision"]),
                    "recall": float(report[name]["recall"]),
                    "f1": float(report[name]["f1-score"]),
                    "support": int(report[name]["support"]),
                }
                for name in class_names
            },
            "confusion_matrix": cm.tolist(),
        },
    }

    with open(EVAL_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nwrote {EVAL_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
