import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/data_full.json")
PROCESSED_DIR = Path("data/processed")

TARGET_INTENTS = [
    "order_status",
    "bill_balance",
    "pay_bill",
    "freeze_account",
    "pin_change",
    "card_declined",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def filter_split(samples, allowed_intents):
    allowed_set = set(allowed_intents)
    return [
        {"text": text, "intent": label}
        for text, label in samples
        if label in allowed_set
    ]


def main():
    with open(RAW_PATH, "r") as f:
        raw_data = json.load(f)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Targeting {len(TARGET_INTENTS)} intents: {TARGET_INTENTS}\n")

    train_data = filter_split(raw_data["train"], TARGET_INTENTS)
    val_data = filter_split(raw_data["val"], TARGET_INTENTS)
    test_raw = filter_split(raw_data["test"], TARGET_INTENTS)

    # Deduplicate: drop any normalized test row that also exists in train
    train_norm_set = {normalize(item["text"]) for item in train_data}
    test_data = [
        item for item in test_raw if normalize(item["text"]) not in train_norm_set
    ]
    dropped_count = len(test_raw) - len(test_data)
    print(
        f"[dedupe] Dropped {dropped_count} test row(s) that appeared in train set."
    )

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    for split, filtered in splits.items():
        out_file = PROCESSED_DIR / f"{split}.json"
        with open(out_file, "w", encoding="utf-8") as out_f:
            json.dump(filtered, out_f, indent=2)
        print(f"Saved {split} set with {len(filtered)} samples to {out_file}")


if __name__ == "__main__":
    main()
