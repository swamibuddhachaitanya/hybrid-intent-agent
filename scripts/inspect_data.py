import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/raw/data_full.json")


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found")
        return
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print("Splits found:")
    for split_name, samples in data.items():
        print(f"  {split_name}: {len(samples)} samples")

    # In-scope training inspection
    train_samples = data["train"]

    labels = [label for text, label in train_samples]
    counts = Counter(labels)

    print(f"\n Unique in-scope intents: {len(counts)}")
    print(
        f" Samples per intent: min={min(counts.values())}, max={max(counts.values())}"
    )

    # look for support-related intents
    keywords = ("card", "order", "transfer", "balance", "pin", "bill", "account")
    matches = sorted({label for label in counts if any(k in label for k in keywords)})

    print(f"\n Matching candidate intents ({len(matches)}) found")
    for tag in matches[:15]:
        print(f"  {tag}")


if __name__ == "__main__":
    main()
