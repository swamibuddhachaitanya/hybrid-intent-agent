import hashlib
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/data_full.json")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "ood_tiers.json"

SUPPORTED_INTENTS = {
    "order_status",
    "bill_balance",
    "pay_bill",
    "freeze_account",
    "pin_change",
    "card_declined",
}

# Explicit mapping and engineering rationale for hard boundary confusers
TIER_C_SPEC = {
    "balance": {
        "collides_with": "bill_balance",
        "reasoning": "Shares balance/inquiry tokens but targets depository/checking balances rather than credit liabilities.",
    },
    "bill_due": {
        "collides_with": "bill_balance",
        "reasoning": "Inquires about bill payment due dates rather than total outstanding balance amount.",
    },
    "order": {
        "collides_with": "order_status",
        "reasoning": "Requests initiation or purchase of an item rather than tracking an existing order shipment.",
    },
    "order_checks": {
        "collides_with": "order_status",
        "reasoning": "Colloquial 'order' collision requesting checkbooks rather than fulfillment tracking.",
    },
    "new_card": {
        "collides_with": "freeze_account",
        "reasoning": "Requests issuing a brand new card/account rather than locking an existing compromised one.",
    },
    "damaged_card": {
        "collides_with": "card_declined",
        "reasoning": "Reports physical card deterioration rather than transactional authorization or network decline.",
    },
    "account_blocked": {
        "collides_with": "freeze_account",
        "reasoning": "Describes involuntary security lockout rather than user-initiated preventative freeze.",
    },
    "pto_balance": {
        "collides_with": "bill_balance",
        "reasoning": "Colloquial 'balance' query regarding paid time off rather than financial statements.",
    },
    "replacement_card_duration": {
        "collides_with": "order_status",
        "reasoning": "Inquires about shipping timeframe for replacement plastic rather than order delivery status.",
    },
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Guard check: Snapshot existing split checksums
    protected_files = ["train.json", "val.json", "test.json"]

    hashes_before = {
        name: compute_file_hash(PROCESSED_DIR / name) for name in protected_files
    }

    # 2. Load existing train/val/test data for overlap checks
    existing_normalized = set()

    for name in protected_files:
        with open(PROCESSED_DIR / name, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                existing_normalized.add(normalize(item["text"]))

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []

    # 3. Tier A: Out-of-domain evaluation set (1,000 samples from oos_test)
    tier_a_raw = raw_data.get("oos_test", [])
    for text, label in tier_a_raw:
        records.append(
            {
                "tier": "A",
                "text": text,
                "source_tag": label,
                "collides_with": None,
                "reasoning": None,
            }
        )

    # 4. Extract test split samples for Tier B and Tier C
    test_samples = raw_data.get("test", [])
    tier_c_counts = {intent: 0 for intent in TIER_C_SPEC}
    tier_b_intents = set()

    for text, label in test_samples:
        if label in SUPPORTED_INTENTS:
            continue
        if label in TIER_C_SPEC:
            spec = TIER_C_SPEC[label]
            records.append(
                {
                    "tier": "C",
                    "text": text,
                    "source_tag": label,
                    "collides_with": spec["collides_with"],
                    "reasoning": spec["reasoning"],
                }
            )
            tier_c_counts[label] += 1
        else:
            records.append(
                {
                    "tier": "B",
                    "text": text,
                    "source_tag": label,
                    "collides_with": None,
                    "reasoning": None,
                }
            )
            tier_b_intents.add(label)

    # 5. Verify zero cross-split contamination
    collisions = [
        r["text"] for r in records if normalize(r["text"]) in existing_normalized
    ]
    if collisions:
        raise ValueError(
            f"Integrity check failed: Found {len(collisions)} overlapping queries in OOD set! "
            f"First collision: {collisions[0]}"
        )

    # 6. Save deliverable
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    # 7. Verify immutability of protected splits
    hashes_after = {
        name: compute_file_hash(PROCESSED_DIR / name) for name in protected_files
    }
    for name in protected_files:
        assert hashes_before[name] == hashes_after[name], (
            f"Fatal: {name} was modified during script execution!"
        )

    # 8. Report audit metrics
    count_a = sum(1 for r in records if r["tier"] == "A")
    count_b = sum(1 for r in records if r["tier"] == "B")
    count_c = sum(1 for r in records if r["tier"] == "C")

    print("=" * 60)
    print("OOD TIERS DATASET SUMMARY")
    print("=" * 60)
    print(f"Total OOD records generated: {len(records)}")
    print(f"  Tier A (Out of domain):         {count_a}")
    print(
        f"  Tier B (In domain, unsupported): {count_b} across {len(tier_b_intents)} intents"
    )
    print(f"  Tier C (Hard boundary confusers):{count_c}")
    print("\nTier C Breakdown:")
    for intent, count in sorted(tier_c_counts.items()):
        spec = TIER_C_SPEC[intent]
        print(
            f"  - {intent:<28} (n={count:>2}) -> collides with '{spec['collides_with']}'"
        )
    print("\nIntegrity Checks:")
    print("  - Overlap with train/val/test:    0 (PASSED)")
    print("  - train/val/test hash immutability: PASSED")
    print(f"Saved artifact to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
