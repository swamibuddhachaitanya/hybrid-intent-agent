"""Deterministic rule-based entity extractor for banking and order operations.

Extracts:
- monetary amounts (currency + float value)
- card brands / account types (visa, mastercard, checking, savings, etc.)
- account identifiers / last 4 digits (ending in 1234, #5678)
- order identifiers (order #ORD-9821, order 12345)
- PIN digits (4-digit sequences in pin-change contexts)

PCI-DSS note: PIN entities are explicitly typed and masked in standard representations.
"""

from dataclasses import dataclass, field
import re
from typing import Optional


@dataclass(frozen=True)
class ExtractedEntities:
    amount: Optional[float] = None
    currency: Optional[str] = None
    card_brand: Optional[str] = None
    account_type: Optional[str] = None
    last_four: Optional[str] = None
    order_id: Optional[str] = None
    pin: Optional[str] = None
    raw_matches: dict = field(default_factory=dict)

    def masked_summary(self) -> dict:
        """Safe dictionary representation with credential masking."""
        d = {
            "amount": self.amount,
            "currency": self.currency,
            "card_brand": self.card_brand,
            "account_type": self.account_type,
            "last_four": self.last_four,
            "order_id": self.order_id,
            "pin": "****" if self.pin else None,
        }
        return {k: v for k, v in d.items() if v is not None}


class EntityExtractor:
    # Common currency symbols and word maps
    CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}

    WORD_TO_NUMBER = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
    }

    # Regex definitions
    RE_AMOUNT_NUMERIC = re.compile(
        r"(?:(?P<symbol>[\$€£])\s*)?(?P<num>\d+(?:\.\d{1,2})?)\s*(?:(?P<currency_word>dollars|usd|bucks|cents|eur|gbp))?",
        re.IGNORECASE,
    )

    RE_AMOUNT_WORDS = re.compile(
        r"\b(?P<words>(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|hundred))*)\s+(?P<unit>dollars|bucks|cents|usd)\b",
        re.IGNORECASE,
    )

    RE_CARD_BRAND = re.compile(
        r"\b(?P<brand>visa|mastercard|master\s*card|amex|american\s*express|discover)\b",
        re.IGNORECASE,
    )

    RE_ACCOUNT_TYPE = re.compile(
        r"\b(?P<type>checking|savings|credit\s*card|debit\s*card|bankcard)\b",
        re.IGNORECASE,
    )

    RE_LAST_FOUR = re.compile(
        r"(?:ending(?:\s+in)?|account\s*(?:#|number)?|card\s*(?:#|number)?|last\s*(?:four|4)?(?:\s*digits)?)\s*[:#]?\s*(?P<digits>\d{4})\b",
        re.IGNORECASE,
    )

    RE_ORDER_ID = re.compile(
        r"(?:order\s*(?:#|number|id)?|tracking\s*(?:#|number)?)\s*[:#]?\s*(?P<order>[A-Za-z0-9_-]{4,20})\b",
        re.IGNORECASE,
    )

    RE_PIN = re.compile(
        r"(?:pin|passcode|code)\s*(?:to|is|number)?\s*[:#]?\s*(?P<pin>\d{4})\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> ExtractedEntities:
        """Extract structured entities from input query text."""
        raw_matches = {}

        # 1. Extract Amount
        amount = None
        currency = None

        # Try numeric amount first
        numeric_match = self.RE_AMOUNT_NUMERIC.search(text)
        if numeric_match and (
            numeric_match.group("symbol")
            or numeric_match.group("currency_word")
            or "$" in text
        ):
            num_str = numeric_match.group("num")
            try:
                val = float(num_str)
                if (
                    numeric_match.group("currency_word")
                    and numeric_match.group("currency_word").lower() == "cents"
                ):
                    val = val / 100.0
                amount = val
                symbol = numeric_match.group("symbol")
                currency = self.CURRENCY_SYMBOLS.get(symbol, "USD")
                raw_matches["amount_match"] = numeric_match.group(0)
            except ValueError:
                pass

        # Try word-based amount if not found (e.g. "fifty dollars")
        if amount is None:
            word_match = self.RE_AMOUNT_WORDS.search(text)
            if word_match:
                words = word_match.group("words").lower().split()
                total = 0
                for w in words:
                    if w in self.WORD_TO_NUMBER:
                        if w == "hundred" and total > 0:
                            total *= 100
                        else:
                            total += self.WORD_TO_NUMBER[w]
                if total > 0:
                    amount = float(total)
                    currency = "USD"
                    raw_matches["amount_match"] = word_match.group(0)

        # 2. Extract Card Brand
        card_brand = None
        brand_match = self.RE_CARD_BRAND.search(text)
        if brand_match:
            b = brand_match.group("brand").lower()
            if "american" in b or "amex" in b:
                card_brand = "amex"
            elif "master" in b:
                card_brand = "mastercard"
            else:
                card_brand = b
            raw_matches["card_brand"] = brand_match.group(0)

        # 3. Extract Account Type
        account_type = None
        acct_match = self.RE_ACCOUNT_TYPE.search(text)
        if acct_match:
            a = acct_match.group("type").lower()
            if "credit" in a:
                account_type = "credit"
            elif "debit" in a:
                account_type = "debit"
            elif "checking" in a:
                account_type = "checking"
            elif "savings" in a:
                account_type = "savings"
            else:
                account_type = a
            raw_matches["account_type"] = acct_match.group(0)

        # 4. Extract Last Four Digits
        last_four = None
        four_match = self.RE_LAST_FOUR.search(text)
        if four_match:
            last_four = four_match.group("digits")
            raw_matches["last_four"] = last_four

        # 5. Extract Order ID
        order_id = None
        order_match = self.RE_ORDER_ID.search(text)
        if order_match:
            order_id = order_match.group("order").upper()
            raw_matches["order_id"] = order_id

        # 6. Extract PIN (in context of pin)
        pin = None
        pin_match = self.RE_PIN.search(text)
        if pin_match:
            pin = pin_match.group("pin")
            raw_matches["pin"] = pin

        return ExtractedEntities(
            amount=amount,
            currency=currency,
            card_brand=card_brand,
            account_type=account_type,
            last_four=last_four,
            order_id=order_id,
            pin=pin,
            raw_matches=raw_matches,
        )
