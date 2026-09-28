"""Mock in-memory banking backend simulator for testing and tool execution."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AccountRecord:
    account_type: str
    account_number: str
    balance: float
    due_date: Optional[str] = None
    is_frozen: bool = False


@dataclass
class CardRecord:
    card_brand: str
    last_four: str
    pin: str
    is_frozen: bool = False
    linked_account_type: str = "checking"


@dataclass
class DeclineRecord:
    decline_id: str
    timestamp: str
    amount: float
    card_last_four: str
    reason: str


@dataclass
class OrderRecord:
    order_id: str
    carrier: str
    status: str
    eta: str
    tracking_number: str


@dataclass
class UserData:
    user_id: str
    name: str
    accounts: Dict[str, AccountRecord] = field(default_factory=dict)
    cards: List[CardRecord] = field(default_factory=list)
    decline_history: List[DeclineRecord] = field(default_factory=list)
    orders: Dict[str, OrderRecord] = field(default_factory=dict)


def _get_initial_mock_db() -> Dict[str, UserData]:
    """Generates a seeded copy of customer database records."""
    return {
        "cust_1001": UserData(
            user_id="cust_1001",
            name="Alice Smith",
            accounts={
                "checking": AccountRecord(
                    account_type="checking",
                    account_number="CHK-1001-01",
                    balance=3450.75,
                ),
                "savings": AccountRecord(
                    account_type="savings",
                    account_number="SAV-1001-02",
                    balance=12500.00,
                ),
                "credit": AccountRecord(
                    account_type="credit",
                    account_number="CRD-1001-03",
                    balance=450.25,
                    due_date="2026-10-15",
                ),
            },
            cards=[
                CardRecord(
                    card_brand="visa",
                    last_four="4321",
                    pin="1234",
                    is_frozen=False,
                    linked_account_type="checking",
                ),
                CardRecord(
                    card_brand="mastercard",
                    last_four="9012",
                    pin="5678",
                    is_frozen=False,
                    linked_account_type="credit",
                ),
            ],
            decline_history=[
                DeclineRecord(
                    decline_id="DEC-8831",
                    timestamp="2026-09-24T14:30:00Z",
                    amount=120.00,
                    card_last_four="4321",
                    reason="Suspected fraud block on international transaction",
                )
            ],
            orders={
                "ORD-9821": OrderRecord(
                    order_id="ORD-9821",
                    carrier="FedEx",
                    status="Out for Delivery",
                    eta="Today by 6:00 PM",
                    tracking_number="FDX-99882211",
                ),
                "ORD-12345": OrderRecord(
                    order_id="ORD-12345",
                    carrier="UPS",
                    status="Shipped",
                    eta="October 2, 2026",
                    tracking_number="UPS-1Z9999999999999999",
                ),
            },
        ),
        "cust_1002": UserData(
            user_id="cust_1002",
            name="Bob Jones",
            accounts={
                "checking": AccountRecord(
                    account_type="checking",
                    account_number="CHK-1002-01",
                    balance=25.00,
                ),
                "credit": AccountRecord(
                    account_type="credit",
                    account_number="CRD-1002-03",
                    balance=850.00,
                    due_date="2026-10-01",
                ),
            },
            cards=[
                CardRecord(
                    card_brand="visa",
                    last_four="7788",
                    pin="9999",
                    is_frozen=True,
                    linked_account_type="credit",
                )
            ],
            decline_history=[],
            orders={},
        ),
    }


class BankingBackendSimulator:
    """Thread-safe in-memory database simulation with reset ability."""

    def __init__(self):
        self._data: Dict[str, UserData] = _get_initial_mock_db()

    def reset(self):
        """Restore backend state to initial seeded values."""
        self._data = _get_initial_mock_db()

    def get_user(self, user_id: str) -> Optional[UserData]:
        return self._data.get(user_id)


# Global singleton instance for app and testing
backend_db = BankingBackendSimulator()
