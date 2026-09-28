"""Deterministic tool execution layer for banking and order operations.

Safe, audited execution layer that performs strict business logic validations
and state mutations. Never interacts directly with raw DB/NLU.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional
from src.tools.banking_backend import BankingBackendSimulator, UserData, backend_db


@dataclass
class ToolResult:
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None


class ToolExecutor:
    """Executes validated domain actions against the banking backend."""

    def __init__(self, backend: Optional[BankingBackendSimulator] = None):
        self.backend = backend or backend_db

    def _resolve_card(
        self,
        user: UserData,
        card_brand: Optional[str] = None,
        last_four: Optional[str] = None,
        account_type: Optional[str] = None,
    ):
        """Finds matching card record by brand, last four, or linked account."""
        candidates = user.cards
        if last_four:
            candidates = [c for c in candidates if c.last_four == last_four]
        if card_brand:
            brand_norm = card_brand.lower()
            candidates = [c for c in candidates if brand_norm in c.card_brand.lower()]
        if account_type:
            type_norm = account_type.lower()
            candidates = [
                c for c in candidates if type_norm in c.linked_account_type.lower()
            ]

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Ambiguous match
            return candidates[0]
        return None

    def tool_pay_bill(
        self,
        user_id: str,
        amount: Optional[float],
        card_brand: Optional[str] = None,
        last_four: Optional[str] = None,
    ) -> ToolResult:
        """Deducts amount from customer bill balance (credit account)."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        if amount is None or amount <= 0:
            return ToolResult(
                success=False,
                message="Payment amount must be greater than zero.",
                error_code="INVALID_AMOUNT",
            )

        credit_acct = user.accounts.get("credit")
        if not credit_acct:
            return ToolResult(
                success=False,
                message="No credit/bill account found for this user.",
                error_code="ACCOUNT_NOT_FOUND",
            )

        if credit_acct.is_frozen:
            return ToolResult(
                success=False,
                message="Account is frozen. Cannot process bill payment.",
                error_code="ACCOUNT_FROZEN",
            )

        # Process payment
        prev_balance = credit_acct.balance
        credit_acct.balance = max(0.0, round(credit_acct.balance - amount, 2))

        return ToolResult(
            success=True,
            message=f"Successfully processed payment of ${amount:.2f}. Remaining balance: ${credit_acct.balance:.2f}.",
            data={
                "previous_balance": prev_balance,
                "amount_paid": amount,
                "remaining_balance": credit_acct.balance,
                "account_number": credit_acct.account_number,
            },
        )

    def tool_bill_balance(
        self,
        user_id: str,
        account_type: Optional[str] = None,
    ) -> ToolResult:
        """Fetches current statement balance and due date."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        target_type = (account_type or "credit").lower()
        acct = user.accounts.get(target_type)

        # Fallback to credit if unspecified or not found directly
        if not acct:
            if "credit" in user.accounts:
                acct = user.accounts["credit"]
            else:
                return ToolResult(
                    success=False,
                    message=f"No account matching '{target_type}' found.",
                    error_code="ACCOUNT_NOT_FOUND",
                )

        return ToolResult(
            success=True,
            message=f"Your {acct.account_type} balance is ${acct.balance:.2f}."
            + (f" Payment due date: {acct.due_date}." if acct.due_date else ""),
            data={
                "account_type": acct.account_type,
                "balance": acct.balance,
                "due_date": acct.due_date,
                "account_number": acct.account_number,
                "is_frozen": acct.is_frozen,
            },
        )

    def tool_freeze_account(
        self,
        user_id: str,
        card_brand: Optional[str] = None,
        last_four: Optional[str] = None,
        account_type: Optional[str] = None,
    ) -> ToolResult:
        """Freezes a specified card or account."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        # First check cards
        card = self._resolve_card(
            user, card_brand=card_brand, last_four=last_four, account_type=account_type
        )
        if card:
            if card.is_frozen:
                return ToolResult(
                    success=False,
                    message=f"Your {card.card_brand.capitalize()} card ending in {card.last_four} is already frozen.",
                    error_code="ALREADY_FROZEN",
                )
            card.is_frozen = True
            return ToolResult(
                success=True,
                message=f"Successfully froze your {card.card_brand.capitalize()} card ending in {card.last_four}.",
                data={
                    "card_brand": card.card_brand,
                    "last_four": card.last_four,
                    "is_frozen": True,
                },
            )

        # If no specific card, check account type
        if account_type and account_type.lower() in user.accounts:
            acct = user.accounts[account_type.lower()]
            if acct.is_frozen:
                return ToolResult(
                    success=False,
                    message=f"Your {account_type} account is already frozen.",
                    error_code="ALREADY_FROZEN",
                )
            acct.is_frozen = True
            return ToolResult(
                success=True,
                message=f"Successfully froze your {account_type} account ({acct.account_number}).",
                data={
                    "account_type": account_type,
                    "account_number": acct.account_number,
                    "is_frozen": True,
                },
            )

        return ToolResult(
            success=False,
            message="Could not find a matching card or account to freeze.",
            error_code="CARD_NOT_FOUND",
        )

    def tool_pin_change(
        self,
        user_id: str,
        new_pin: Optional[str],
        card_brand: Optional[str] = None,
        last_four: Optional[str] = None,
    ) -> ToolResult:
        """Updates PIN for a specified card after validation."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        if not new_pin or not re.match(r"^\d{4}$", str(new_pin)):
            return ToolResult(
                success=False,
                message="PIN must be exactly 4 numeric digits.",
                error_code="INVALID_PIN_FORMAT",
            )

        # Avoid weak PINs
        if new_pin in {"0000", "1111", "1234", "9999"}:
            return ToolResult(
                success=False,
                message="PIN is too common or weak. Please select a stronger 4-digit PIN.",
                error_code="WEAK_PIN",
            )

        card = self._resolve_card(user, card_brand=card_brand, last_four=last_four)
        if not card:
            # If user has only one card, default to it
            if len(user.cards) == 1:
                card = user.cards[0]
            else:
                return ToolResult(
                    success=False,
                    message="Please specify which card you want to change the PIN for.",
                    error_code="CARD_NOT_SPECIFIED",
                )

        if card.is_frozen:
            return ToolResult(
                success=False,
                message="Cannot change PIN on a frozen card.",
                error_code="CARD_FROZEN",
            )

        card.pin = new_pin
        return ToolResult(
            success=True,
            message=f"PIN for {card.card_brand.capitalize()} ending in {card.last_four} updated successfully.",
            data={
                "card_brand": card.card_brand,
                "last_four": card.last_four,
                "pin_updated": True,
            },
        )

    def tool_card_declined(self, user_id: str) -> ToolResult:
        """Fetches the reason and details for the most recent card decline."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        if not user.decline_history:
            return ToolResult(
                success=True,
                message="No recent card declines found on your account.",
                data={"has_declines": False},
            )

        latest_decline = user.decline_history[-1]
        return ToolResult(
            success=True,
            message=f"Your card ending in {latest_decline.card_last_four} was declined for ${latest_decline.amount:.2f}. Reason: {latest_decline.reason}.",
            data={
                "has_declines": True,
                "decline_id": latest_decline.decline_id,
                "timestamp": latest_decline.timestamp,
                "amount": latest_decline.amount,
                "card_last_four": latest_decline.card_last_four,
                "reason": latest_decline.reason,
            },
        )

    def tool_order_status(self, user_id: str, order_id: Optional[str]) -> ToolResult:
        """Returns shipping and delivery status for an order."""
        user = self.backend.get_user(user_id)
        if not user:
            return ToolResult(
                success=False,
                message=f"User {user_id} not found.",
                error_code="USER_NOT_FOUND",
            )

        if not order_id:
            return ToolResult(
                success=False,
                message="Order ID is required to look up tracking.",
                error_code="ORDER_ID_REQUIRED",
            )

        norm_order_id = order_id.strip().upper()
        # Allow checking with or without #
        norm_order_id = norm_order_id.lstrip("#")
        # Search keys with and without #
        order = user.orders.get(norm_order_id) or user.orders.get(f"#{norm_order_id}")
        if not order:
            # Check case-insensitive match
            for k, v in user.orders.items():
                if k.upper().lstrip("#") == norm_order_id:
                    order = v
                    break

        if not order:
            return ToolResult(
                success=False,
                message=f"Order {order_id} could not be found for your account.",
                error_code="ORDER_NOT_FOUND",
            )

        return ToolResult(
            success=True,
            message=f"Order {order.order_id} via {order.carrier} is currently {order.status}. Estimated arrival: {order.eta}.",
            data={
                "order_id": order.order_id,
                "carrier": order.carrier,
                "status": order.status,
                "eta": order.eta,
                "tracking_number": order.tracking_number,
            },
        )
