import pytest
from src.tools.banking_backend import BankingBackendSimulator
from src.tools.tool_executor import ToolExecutor


@pytest.fixture
def backend():
    b = BankingBackendSimulator()
    b.reset()
    return b


@pytest.fixture
def executor(backend):
    return ToolExecutor(backend=backend)


# 1. tool_pay_bill tests
def test_pay_bill_success(executor):
    res = executor.tool_pay_bill("cust_1001", amount=150.0)
    assert res.success is True
    assert res.data["amount_paid"] == 150.0
    assert res.data["remaining_balance"] == 300.25  # 450.25 - 150.0


def test_pay_bill_invalid_amount(executor):
    res_zero = executor.tool_pay_bill("cust_1001", amount=0.0)
    assert res_zero.success is False
    assert res_zero.error_code == "INVALID_AMOUNT"

    res_negative = executor.tool_pay_bill("cust_1001", amount=-50.0)
    assert res_negative.success is False
    assert res_negative.error_code == "INVALID_AMOUNT"


def test_pay_bill_user_not_found(executor):
    res = executor.tool_pay_bill("non_existent_user", amount=100.0)
    assert res.success is False
    assert res.error_code == "USER_NOT_FOUND"


# 2. tool_bill_balance tests
def test_bill_balance_success(executor):
    res = executor.tool_bill_balance("cust_1001")
    assert res.success is True
    assert res.data["account_type"] == "credit"
    assert res.data["balance"] == 450.25
    assert res.data["due_date"] == "2026-10-15"


def test_bill_balance_specific_account(executor):
    res = executor.tool_bill_balance("cust_1001", account_type="checking")
    assert res.success is True
    assert res.data["account_type"] == "checking"
    assert res.data["balance"] == 3450.75


# 3. tool_freeze_account tests
def test_freeze_account_success(executor):
    res = executor.tool_freeze_account("cust_1001", card_brand="visa", last_four="4321")
    assert res.success is True
    assert res.data["is_frozen"] is True

    # Attempting to freeze again should fail with ALREADY_FROZEN
    res_repeat = executor.tool_freeze_account("cust_1001", card_brand="visa", last_four="4321")
    assert res_repeat.success is False
    assert res_repeat.error_code == "ALREADY_FROZEN"


def test_freeze_account_not_found(executor):
    res = executor.tool_freeze_account("cust_1001", card_brand="amex", last_four="1111")
    assert res.success is False
    assert res.error_code == "CARD_NOT_FOUND"


# 4. tool_pin_change tests
def test_pin_change_success(executor):
    res = executor.tool_pin_change("cust_1001", new_pin="7429", last_four="4321")
    assert res.success is True
    assert res.data["pin_updated"] is True


def test_pin_change_invalid_format(executor):
    res_short = executor.tool_pin_change("cust_1001", new_pin="12", last_four="4321")
    assert res_short.success is False
    assert res_short.error_code == "INVALID_PIN_FORMAT"

    res_alpha = executor.tool_pin_change("cust_1001", new_pin="abcd", last_four="4321")
    assert res_alpha.success is False
    assert res_alpha.error_code == "INVALID_PIN_FORMAT"


def test_pin_change_weak_pin(executor):
    res = executor.tool_pin_change("cust_1001", new_pin="1234", last_four="4321")
    assert res.success is False
    assert res.error_code == "WEAK_PIN"


def test_pin_change_frozen_card(executor):
    # cust_1002 has a pre-frozen visa card 7788
    res = executor.tool_pin_change("cust_1002", new_pin="8492", last_four="7788")
    assert res.success is False
    assert res.error_code == "CARD_FROZEN"


# 5. tool_card_declined tests
def test_card_declined_with_history(executor):
    res = executor.tool_card_declined("cust_1001")
    assert res.success is True
    assert res.data["has_declines"] is True
    assert res.data["card_last_four"] == "4321"
    assert "Suspected fraud" in res.data["reason"]


def test_card_declined_no_history(executor):
    res = executor.tool_card_declined("cust_1002")
    assert res.success is True
    assert res.data["has_declines"] is False


# 6. tool_order_status tests
def test_order_status_success(executor):
    res = executor.tool_order_status("cust_1001", order_id="ORD-9821")
    assert res.success is True
    assert res.data["carrier"] == "FedEx"
    assert res.data["status"] == "Out for Delivery"


def test_order_status_with_hash(executor):
    res = executor.tool_order_status("cust_1001", order_id="#ORD-12345")
    assert res.success is True
    assert res.data["carrier"] == "UPS"


def test_order_status_not_found(executor):
    res = executor.tool_order_status("cust_1001", order_id="ORD-NON-EXISTENT")
    assert res.success is False
    assert res.error_code == "ORDER_NOT_FOUND"


def test_order_status_missing_id(executor):
    res = executor.tool_order_status("cust_1001", order_id="")
    assert res.success is False
    assert res.error_code == "ORDER_ID_REQUIRED"
