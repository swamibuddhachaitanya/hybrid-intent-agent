import pytest
from src.extraction.entity_extractor import EntityExtractor


@pytest.fixture
def extractor():
    return EntityExtractor()


def test_extract_numeric_amount_with_symbol(extractor):
    text = "Please pay $150.25 to my visa card"
    entities = extractor.extract(text)
    assert entities.amount == 150.25
    assert entities.currency == "USD"
    assert entities.card_brand == "visa"


def test_extract_word_amount(extractor):
    text = "Transfer fifty dollars to my checking account"
    entities = extractor.extract(text)
    assert entities.amount == 50.0
    assert entities.currency == "USD"
    assert entities.account_type == "checking"


def test_extract_card_ending_digits(extractor):
    text = "Please freeze my card ending in 4321 immediately"
    entities = extractor.extract(text)
    assert entities.last_four == "4321"


def test_extract_order_id(extractor):
    text = "Can you track my order #ORD-98214 please"
    entities = extractor.extract(text)
    assert entities.order_id == "ORD-98214"


def test_extract_pin_with_masking(extractor):
    text = "I would like to change my pin to 8842"
    entities = extractor.extract(text)
    assert entities.pin == "8842"
    # Ensure sensitive PIN is masked in audit logs
    masked = entities.masked_summary()
    assert masked["pin"] == "****"


def test_multi_entity_query(extractor):
    text = "Pay $250 on my mastercard ending 9012"
    entities = extractor.extract(text)
    assert entities.amount == 250.0
    assert entities.card_brand == "mastercard"
    assert entities.last_four == "9012"


def test_empty_when_no_entities(extractor):
    text = "Why was my card declined yesterday"
    entities = extractor.extract(text)
    assert entities.amount is None
    assert entities.card_brand is None
    assert entities.last_four is None
    assert entities.order_id is None
