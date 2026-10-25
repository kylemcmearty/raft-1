"""Bad LLM output must be rejected, never passed through."""

from agents.records import check_record, parse_records
from agents.registry import load_servers
from tests.fakes import ORDERS, FakeLLM, honest_parse

SCHEMA = load_servers()["customer-orders"]["record_schema"]
SRC = ORDERS[0]  # Order 1001: John Davis, Columbus, OH, $742.10
GOOD = {"orderId": "1001", "buyer": "John Davis", "city": "Columbus", "state": "OH",
        "total": 742.10, "items": ["laptop", "hdmi cable"]}


def test_good_record_passes():
    assert check_record(GOOD, SRC, SCHEMA) == []


def test_wrong_total_rejected():
    assert check_record({**GOOD, "total": 724.10}, SRC, SCHEMA)


def test_invented_buyer_rejected():
    assert check_record({**GOOD, "buyer": "Jane Doe"}, SRC, SCHEMA)


def test_wrong_state_rejected():
    assert check_record({**GOOD, "state": "TX"}, SRC, SCHEMA)
    assert check_record({**GOOD, "state": "Ohio"}, SRC, SCHEMA)  # must be a code


def test_invented_item_rejected():
    assert check_record({**GOOD, "items": ["laptop", "mouse"]}, SRC, SCHEMA)


def test_total_as_string_rejected():
    assert check_record({**GOOD, "total": "$742.10"}, SRC, SCHEMA)


def test_honest_parse_accepts_all():
    orders, rejected = parse_records(FakeLLM(), ORDERS, SCHEMA)
    assert len(orders) == 5 and rejected == []


def test_invented_extra_order_dropped():
    def parse(messages):
        out = honest_parse(messages)
        out["records"].append({**GOOD, "index": 99, "orderId": "9999"})
        return out

    orders, _ = parse_records(FakeLLM(parse=parse), ORDERS, SCHEMA)
    assert "9999" not in {o["orderId"] for o in orders}


def test_dropped_record_retried_then_rejected():
    def parse(messages):  # always "forgets" order 1003
        out = honest_parse(messages)
        out["records"] = [o for o in out["records"] if o["orderId"] != "1003"]
        return out

    llm = FakeLLM(parse=parse)
    orders, rejected = parse_records(llm, ORDERS, SCHEMA)
    assert len(orders) == 4 and len(rejected) == 1
    assert llm.parse_calls == 2  # one retry


def test_retry_fixes_record():
    attempts = {"n": 0}

    def parse(messages):
        attempts["n"] += 1
        out = honest_parse(messages)
        if attempts["n"] == 1:
            out["records"][0]["total"] = 1.0  # hallucinated on the first try
        return out

    orders, rejected = parse_records(FakeLLM(parse=parse), ORDERS, SCHEMA)
    assert len(orders) == 5 and rejected == []


def test_llm_crash_rejects_instead_of_guessing():
    def parse(messages):
        raise RuntimeError("provider error")

    orders, rejected = parse_records(FakeLLM(parse=parse), ORDERS, SCHEMA)
    assert orders == [] and len(rejected) == 5


def test_large_input_is_batched():
    many = [f"Order {2000 + i}: Buyer=A B, Location=X, OH, Total=$1.00, Items: y" for i in range(100)]
    llm = FakeLLM()
    orders, _ = parse_records(llm, many, SCHEMA)
    assert len(orders) == 100 and llm.parse_calls == 5  # 20 per batch


def test_provider_error_stops_instead_of_rejecting():
    import httpx
    import openai
    import pytest

    def parse(messages):  # e.g. out of credits, or the provider is down
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai"))

    with pytest.raises(openai.APIError):
        parse_records(FakeLLM(parse=parse), ORDERS, SCHEMA)


def test_duplicate_ids_are_all_rejected():
    # Which copy comes "first" depends on the API's random order, so neither is kept.
    fake = "Order 1001: Buyer=Kim Fake, Location=Austin, TX, Total=$99.99, Items: mouse"
    for texts in (ORDERS + [fake], [fake] + ORDERS):
        orders, rejected = parse_records(FakeLLM(), texts, SCHEMA)
        assert "1001" not in {o["orderId"] for o in orders}
        assert len(orders) == 4 and len(rejected) == 2
