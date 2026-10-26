"""The API's format changes: the same checks still accept honest parses and
reject invented values, and moved records are still found."""

from agents.records import check_record, parse_records
from agents.registry import load_servers
from agents.processors import extract_texts
from tests.changed_api import ORDERS as CHANGED
from tests.fakes import FakeLLM

SCHEMA = load_servers()["customer-orders"]["record_schema"]

# What an honest LLM would extract from each changed-format record.
EXPECTED = [
    {"orderId": "1006", "buyer": "Ana Reyes", "city": "Toledo", "state": "OH", "total": 620.0, "items": ["tv"]},
    {"orderId": "1007", "buyer": "Ben Ortiz", "city": "Dayton", "state": "OH", "total": 1250.0, "items": ["sofa", "rug"]},
    {"orderId": "1008", "buyer": "Cara Nguyen", "city": "", "state": "TX", "total": 310.25, "items": []},
    {"orderId": "1009", "buyer": "Dev Patel", "city": "Akron", "state": "OH", "total": 75.0, "items": ["headphones"]},
]


def scripted_parse(messages):
    """Returns EXPECTED for whichever records are in the prompt."""
    out = []
    for line in messages[-1].content.splitlines():
        if line.startswith("["):
            i = int(line[1:line.index("]")])
            out.append({"index": i, **EXPECTED[i]})
    return {"records": out}


def test_new_formats_pass_checks():
    for rec, src in zip(EXPECTED, CHANGED):
        assert check_record(rec, src, SCHEMA) == [], src


def test_state_name_and_comma_totals_are_recognized():
    # "Ohio" spelled out, and "$1,250.00" with a comma
    assert check_record(EXPECTED[1], CHANGED[1], SCHEMA) == []


def test_missing_city_is_allowed():
    assert check_record(EXPECTED[2], CHANGED[2], SCHEMA) == []


def test_invented_values_still_rejected_in_new_format():
    assert check_record({**EXPECTED[0], "total": 602.0}, CHANGED[0], SCHEMA)
    assert check_record({**EXPECTED[0], "city": "Columbus"}, CHANGED[0], SCHEMA)
    assert check_record({**EXPECTED[2], "state": "OH"}, CHANGED[2], SCHEMA)


def test_parse_changed_records():
    orders, rejected = parse_records(FakeLLM(parse=scripted_parse), CHANGED, SCHEMA)
    assert len(orders) == 4 and rejected == []


def test_moved_records_key_is_found():
    payload = {"status": "ok", "data": CHANGED}
    assert extract_texts(payload, "raw_orders") == CHANGED
    assert extract_texts({"status": "ok", "data": CHANGED[0]}, "raw_order") == [CHANGED[0]]


def test_optional_fields_that_come_and_go():
    import dummy_customer_api as extended_api

    rows = extended_api.generate_rows(50)  # enough rows to find both cases
    with_all = next(r for r in rows if r["date"] and r["email"] and r["payment"])
    without = next(r for r in rows if not r["date"] and not r["payment"])
    for row in (with_all, without):
        src = row["format"](row)
        rec = {"orderId": str(row["id"]), "buyer": row["buyer"], "city": row["city"],
               "state": row["state"], "total": row["total"], "items": row["items"],
               "date": row["date"] or "", "email": row["email"] or "", "payment": row["payment"] or ""}
        assert check_record(rec, src, SCHEMA) == [], src
        assert check_record({**rec, "date": "2025-12-31"}, src, SCHEMA)  # invented date


def test_date_filters():
    from agents import records

    fields = load_servers()["customer-orders"]["record_fields"]
    march = records.validate_where([{"field": "date", "op": "gte", "value": "2026-03-01"},
                                    {"field": "date", "op": "lte", "value": "2026-03-31"}], fields)
    rows = [{"date": "2026-02-28"}, {"date": "2026-03-15"}, {"date": ""}]
    assert records.filter_records(rows, march) == [{"date": "2026-03-15"}]
    for bad in ({"field": "date", "op": "gt", "value": "March"},
                {"field": "email", "op": "gt", "value": "a"}):
        try:
            records.validate_where([bad], fields)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


def test_missing_fields_are_null_and_filterable():
    from agents import records

    rows = [{"orderId": "1", "email": None}, {"orderId": "2", "email": "a@b.com"}]
    fields = load_servers()["customer-orders"]["record_fields"]
    no_email = records.validate_where([{"field": "email", "op": "eq", "value": ""}], fields)
    assert [r["orderId"] for r in records.filter_records(rows, no_email)] == ["1"]
    assert records.select(rows, ["orderId", "email"], "orderId")[0]["email"] is None
