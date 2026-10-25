"""End-to-end through the LangGraph graph with a scripted LLM and fake API."""

import json

import pytest

from agents import records
from agents.graph import DEFAULT_UNSUPPORTED, run_agent
from agents.registry import load_servers
from tests.fakes import FakeLLM, call, fake_api
from langchain_core.messages import AIMessage

OHIO_OVER_500 = [{"field": "state", "op": "eq", "value": "Ohio"},
                 {"field": "total", "op": "gt", "value": 500}]


def script(invoke_args):
    return [
        call("discover_server", {"query": "orders"}, "1"),
        call("discover_tools", {"server": "customer-orders"}, "2"),
        call("get_tool_schema", {"tool_name": invoke_args["tool_name"]}, "3"),
        call("invoke_tool", invoke_args, "4"),
        AIMessage("done"),
    ]


def run(invoke_args):
    return run_agent("q", FakeLLM(script(invoke_args)), get=fake_api())


def test_ohio_over_500():
    out = run({"tool_name": "list_orders", "where": OHIO_OVER_500})
    assert out == {"orders": [
        {"orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.1},
        {"orderId": "1003", "buyer": "Mike Turner", "state": "OH", "total": 1299.99},
        {"orderId": "1005", "buyer": "Chris Myers", "state": "OH", "total": 512.0},
    ]}


def test_deterministic_despite_shuffled_api():
    outs = {json.dumps(run({"tool_name": "list_orders", "where": OHIO_OVER_500})) for _ in range(5)}
    assert len(outs) == 1


def test_get_order_requires_exact_id():
    # The real API returns order 1001 for "100" (substring match); we must not.
    assert run({"tool_name": "get_order", "params": {"order_id": "100"}}) == {"orders": []}
    assert [o["orderId"] for o in run({"tool_name": "get_order", "params": {"order_id": "1003"}})["orders"]] == ["1003"]


def test_bad_tool_args_go_back_to_llm():
    replies = [
        call("invoke_tool", {"tool_name": "list_orders", "where": [{"field": "price", "op": "gt", "value": 1}]}, "1"),
        call("invoke_tool", {"tool_name": "list_orders", "where": OHIO_OVER_500}, "2"),
        AIMessage("done"),
    ]
    out = run_agent("q", FakeLLM(replies), get=fake_api())
    assert len(out["orders"]) == 3


def test_agent_never_passes_on_invented_text():
    # The LLM "answers" in prose without fetching: nothing it wrote reaches the output.
    out = run_agent("q", FakeLLM([AIMessage("here are your orders: 1001, 9999")]), get=fake_api())
    assert out == {"error": DEFAULT_UNSUPPORTED}


def test_unsupported_request_gives_clear_error():
    out = run_agent("delete order 1001", FakeLLM([AIMessage("UNSUPPORTED: there is no delete tool")]),
                    get=fake_api())
    assert out == {"error": "there is no delete tool"}


def test_hidden_limit_param_rejected():
    replies = [call("invoke_tool", {"tool_name": "list_orders", "params": {"limit": 2}}, "1"),
               AIMessage("done")]
    # The raw API param `limit` is hidden; the error goes to the LLM, which gives up.
    assert run_agent("q", FakeLLM(replies), get=fake_api()) == {"error": DEFAULT_UNSUPPORTED}


def test_filters_normalize_and_validate():
    fields = load_servers()["customer-orders"]["record_fields"]
    assert records.validate_where([{"field": "state", "op": "eq", "value": "ohio"}], fields)[0]["value"] == "oh"  # compared case-insensitively
    for bad in ({"field": "price", "op": "eq", "value": 1},
                {"field": "total", "op": "between", "value": 1},
                {"field": "buyer", "op": "gt", "value": 1},
                {"field": "state", "op": "eq", "value": "Narnia"}):
        with pytest.raises(ValueError):
            records.validate_where([bad], fields)


def test_in_operator():
    out = run({"tool_name": "list_orders", "where": [{"field": "state", "op": "in", "value": ["TX", "WA"]}]})
    assert [o["orderId"] for o in out["orders"]] == ["1002", "1004"]


def test_top_n_by_total():
    out = run({"tool_name": "list_orders", "sort_by": "total", "order": "desc", "limit": 2})
    assert [o["orderId"] for o in out["orders"]] == ["1003", "1001"]


def test_cheapest_ohio_order():
    out = run({"tool_name": "list_orders", "where": [{"field": "state", "op": "eq", "value": "OH"}],
               "sort_by": "total", "order": "asc", "limit": 1})
    assert [o["orderId"] for o in out["orders"]] == ["1005"]


def test_average_total():
    out = run({"tool_name": "list_orders", "aggregate": {"op": "avg", "field": "total"}})
    assert out == {"aggregate": {"op": "avg", "field": "total", "value": 560.03, "count": 5}}


def test_count_ohio_orders():
    out = run({"tool_name": "list_orders", "where": [{"field": "state", "op": "eq", "value": "OH"}],
               "aggregate": {"op": "count"}})
    assert out == {"aggregate": {"op": "count", "value": 3}}


def test_shape_validation():
    fields = load_servers()["customer-orders"]["record_fields"]
    for bad in ({"sort_by": "price"}, {"order": "sideways"}, {"limit": 0}, {"limit": 5000},
                {"aggregate": {"op": "avg", "field": "buyer"}},
                {"aggregate": {"op": "median", "field": "total"}},
                {"aggregate": {"op": "sum"}}):
        args = {"sort_by": None, "order": None, "limit": None, "aggregate": None, **bad}
        with pytest.raises(ValueError):
            records.validate_shape(args["sort_by"], args["order"], args["limit"], args["aggregate"], fields)


def test_parsed_count_is_before_exact_match():
    from agents.processors import text_records
    from agents.tools import RunContext

    ctx = RunContext(FakeLLM(), get=fake_api())
    server = ctx.servers["customer-orders"]
    summary = text_records(ctx, server, server["tools"]["get_order"], {"params": {"order_id": "100"}})
    assert summary == {"fetched": 1, "parsed": 1, "rejected": 0, "matched": 0}
