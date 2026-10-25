"""The agent's four tools, modeled on Uber's Omni MCP:
discover_server -> discover_tools -> get_tool_schema -> invoke_tool.
Tools 1-3 are lookups in the spec registry. invoke_tool looks up the tool and
hands it to the processor its spec names; only a summary goes back to the LLM,
the records stay in RunContext."""

import json
import logging
from typing import Literal, NotRequired, TypedDict

from langchain_core.tools import tool

from agents import processors, records, registry

log = logging.getLogger(__name__)


class Condition(TypedDict):
    field: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in"]
    value: str | float | list[str]


class Aggregate(TypedDict):
    op: Literal["count", "sum", "avg", "min", "max"]
    field: NotRequired[str]


class RunContext:
    """Per-run state. `result` is the output finalize returns."""

    def __init__(self, llm, servers: dict | None = None, get=processors.http_get):
        self.llm = llm
        self.servers = servers if servers is not None else registry.load_servers()
        self.get = get
        self.result: dict | None = None  # the final output, built by the processor
        self.rejected: list[dict] = []


def build_tools(ctx: RunContext) -> list:
    @tool
    def discover_server(query: str = "") -> str:
        """Search the available API servers. Start here."""
        words = query.lower().split()
        hits = [s for s in ctx.servers.values()
                if not words or any(w in f"{s['name']} {s['description']}".lower() for w in words)]
        hits = hits or list(ctx.servers.values())  # never dead-end the agent
        return json.dumps([{"server": s["name"], "description": s["description"]} for s in hits])

    @tool
    def discover_tools(server: str) -> str:
        """List the tools (API endpoints) on a server."""
        if server not in ctx.servers:
            raise ValueError(f"unknown server '{server}'. Use discover_server.")
        return json.dumps([{"tool": t["name"], "summary": t["summary"]}
                           for t in ctx.servers[server]["tools"].values()])

    @tool
    def get_tool_schema(tool_name: str) -> str:
        """Get a tool's parameters and the fields of the records it returns."""
        server, t = registry.find_tool(ctx.servers, tool_name)
        return json.dumps({
            "tool": t["name"],
            "params": {p["name"]: {"type": p["schema"]["type"], "required": p.get("required", False)}
                       for p in t["params"]},
            "record_fields": {k: {"type": v["type"], "required": k in server["record_schema"].get("required", []),
                                  **({"format": v["format"]} if "format" in v else {}),
                                  "description": v.get("description", "")}
                              for k, v in server["record_fields"].items()},
            "default_fields": server["default_fields"],
            "where_ops": list(records.OPS),
            "sort": {"order": ["asc", "desc"], "max_limit": records.MAX_LIMIT},
            "aggregate_ops": list(records.AGG_OPS),
        })

    @tool
    def invoke_tool(tool_name: str, params: dict | None = None,
                    fields: list[str] | None = None, where: list[Condition] | None = None,
                    sort_by: str | None = None, order: Literal["asc", "desc"] | None = None,
                    limit: int | None = None, aggregate: Aggregate | None = None) -> str:
        """Call a tool. The records are parsed, filtered, sorted and counted in code;
        you get back counts only.
        where: conditions, all ANDed, e.g. [{"field": "state", "op": "eq", "value": "OH"},
        {"field": "total", "op": "gt", "value": 500}]. Use "in" with a list for OR on one field.
        fields: record fields to return (default: the tool's default_fields).
        sort_by + order + limit: e.g. top 2 by total = sort_by "total", order "desc", limit 2.
        aggregate: one summary number instead of records, e.g. {"op": "avg", "field": "total"}
        or {"op": "count"}."""
        server, t = registry.find_tool(ctx.servers, tool_name)
        processor = processors.PROCESSORS[t["processor"]]
        args = {"params": params, "fields": fields, "where": where, "sort_by": sort_by,
                "order": order, "limit": limit, "aggregate": aggregate}
        summary = processor(ctx, server, t, args)
        log.info("invoke result: %s", summary)
        return json.dumps(summary)

    return [discover_server, discover_tools, get_tool_schema, invoke_tool]
