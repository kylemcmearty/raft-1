"""Processors: what invoke_tool does for a tool. Each tool's spec entry names one
(`x-processor`); invoke_tool looks it up and hands over. A new kind of API (clean
JSON, a POST action, ...) means one new processor here, then only spec edits.

Every processor takes (ctx, server, tool, params, fields, where), stores its
result on ctx, and returns a small summary: the only thing the LLM sees."""

import json
import logging
import re

import requests
from langgraph.config import get_stream_writer

from agents import records

log = logging.getLogger(__name__)

def progress(message: str) -> None:
    """Sends a progress step to anyone streaming the agent (the web UI). A no-op
    when the processor runs outside a graph, e.g. in tests."""
    try:
        get_stream_writer()({"type": "progress", "message": message})
    except Exception:
        pass


class ApiError(Exception):
    """An API is unreachable or broken. Fatal: not retried by the LLM."""

def http_get(url: str, params: dict) -> tuple[int, dict]:
    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as e:
        raise ApiError(f"API unreachable at {url}: {e}") from e
    if resp.status_code == 404:
        return 404, {}
    if not resp.ok:
        raise ApiError(f"API returned HTTP {resp.status_code} for {url}")
    try:
        return resp.status_code, resp.json()
    except ValueError as e:
        raise ApiError(f"API returned non-JSON from {url}") from e

def build_request(tool: dict, params: dict | None) -> tuple[str, dict, dict]:
    """Checks the LLM's params against the spec and builds the request.
    Returns (path, query, must_equal). Only spec-declared params get through."""
    params = dict(params or {})
    allowed = {p["name"]: p for p in tool["params"]}
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}. Allowed: {sorted(allowed) or 'none'}")

    path, query, must_equal = tool["path"], {}, {}
    for name, p in allowed.items():
        if name not in params:
            if p.get("required"):
                raise ValueError(f"missing required param '{name}'")
            continue
        value = str(params[name]).strip()
        pattern = p["schema"].get("pattern")
        if pattern and not re.fullmatch(pattern, value):
            raise ValueError(f"param '{name}'={value!r} must match {pattern}")
        if p.get("x-must-equal-field"):
            must_equal[p["x-must-equal-field"]] = value
        if p["in"] == "path":
            path = path.replace("{" + name + "}", value)
        else:
            query[name] = value
    return path, query, must_equal

def extract_texts(payload: dict, records_path: str | None) -> list[str]:
    """Finds the raw text records. Uses the spec's path, and falls back to
    searching the payload if the API's response shape has changed."""
    value = payload.get(records_path) if records_path else None
    if value is None:
        log.warning("response has no '%s' key (API schema drift?); searching payload", records_path)
        candidates = [v for k, v in payload.items() if k != "status" and isinstance(v, (str, list))]
        value = candidates[0] if candidates else []
    if isinstance(value, str):
        value = [value]
    return [v if isinstance(v, str) else json.dumps(v) for v in value]

def text_records(ctx, server, tool, args: dict) -> dict:
    """For APIs that return messy text records: fetch -> LLM parses -> code checks
    each value against the text -> code filters, sorts and limits (or aggregates)."""
    fields = server["record_fields"]
    conditions = records.validate_where(args.get("where"), fields)
    out_fields = records.validate_fields(args.get("fields"), fields, server["default_fields"], server["key"])
    shape = records.validate_shape(args.get("sort_by"), args.get("order"), args.get("limit"),
                                   args.get("aggregate"), fields)
    path, query, must_equal = build_request(tool, args.get("params"))

    log.info("invoke %s %s params=%s where=%s fields=%s shape=%s",
             tool["method"], path, query, conditions, out_fields, shape)
    progress(f"Calling the API: {tool['method']} {path}")
    status, payload = ctx.get(server["base_url"] + path, query)
    texts = [] if status == 404 else extract_texts(payload, tool["records_path"])

    progress(f"Got {len(texts)} raw record(s)")
    parsed, rejected = records.parse_records(ctx.llm, texts, server["record_schema"], progress)
    progress(f"Checked every value against the source text: {len(parsed)} kept, {len(rejected)} rejected")
    # The API's lookups are fuzzy (substring match); enforce exact matches here.
    exact = [r for r in parsed if all(str(r[f]) == v for f, v in must_equal.items())]
    matched = records.filter_records(exact, conditions)

    agg = shape["aggregate"]
    if agg:
        ctx.result = {"aggregate": records.aggregate(matched, agg["op"], agg["field"])}
    else:
        ctx.result = {server["collection"]: records.select(
            matched, out_fields, server["key"], shape["sort_by"], shape["desc"], shape["limit"])}
    ctx.rejected = rejected
    return {"fetched": len(texts), "parsed": len(parsed),
            "rejected": len(rejected), "matched": len(matched)}


PROCESSORS = {
    "text-records": text_records,
}
