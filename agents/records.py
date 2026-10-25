"""Turns raw API text into checked, filtered records.

Parsing: the LLM parses raw text into records shaped by the spec's
x-record-schema; plain-Python checks reject anything not backed by the source text.
Filtering: the LLM's `where`/`fields` are validated and applied in plain code;
the LLM only describes the filter, it never decides which records match.
Nothing here knows about orders: fields, types and formats all come from the spec."""

import json
import logging
import re
from collections import Counter
from datetime import date

import openai
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "contains", "in")

# Value formats a spec field can declare (`format: us-state`): how to normalize
# a value, find it in source text, and describe it to the LLM.

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
_NAME_TO_CODE = {name.lower(): code for code, name in US_STATES.items()}


def in_text(value: str, source: str) -> bool:
    """Case-insensitive match on word boundaries, so "100" is not found in "1001"."""
    return re.search(rf"(?<!\w){re.escape(value.strip())}(?!\w)", source, re.IGNORECASE) is not None


def to_state_code(value) -> str:
    """'Ohio', 'oh', 'OH' -> 'OH'. Raises ValueError for anything else."""
    text = str(value).strip()
    if text.upper() in US_STATES:
        return text.upper()
    if text.lower() in _NAME_TO_CODE:
        return _NAME_TO_CODE[text.lower()]
    raise ValueError(f"'{value}' is not a US state")


def _state_in_text(code: str, source: str) -> bool:
    # The code must appear as-is ("OH"), or the full name in any case ("Ohio").
    return re.search(rf"\b{code}\b", source) is not None or in_text(US_STATES[code], source)


def to_iso_date(value) -> str:
    """'2026-03-05' -> '2026-03-05'. Raises ValueError for anything else."""
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"'{value}' is not a date (use YYYY-MM-DD)") from None


FORMATS = {
    "us-state": {
        "hint": "2-letter US state code, e.g. OH",
        "normalize": to_state_code,
        "in_text": _state_in_text,
    },
    "date": {
        "hint": "date as YYYY-MM-DD",
        "normalize": to_iso_date,
        "in_text": in_text,
        "ordered": True,  # allows gt/lt and sorting, e.g. "orders in March"
    },
}


# --- Parsing ---------------------------------------------------------------

# Batches keep each LLM call small no matter how many records the API returns.
MAX_BATCH_RECORDS = 20
MAX_BATCH_CHARS = 6000
MAX_RECORD_CHARS = 2000

SYSTEM_PROMPT = """You extract structured records from raw text.
Rules:
- Return exactly one object per numbered record, with its [n] number as "index".
- Copy every value from the record text. Never invent, guess or fill in values.
- Numbers are plain numbers (742.10, not "$742.10").
- If a value is not in the record, use an empty string (or [] for lists)."""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EMPTY = {"string": "", "array": [], "number": None}


def llm_schema(record_schema: dict) -> dict:
    """The structured-output schema the LLM must fill, built from the spec."""
    props = {"index": {"type": "integer", "description": "The [n] number of the input record"}}
    for name, spec in record_schema["properties"].items():
        prop = {"type": spec["type"]}
        if spec["type"] == "array":
            prop["items"] = {"type": "string"}
        desc = spec.get("description", "")
        if spec.get("format") in FORMATS:
            desc = f"{desc} ({FORMATS[spec['format']]['hint']})".strip()
        if desc:
            prop["description"] = desc
        props[name] = prop
    return {
        "title": "parsed_records",
        "description": "One parsed object per numbered input record.",
        "type": "object",
        "properties": {"records": {"type": "array", "items": {
            "type": "object", "properties": props, "required": list(props)}}},
        "required": ["records"],
    }


def check_record(rec: dict, source: str, record_schema: dict) -> list[str]:
    """Returns a list of problems; empty means every value is backed by the source text."""
    errors = []
    required = set(record_schema.get("required", []))
    for name, spec in record_schema["properties"].items():
        value, kind = rec.get(name), spec["type"]
        if value is None or value == "" or value == []:
            if name in required:
                errors.append(f"{name} missing")
            continue

        if kind == "string":
            if not isinstance(value, str):
                errors.append(f"{name} must be a string, got {value!r}")
                continue
            fmt = FORMATS.get(spec.get("format"))
            if spec.get("pattern") and not re.fullmatch(spec["pattern"], value):
                errors.append(f"{name} {value!r} must match {spec['pattern']}")
            elif fmt:
                try:
                    code = fmt["normalize"](value)
                except ValueError as e:
                    errors.append(f"{name}: {e}")
                    continue
                if code != value:
                    errors.append(f"{name} must be {fmt['hint']}, got {value!r}")
                elif not fmt["in_text"](value, source):
                    errors.append(f"{name} {value!r} not in source")
            elif not in_text(value, source):
                errors.append(f"{name} {value!r} not in source")

        elif kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name} must be a number, got {value!r}")
            elif "minimum" in spec and value < spec["minimum"]:
                errors.append(f"{name} {value} is below {spec['minimum']}")
            elif round(float(value), 2) not in {round(float(n.replace(",", "")), 2)
                                                for n in _NUMBER.findall(source)}:
                errors.append(f"{name} {value} not in source")

        elif kind == "array":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{name} must be a list of strings")
            else:
                errors += [f"{name} item {v!r} not in source" for v in value if not in_text(v, source)]
    return errors


def _normalize(rec: dict, record_schema: dict) -> dict:
    out = {}
    for name, spec in record_schema["properties"].items():
        value = rec.get(name)
        if value is None or value == "" or value == []:
            value = _EMPTY.get(spec["type"])
        elif spec["type"] == "number":
            value = float(value)
        out[name] = value
    return out


def _batches(indexed: list[tuple[int, str]]):
    batch, size = [], 0
    for i, text in indexed:
        if batch and (len(batch) >= MAX_BATCH_RECORDS or size + len(text) > MAX_BATCH_CHARS):
            yield batch
            batch, size = [], 0
        batch.append((i, text))
        size += len(text)
    if batch:
        yield batch


def _parse_batch(llm, batch, feedback, record_schema) -> dict[int, list[dict]]:
    """One LLM call. Returns {index: [records the LLM returned for it]}."""
    lines = []
    for i, text in batch:
        lines.append(f"[{i}] {text}")
        if i in feedback:
            lines.append(f"    (previous attempt was rejected: {'; '.join(feedback[i])})")
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage("\n".join(lines))]

    result = llm.with_structured_output(llm_schema(record_schema), method="function_calling").invoke(messages)
    if isinstance(result, str):
        result = json.loads(result)

    by_index: dict[int, list[dict]] = {}
    for rec in (result or {}).get("records", []):
        if isinstance(rec, dict) and isinstance(rec.get("index"), int):
            by_index.setdefault(rec["index"], []).append(rec)
    return by_index


def _attempt(llm, pending, feedback, record_schema, progress=None):
    """Parses and checks pending records. Returns (accepted, failures)."""
    accepted, failures = {}, {}
    batches = list(_batches(pending))
    for n, batch in enumerate(batches, 1):
        if progress:
            progress(f"Model parsing batch {n} of {len(batches)} ({len(batch)} records)")
        try:
            by_index = _parse_batch(llm, batch, feedback, record_schema)
        except openai.APIError:
            raise  # the model provider is down or out of credits: stop, don't reject good records
        except Exception as e:  # bad JSON or a malformed reply: these records fail checks
            log.warning("parse batch of %d failed: %s", len(batch), e)
            by_index = {}
        for i, text in batch:
            recs = by_index.get(i, [])
            if len(recs) != 1:
                failures[i] = [f"expected exactly 1 record, got {len(recs)}"]
                continue
            errors = check_record(recs[0], text, record_schema)
            if errors:
                failures[i] = errors
            else:
                accepted[i] = _normalize(recs[0], record_schema)  # drops anything extra
    return accepted, failures


def parse_records(llm, texts: list[str], record_schema: dict, progress=None) -> tuple[list[dict], list[dict]]:
    """Parses raw texts. Returns (records, rejected). Each failing record gets
    one retry with its errors as feedback; after that it is rejected, never guessed."""
    pending, rejected = [], []
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            rejected.append({"source": text, "errors": ["empty record"]})
        elif len(text) > MAX_RECORD_CHARS:
            rejected.append({"source": text[:100] + "...", "errors": ["record too long"]})
        else:
            pending.append((i, text))

    accepted, failures = _attempt(llm, pending, {}, record_schema, progress)
    if failures:
        log.info("retrying %d record(s) that failed checks", len(failures))
        if progress:
            progress(f"{len(failures)} record(s) failed the checks; asking the model again")
        for i, errors in failures.items():
            log.info("  record %d: %s", i, "; ".join(errors))
        retry = [(i, t) for i, t in pending if i in failures]
        more, failures = _attempt(llm, retry, failures, record_schema, progress)
        accepted.update(more)

    for i, errors in failures.items():
        log.warning("rejected record %d: %s | %s", i, "; ".join(errors), texts[i])
        rejected.append({"source": texts[i], "errors": errors})

    # Two records with the same key are ambiguous, and the API's order is random,
    # so "keep the first" would pick a different one each run. Reject them all.
    key = record_schema.get("x-key")
    counts = Counter(accepted[i].get(key) for i in accepted) if key else Counter()
    unique = []
    for i in sorted(accepted):
        rec = accepted[i]
        if key and counts[rec[key]] > 1:
            log.warning("rejected record %d: duplicate %s %s | %s", i, key, rec[key], texts[i])
            rejected.append({"source": texts[i], "errors": [f"duplicate {key} {rec[key]}"]})
        else:
            unique.append(rec)
    return unique, rejected


# --- Filtering -------------------------------------------------------------

def _coerce(value, spec: dict, field: str):
    fmt = FORMATS.get(spec.get("format"))
    if fmt:
        return fmt["normalize"](value).lower()
    if spec["type"] == "number":
        if isinstance(value, bool):
            raise ValueError(f"{field} needs a number, got {value!r}")
        try:
            return float(str(value).replace("$", "").replace(",", ""))
        except ValueError:
            raise ValueError(f"{field} needs a number, got {value!r}") from None
    return str(value).strip().lower()


def validate_where(where: list[dict] | None, fields: dict) -> list[dict]:
    """Checks and normalizes conditions. Raises ValueError with a message the
    LLM can act on."""
    conditions = []
    for cond in where or []:
        field, op, value = cond.get("field"), cond.get("op"), cond.get("value")
        if field not in fields:
            raise ValueError(f"unknown field '{field}'. Allowed: {', '.join(fields)}")
        if op not in OPS:
            raise ValueError(f"unknown op '{op}'. Allowed: {', '.join(OPS)}")
        ordered = FORMATS.get(fields[field].get("format"), {}).get("ordered")
        if op in ("gt", "gte", "lt", "lte") and fields[field]["type"] != "number" and not ordered:
            raise ValueError(f"op '{op}' only works on number or date fields, not '{field}'")
        if op == "in":
            if not isinstance(value, list) or not value:
                raise ValueError("op 'in' needs a non-empty list value")
            value = [_coerce(v, fields[field], field) for v in value]
        else:
            value = _coerce(value, fields[field], field)
        conditions.append({"field": field, "op": op, "value": value})
    return conditions


def validate_fields(requested: list[str] | None, fields: dict, default: list[str], key: str) -> list[str]:
    if not requested:
        return list(default)
    unknown = [f for f in requested if f not in fields]
    if unknown:
        raise ValueError(f"unknown fields {unknown}. Allowed: {', '.join(fields)}")
    return list(dict.fromkeys([key, *requested]))  # the key is always kept for sorting


def _matches(actual, cond: dict) -> bool:
    if isinstance(actual, list):  # e.g. items: match if any element matches
        return any(_matches(a, cond) for a in actual)
    if actual is None:
        return False
    if isinstance(actual, str):
        actual = actual.lower()
    op, value = cond["op"], cond["value"]
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "in":
        return actual in value
    if op == "contains":
        return str(value) in str(actual)
    return {"gt": actual > value, "gte": actual >= value,
            "lt": actual < value, "lte": actual <= value}[op]


def filter_records(records: list[dict], conditions: list[dict]) -> list[dict]:
    """All conditions are ANDed."""
    return [r for r in records if all(_matches(r.get(c["field"]), c) for c in conditions)]


# --- Sorting, limit and aggregates ---------------------------------------------

AGG_OPS = ("count", "sum", "avg", "min", "max")
MAX_LIMIT = 1000


def validate_shape(sort_by, order, limit, aggregate, fields: dict) -> dict:
    """Checks the LLM's sort/limit/aggregate request. Raises ValueError with a
    message the LLM can act on."""
    if sort_by is not None and sort_by not in fields:
        raise ValueError(f"unknown sort_by field '{sort_by}'. Allowed: {', '.join(fields)}")
    if order not in (None, "asc", "desc"):
        raise ValueError("order must be 'asc' or 'desc'")
    if isinstance(limit, float) and limit.is_integer():
        limit = int(limit)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or not 1 <= limit <= MAX_LIMIT):
        raise ValueError(f"limit must be a whole number from 1 to {MAX_LIMIT}")

    agg = None
    if aggregate:
        op, field = aggregate.get("op"), aggregate.get("field")
        if op not in AGG_OPS:
            raise ValueError(f"unknown aggregate op '{op}'. Allowed: {', '.join(AGG_OPS)}")
        if op != "count":
            if field not in fields:
                raise ValueError(f"aggregate '{op}' needs a field. Allowed: {', '.join(fields)}")
            if fields[field]["type"] != "number":
                raise ValueError(f"aggregate '{op}' only works on number fields, not '{field}'")
        agg = {"op": op, "field": None if op == "count" else field}
    return {"sort_by": sort_by, "desc": order == "desc", "limit": limit, "aggregate": agg}


def sort_key(value):
    """Numeric IDs sort as numbers ("2" before "10"); anything else as text."""
    text = str(value)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _value_key(value):
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return ", ".join(value).lower()
    return value


def select(records: list[dict], fields: list[str], key: str | None,
           sort_by: str | None = None, desc: bool = False, limit: int | None = None) -> list[dict]:
    """Sorts (by the key, then by sort_by if given), applies the limit and keeps
    only the requested fields. Ties keep key order, so output is deterministic."""
    rows = sorted(records, key=lambda r: sort_key(r.get(key))) if key else list(records)
    if sort_by:
        present = [r for r in rows if r.get(sort_by) not in (None, "", [])]
        missing = [r for r in rows if r.get(sort_by) in (None, "", [])]
        rows = sorted(present, key=lambda r: _value_key(r[sort_by]), reverse=desc) + missing
    if limit:
        rows = rows[:limit]
    return [{f: r.get(f) for f in fields} for r in rows]


def aggregate(records: list[dict], op: str, field: str | None) -> dict:
    if op == "count":
        return {"op": "count", "value": len(records)}
    values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    if not values:
        value = None
    elif op == "sum":
        value = round(sum(values), 2)
    elif op == "avg":
        value = round(sum(values) / len(values), 2)
    else:
        value = round(min(values) if op == "min" else max(values), 2)
    return {"op": op, "field": field, "value": value, "count": len(values)}
