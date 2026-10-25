"""Loads the single OpenAPI spec into servers and tools (Forge-style: the spec is
the single source of truth; nothing about the API is hard-coded). Each tag in the
spec is a server the agent can discover; each endpoint under it is a tool."""

import os
from pathlib import Path

import yaml

from agents.processors import PROCESSORS

SPEC_PATH = Path(__file__).resolve().parent.parent / "spec" / "api.yaml"


def load_servers(spec_path: Path = SPEC_PATH) -> dict[str, dict]:
    spec = yaml.safe_load(Path(spec_path).read_text())
    schemas = spec.get("components", {}).get("schemas", {})
    # Env override lets the same spec point at a deployed copy of the API.
    base_url = os.getenv("ORDERS_API_URL", spec["servers"][0]["url"]).rstrip("/")

    servers = {}
    for tag in spec.get("tags", []):
        record_schema = schemas[tag["x-record-schema"]]
        servers[tag["name"]] = {
            "name": tag["name"],
            "description": tag.get("description", ""),
            "base_url": base_url,
            "tools": {},
            "record_schema": record_schema,
            "record_fields": record_schema["properties"],
            "key": record_schema.get("x-key"),
            "collection": tag.get("x-collection-name", "records"),
            "default_fields": tag.get("x-default-fields", list(record_schema["properties"])),
        }

    for route, methods in spec["paths"].items():
        for method, op in methods.items():
            params = [p for p in op.get("parameters", []) if p.get("x-agent-exposed", True)]
            processor = op.get("x-processor", "text-records")
            if processor not in PROCESSORS:
                raise ValueError(f"{op['operationId']}: unknown x-processor '{processor}'")
            for tag_name in op.get("tags", []):
                servers[tag_name]["tools"][op["operationId"]] = {
                    "name": op["operationId"],
                    "method": method.upper(),
                    "path": route,
                    "summary": op.get("summary", ""),
                    "params": params,
                    "records_path": op.get("x-records-path"),
                    "processor": processor,
                }
    return servers


def find_tool(servers: dict, tool_name: str) -> tuple[dict, dict]:
    """Returns (server, tool) for a tool name, or raises KeyError."""
    for server in servers.values():
        if tool_name in server["tools"]:
            return server, server["tools"][tool_name]
    raise KeyError(f"unknown tool '{tool_name}'. Use discover_tools to list tools.")
