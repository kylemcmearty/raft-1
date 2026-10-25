"""Vercel entrypoint: the dummy orders API plus POST /api/ask, which runs the agent.

The agent reaches the orders API over HTTP like it does locally; set ORDERS_API_URL
to this deployment's URL so it calls its own /api/orders routes."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import openai
from flask import Response, jsonify, request, stream_with_context
from langgraph.errors import GraphRecursionError

from agents.graph import run_agent, stream_agent
from agents.llm import ConfigError, get_llm
from agents.processors import ApiError
from agents.registry import load_servers
from dummy_customer_api import app

app.json.sort_keys = False  # keep fields in order: orderId, buyer, state, total

# Vercel's filesystem is read-only, so log to stderr only.
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("api")


def _provider_error(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    if status == 402:
        return "the model provider is out of credits; try again later"
    if status == 429:
        return "the model provider is rate-limiting us; try again in a minute"
    return f"the model provider failed{f' (HTTP {status})' if status else ''}; try again"


def _servers(body: dict) -> dict:
    """The spec's servers; with {"extended": true}, pointed at the /extended data."""
    servers = load_servers()
    if body.get("extended") is True:
        for server in servers.values():
            server["base_url"] += "/extended"
    return servers


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if len(query) > 500:
        return jsonify({"error": "query too long (max 500 characters)"}), 400
    log.info("query: %s", query)
    try:
        return jsonify(run_agent(query, get_llm(), servers=_servers(body)))
    except ConfigError as e:
        log.error("%s", e)
        return jsonify({"error": "server is not configured"}), 500
    except ApiError as e:
        log.error("%s", e)
        return jsonify({"error": str(e)}), 502
    except openai.APIError as e:
        log.error("model provider error: %s", e)
        return jsonify({"error": _provider_error(e)}), 502
    except GraphRecursionError:
        return jsonify({"error": "agent did not finish within the step limit"}), 504
    except Exception:
        log.exception("unexpected error")
        return jsonify({"error": "unexpected error"}), 500


@app.route("/api/ask/stream", methods=["POST"])
def ask_stream():
    """Same as /api/ask, but streams the agent's steps as newline-delimited JSON
    (tool_call, progress, tool_result, then done or error) so a UI can show them live."""
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", "")).strip()
    if not query or len(query) > 500:
        return jsonify({"error": "query is required (max 500 characters)"}), 400
    log.info("query (stream%s): %s", ", extended" if body.get("extended") is True else "", query)

    def events():
        try:
            for event in stream_agent(query, get_llm(), servers=_servers(body)):
                yield json.dumps(event) + "\n"
        except ConfigError as e:
            log.error("%s", e)
            yield json.dumps({"type": "error", "message": "server is not configured"}) + "\n"
        except ApiError as e:
            log.error("%s", e)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        except openai.APIError as e:
            log.error("model provider error: %s", e)
            yield json.dumps({"type": "error", "message": _provider_error(e)}) + "\n"
        except GraphRecursionError:
            yield json.dumps({"type": "error", "message": "agent did not finish within the step limit"}) + "\n"
        except Exception:
            log.exception("unexpected error")
            yield json.dumps({"type": "error", "message": "unexpected error"}) + "\n"

    return Response(stream_with_context(events()), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

