"""Vercel entrypoint: the dummy orders API plus POST /api/ask, which runs the agent.

The agent reaches the orders API over HTTP like it does locally; set ORDERS_API_URL
to this deployment's URL so it calls its own /api/orders routes."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import jsonify, request
from langgraph.errors import GraphRecursionError

from agents.graph import run_agent
from agents.llm import ConfigError, get_llm
from agents.processors import ApiError
from dummy_customer_api import app

app.json.sort_keys = False  # keep fields in order: orderId, buyer, state, total

# Vercel's filesystem is read-only, so log to stderr only.
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("api")


@app.route("/api/ask", methods=["POST"])
def ask():
    query = str((request.get_json(silent=True) or {}).get("query", "")).strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if len(query) > 500:
        return jsonify({"error": "query too long (max 500 characters)"}), 400
    log.info("query: %s", query)
    try:
        return jsonify(run_agent(query, get_llm()))
    except ConfigError as e:
        log.error("%s", e)
        return jsonify({"error": "server is not configured"}), 500
    except ApiError as e:
        log.error("%s", e)
        return jsonify({"error": str(e)}), 502
    except GraphRecursionError:
        return jsonify({"error": "agent did not finish within the step limit"}), 504
    except Exception:
        log.exception("unexpected error")
        return jsonify({"error": "unexpected error"}), 500
