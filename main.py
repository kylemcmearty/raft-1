import argparse
import atexit
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import openai
import requests
from langgraph.errors import GraphRecursionError

from agents.graph import run_agent
from agents.llm import ConfigError, get_llm
from agents.registry import load_servers
from agents.processors import ApiError
from logging_config import LOG_DIR, setup_logging



log = logging.getLogger("main")

ROOT = Path(__file__).resolve().parent


def api_is_up(base_url: str) -> bool:
    try:
        return requests.get(base_url + "/api/orders", timeout=2).ok
    except requests.RequestException:
        return False


def ensure_api(base_url: str, extended: bool = False) -> None:
    if api_is_up(base_url):
        count = len(requests.get(base_url + "/api/orders", timeout=5).json().get("raw_orders", []))
        if "localhost" in base_url and extended != (count > 5):
            want = "with" if extended else "without"
            raise ApiError(f"the dummy API is already running in the other mode; stop it, "
                           f"or rerun {want} --extended")
        return
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise ApiError(f"orders API at {base_url} is not reachable")
    log.info("starting dummy API at %s%s", base_url, " (extended data)" if extended else "")
    
    # Run without Flask's debug reloader so the process can be stopped cleanly.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import dummy_customer_api as d; "
         + ("d.use_extended(); " if extended else "") + "d.app.run(port=5001)"],
        cwd=ROOT, stdout=open(LOG_DIR / "api.log", "a"), stderr=subprocess.STDOUT,
    )
    atexit.register(proc.terminate)
    for _ in range(40):
        if api_is_up(base_url):
            return
        time.sleep(0.25)
    raise ApiError("dummy API did not start; see logs/api.log")


def answer(query: str, llm) -> None:
    """Runs one question and prints the JSON. Errors are logged, not raised."""
    if len(query) > 500:
        log.error("query too long (max 500 characters)")
        return
    log.info("query: %s", query)
    try:
        output = run_agent(query, llm)
    except ApiError as e:
        log.error("%s", e)
        return
    except openai.APIError as e:
        status = getattr(e, "status_code", None)
        log.error("model provider error%s: %s", f" (HTTP {status})" if status else "", str(e)[:300])
        return
    except GraphRecursionError:
        log.error("agent did not finish within the step limit")
        return
    except Exception:
        log.exception("unexpected error")
        return
    print(json.dumps(output, indent=2), flush=True)


def interactive(llm) -> None:
    print("Ask about orders ('quit' to exit).", file=sys.stderr)
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            return
        answer(query, llm)  # a failed question doesn't end the session


def main() -> int:
    parser = argparse.ArgumentParser(description="Natural-language order search agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--extended", action="store_true",
                        help="serve 8 generated orders (messy layouts, bad records) instead of Raft's 5")
    args = parser.parse_args()
    log_file = setup_logging(args.verbose)
    log.info("logging to %s", log_file)

    try:
        for server in load_servers().values():
            ensure_api(server["base_url"], args.extended)
        llm = get_llm()
    except (ConfigError, ApiError) as e:
        log.error("%s", e)
        return 1

    interactive(llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())


