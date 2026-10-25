import json
import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so log files are easy to search and parse."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(verbose: bool = False) -> Path:
    """Readable INFO (or DEBUG with --verbose) to stderr, DEBUG as JSON lines to a
    new logs/agent_<datetime>.jsonl per run. stdout is kept clean for the JSON output."""
    LOG_DIR.mkdir(exist_ok=True)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    log_file = LOG_DIR / f"agent_{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl"
    file = logging.FileHandler(log_file)
    file.setLevel(logging.DEBUG)
    file.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers[:] = [console, file]
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "urllib3", "dotenv"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file
