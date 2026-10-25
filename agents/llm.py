import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class ConfigError(Exception):
    pass


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("<"):
        raise ConfigError(f"{name} is not set. Copy .env.template to .env and fill it in.")
    return value


def get_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=_require("OPENROUTER_MODEL"),
        api_key=_require("OPENROUTER_API_KEY"),
        base_url=_require("OPENROUTER_BASE_URL"),
        temperature=0,
        # Without a cap, every call reserves the model's full output budget
        # (~116k tokens) against the account's credits.
        max_tokens=8192,
        seed=0,
        timeout=60,
        max_retries=2,
    )
