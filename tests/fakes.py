"""A scripted stand-in for the LLM so tests run without network or credits."""

import re

from langchain_core.messages import AIMessage

ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
    "Order 1004: Buyer=Rachel Kim, Location=Seattle, WA, Total=$89.50, Items: coffee maker",
    "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp",
]

_LINE = re.compile(r"^\[(\d+)\] Order (\d+): Buyer=(.+?), Location=(.+?), ([A-Z]{2}), "
                   r"Total=\$([\d.]+), Items: (.*)$")


def honest_parse(messages) -> dict:
    """Parses the numbered records in the prompt correctly."""
    orders = []
    for line in messages[-1].content.splitlines():
        m = _LINE.match(line)
        if m:
            i, oid, buyer, city, state, total, items = m.groups()
            orders.append({"index": int(i), "orderId": oid, "buyer": buyer, "city": city,
                           "state": state, "total": float(total),
                           "items": [s.strip() for s in items.split(",")]})
    return {"records": orders}


class FakeLLM:
    def __init__(self, agent_replies=(), parse=honest_parse):
        self.agent_replies = list(agent_replies)
        self.parse = parse
        self.parse_calls = 0

    # agent side
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.agent_replies.pop(0)

    # parser side
    def with_structured_output(self, schema, method=None):
        fake = self

        class Runner:
            def invoke(self, messages):
                fake.parse_calls += 1
                return fake.parse(messages)

        return Runner()


def call(name, args, id_):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id_}])


def fake_api(orders=ORDERS):
    """Mimics dummy_customer_api.py, including its random order."""
    import random

    def get(url, params):
        if url.endswith("/api/orders"):
            return 200, {"status": "ok", "raw_orders": random.sample(orders, len(orders))}
        order_id = url.rsplit("/", 1)[-1]
        for text in orders:
            if order_id in text:  # same fuzzy match as the real API
                return 200, {"status": "ok", "raw_order": text}
        return 404, {}

    return get
