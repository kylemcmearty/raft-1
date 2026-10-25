"""A different API (products) works by describing it in the spec only: no code changes."""

import re

from langchain_core.messages import AIMessage

from agents.graph import run_agent
from agents.registry import load_servers
from tests.fakes import FakeLLM, call

PRODUCTS_SPEC = """
openapi: 3.0.0
info:
  title: Products API
  version: "1.0"
servers:
  - url: http://products.test
tags:
  - name: products
    description: Product catalog with prices and stock.
    x-record-schema: Product
    x-collection-name: products
    x-default-fields: [sku, name, price]
paths:
  /products:
    get:
      tags: [products]
      operationId: list_products
      summary: List all products.
      x-records-path: items
components:
  schemas:
    Product:
      type: object
      x-key: sku
      required: [sku, name, price]
      properties:
        sku:   { type: string }
        name:  { type: string }
        price: { type: number, minimum: 0 }
        stock: { type: number }
"""

PRODUCTS = [
    "SKU B-200 | Desk Lamp | $45.00 | 12 in stock",
    "SKU A-100 | Standing Desk | $399.99 | 3 in stock",
    "SKU C-300 | Monitor Arm | $89.50 | 0 in stock",
]


def parse_products(messages):
    out = []
    for line in messages[-1].content.splitlines():
        m = re.match(r"^\[(\d+)\] SKU (\S+) \| (.+?) \| \$([\d.]+) \| (\d+) in stock$", line)
        if m:
            i, sku, name, price, stock = m.groups()
            out.append({"index": int(i), "sku": sku, "name": name,
                        "price": float(price), "stock": float(stock)})
    return {"records": out}


def test_new_api_from_spec_alone(tmp_path):
    (tmp_path / "api.yaml").write_text(PRODUCTS_SPEC)
    servers = load_servers(tmp_path / "api.yaml")

    replies = [
        call("discover_server", {"query": "products"}, "1"),
        call("discover_tools", {"server": "products"}, "2"),
        call("invoke_tool", {"tool_name": "list_products",
                             "where": [{"field": "price", "op": "lt", "value": 100}]}, "3"),
        AIMessage("done"),
    ]
    out = run_agent("cheap products", FakeLLM(replies, parse=parse_products),
                    servers=servers, get=lambda url, params: (200, {"items": PRODUCTS}))

    assert out == {"products": [
        {"sku": "B-200", "name": "Desk Lamp", "price": 45.0},
        {"sku": "C-300", "name": "Monitor Arm", "price": 89.5},
    ]}


def test_new_api_checks_still_apply(tmp_path):
    (tmp_path / "api.yaml").write_text(PRODUCTS_SPEC)
    schema = load_servers(tmp_path / "api.yaml")["products"]["record_schema"]
    from agents.records import check_record

    good = {"sku": "A-100", "name": "Standing Desk", "price": 399.99, "stock": 3}
    assert check_record(good, PRODUCTS[1], schema) == []
    assert check_record({**good, "price": 39.99}, PRODUCTS[1], schema)     # invented price
    assert check_record({**good, "name": "Office Chair"}, PRODUCTS[1], schema)  # invented name
