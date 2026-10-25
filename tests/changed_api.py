"""A copy of the dummy API whose format has changed: new record layouts, and the
records moved from "raw_orders" to "data". Used to show the agent copes.

    python tests/changed_api.py          # serves on port 5002
    ORDERS_API_URL=http://localhost:5002 python main.py "Ohio orders over 500"
"""

from flask import Flask, jsonify

app = Flask(__name__)

ORDERS = [
    "#1006 | buyer: Ana Reyes | Toledo, Ohio | $620.00 | items: tv",
    "Ben Ortiz placed order 1007 for $1,250.00 (Dayton, Ohio): sofa, rug",
    "order=1008;customer=Cara Nguyen;st=TX;amount=310.25",
    "1009 / Dev Patel / Akron OH / 75.00 USD / headphones",
]


@app.route("/api/orders")
def get_orders():
    return jsonify({"status": "ok", "data": ORDERS})


@app.route("/api/order/<order_id>")
def get_order(order_id):
    for text in ORDERS:
        if order_id in text:
            return jsonify({"status": "ok", "data": text})
    return jsonify({"status": "not_found"}), 404


if __name__ == "__main__":
    app.run(port=5002)
