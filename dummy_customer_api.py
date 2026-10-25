from flask import Flask, request, jsonify
import random
import sys
from datetime import date, timedelta

app = Flask(__name__)

# simulate messy unstructured text responses
ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
    "Order 1004: Buyer=Rachel Kim, Location=Seattle, WA, Total=$89.50, Items: coffee maker",
    "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp"
]

@app.route("/api/orders", methods=["GET"])
def get_orders():
    """
    Returns orders as messy text. In real life, customers
    would have unpredictable formatting. The AI must parse it.
    """
    limit = request.args.get("limit", default=len(ORDERS), type=int)
    sample = random.sample(ORDERS, min(limit, len(ORDERS)))

    return jsonify({
        "status": "ok",
        "raw_orders": sample
    })


@app.route("/api/order/<order_id>", methods=["GET"])
def get_order_by_id(order_id):
    """
    Fetch a single order by scanning the text.
    """
    for text in ORDERS:
        if order_id in text:
            return jsonify({
                "status": "ok",
                "raw_order": text
            })

    return jsonify({"status": "not_found"}), 404


# --- Extended data (python dummy_customer_api.py --extended) --------------------
# 50 generated orders instead of the 5 above: most in the format above, about
# 25% in other layouts, optional fields (date, email, payment) that come and go,
# and a few bad records on purpose. A fixed seed keeps the data the same every run.

FIRST = ["John", "Sarah", "Mike", "Rachel", "Chris", "Ana", "Ben", "Cara", "Dev", "Emma",
         "Liam", "Olivia", "Noah", "Ava", "Ethan", "Mia", "Lucas", "Zoe", "Omar", "Priya"]
LAST = ["Davis", "Liu", "Turner", "Kim", "Myers", "Reyes", "Ortiz", "Nguyen", "Patel", "Brown",
        "Garcia", "Smith", "Lopez", "Walker", "Young", "Hall", "Khan", "Rossi", "Silva", "Cohen"]
CITIES = [("Columbus", "OH", "Ohio"), ("Cleveland", "OH", "Ohio"), ("Cincinnati", "OH", "Ohio"),
          ("Toledo", "OH", "Ohio"), ("Austin", "TX", "Texas"), ("Dallas", "TX", "Texas"),
          ("Houston", "TX", "Texas"), ("Seattle", "WA", "Washington"), ("Spokane", "WA", "Washington"),
          ("Denver", "CO", "Colorado"), ("Miami", "FL", "Florida"), ("Tampa", "FL", "Florida"),
          ("Chicago", "IL", "Illinois"), ("Boston", "MA", "Massachusetts"),
          ("Phoenix", "AZ", "Arizona"), ("Atlanta", "GA", "Georgia")]
PAYMENTS = ["card", "paypal", "cash", "gift card"]
ITEMS = ["laptop", "hdmi cable", "headphones", "gaming pc", "mouse", "coffee maker", "monitor",
         "desk lamp", "keyboard", "webcam", "tv", "sofa", "rug", "tablet", "printer", "router"]


def _opt(o, parts):
    """Only the optional fields this order has: real APIs send fields that come and go."""
    return [text.format(o[k]) for k, text in parts if o.get(k)]


def _raft(o):
    extra = _opt(o, [("date", "Date={}"), ("email", "Email={}"), ("payment", "Payment={}")])
    return ", ".join([f"Order {o['id']}: Buyer={o['buyer']}", f"Location={o['city']}", o["state"],
                      f"Total=${o['total']:.2f}", *extra, f"Items: {', '.join(o['items'])}"])


def _pipes(o):
    extra = _opt(o, [("date", "{}"), ("email", "{}"), ("payment", "paid by {}")])
    return " | ".join([f"#{o['id']}", f"buyer: {o['buyer']}", f"{o['city']}, {o['state_name']}",
                       f"${o['total']:,.2f}", *extra, f"items: {', '.join(o['items'])}"])


def _sentence(o):
    email = f" ({o['email']})" if o.get("email") else ""
    on = f" on {o['date']}" if o.get("date") else ""
    paid = f", paid by {o['payment']}" if o.get("payment") else ""
    return (f"{o['buyer']}{email} placed order {o['id']}{on} for ${o['total']:,.2f} "
            f"({o['city']}, {o['state_name']}){paid}: {', '.join(o['items'])}")


def _key_value(o):
    extra = _opt(o, [("date", "date={}"), ("email", "email={}"), ("payment", "pay={}")])
    return ";".join([f"order={o['id']}", f"customer={o['buyer']}", f"city={o['city']}",
                     f"st={o['state']}", f"amount={o['total']:.2f}", *extra,
                     f"items={'/'.join(o['items'])}"])


def _slashes(o):
    extra = _opt(o, [("date", "{}"), ("email", "{}"), ("payment", "{}")])
    return " / ".join([str(o["id"]), o["buyer"], f"{o['city']} {o['state']}",
                       f"{o['total']:.2f} USD", *extra, "; ".join(o["items"])])


MESSY = [_pipes, _sentence, _key_value, _slashes]


def generate_rows(n: int = 50, seed: int = 42) -> list[dict]:
    """The structured orders behind the text, so answers can be checked."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        city, state, state_name = rng.choice(CITIES)
        first, last = rng.choice(FIRST), rng.choice(LAST)
        rows.append({
            "id": 2001 + i, "buyer": f"{first} {last}",
            "city": city, "state": state, "state_name": state_name,
            "total": round(rng.uniform(5, 2500), 2), "items": rng.sample(ITEMS, rng.randint(1, 3)),
            "date": (date(2026, 1, 1) + timedelta(days=rng.randint(0, 272))).isoformat(),
            "email": f"{first}.{last}{i}@example.com".lower(),
            "payment": rng.choice(PAYMENTS),
            "format": _raft if rng.random() < 0.7 else rng.choice(MESSY),
        })
        # Each optional field is missing from about 1 in 5 orders.
        for field in ("date", "email", "payment"):
            if rng.random() < 0.2:
                rows[-1][field] = None
    return rows


def generate(n: int = 50, seed: int = 42) -> list[str]:
    orders = [o["format"](o) for o in generate_rows(n, seed)]

    # Bad records on purpose.
    orders[15] = ""                                                              # blank
    orders[30] = "Order 2031: Buyer=Sam Fox, Location=Dayton, OH, Total=pending, Items: chair"  # no total
    orders[45] = _raft({"id": 2010, "buyer": "Kim Duplicate", "city": "Austin", "state": "TX",
                         "state_name": "Texas", "total": 99.99, "items": ["mouse"]})  # duplicate ID
    return orders


def use_extended() -> None:
    """Swap the 5 orders above for the generated ones (same endpoints, same shape)."""
    ORDERS[:] = generate()


# The extended data is also always served under /extended, so one deployment
# can offer both (the web UI's "extended" toggle uses these routes).
EXTENDED_ORDERS = generate()


@app.route("/extended/api/orders", methods=["GET"])
def get_extended_orders():
    return jsonify({"status": "ok", "raw_orders": random.sample(EXTENDED_ORDERS, len(EXTENDED_ORDERS))})


@app.route("/extended/api/order/<order_id>", methods=["GET"])
def get_extended_order_by_id(order_id):
    for text in EXTENDED_ORDERS:
        if text and order_id in text:  # same fuzzy match as above
            return jsonify({"status": "ok", "raw_order": text})
    return jsonify({"status": "not_found"}), 404


if __name__ == "__main__":
    if "--extended" in sys.argv:
        use_extended()
    app.run(host="0.0.0.0", port=5001, debug=True)