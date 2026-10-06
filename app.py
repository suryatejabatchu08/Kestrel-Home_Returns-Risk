"""Kestrel returns-risk service.  python app.py  ->  http://localhost:8000
POST /predict  one order as JSON -> score, action, reasons.   GET / -> the screen.   GET /health.
Needs only Flask + numpy. No model API, no key, no network: nothing here costs money per prediction."""
import json, math, os
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = json.load(open(os.path.join(HERE, "model", "model.json")))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
CUSTOMERS = {}   # optional lookup of shield_member by customer_id if data/customers.csv exists (never committed: ops-policy s10)
_cp = os.path.join(HERE, "data", "customers.csv")
if os.path.exists(_cp):
    import csv
    CUSTOMERS = {r["customer_id"]: r["shield_member"] for r in csv.DictReader(open(_cp))}

# Post-outcome fields: accepted but deliberately NOT used, and we tell the caller.
IGNORED = {"pickup_scheduled_at", "last_service_event_type", "returned"}
REQUIRED = ["sku", "payment_mode", "sales_channel", "discount_pct", "promised_delivery_days",
            "customer_prior_orders", "customer_prior_returns"]
LABEL = {"payment_mode": "Payment", "family": "Product", "sales_channel": "Channel"}
PAY = {"cod": "Cash on delivery", "emi": "EMI", "prepaid_card": "Paid by card", "prepaid_upi": "Paid by UPI"}

def _num(rec, k, lo, hi, errs, default=None):
    v = rec.get(k, default)
    try: v = float(v)
    except (TypeError, ValueError): errs.append(f"'{k}' must be a number"); return None
    if not (lo <= v <= hi): errs.append(f"'{k}' must be between {lo} and {hi}"); return None
    return v

def parse(rec):
    errs = []; warn = []
    for k in REQUIRED:
        if rec.get(k) in (None, ""): errs.append(f"'{k}' is missing")
    if errs: return None, errs, warn
    sku = str(rec["sku"]).upper()
    if sku not in MODEL["sku_family"]: errs.append(f"unknown sku '{rec['sku']}' (known: {', '.join(sorted(MODEL['sku_family']))})")
    for k in ("payment_mode", "sales_channel"):
        if rec[k] not in MODEL["cats"][k]: errs.append(f"'{k}' must be one of {MODEL['cats'][k]}")
    disc = _num(rec, "discount_pct", 0, 90, errs); days = _num(rec, "promised_delivery_days", 0, 60, errs)
    po = _num(rec, "customer_prior_orders", 0, 500, errs); pr = _num(rec, "customer_prior_returns", 0, 500, errs)
    if po is not None and pr is not None and pr > po: warn.append("prior returns exceed prior orders - check the customer history")
    shield = str(rec.get("shield_member", "")).upper()
    if shield not in ("Y", "N"):
        shield = CUSTOMERS.get(str(rec.get("customer_id", "")), "")
        if shield == "": shield = "N"; warn.append("shield_member not supplied - assumed N (Shield orders score higher; supply it for a better score)")
    pin = str(rec.get("delivery_pincode", "")).strip(); default_pin = int(pin in ("", "0", "000000"))
    if errs: return None, errs, warn
    return {"sku": sku, "family": MODEL["sku_family"][sku], "payment_mode": rec["payment_mode"], "sales_channel": rec["sales_channel"],
            "discount_pct": disc, "promised_delivery_days": days, "customer_prior_orders": po, "customer_prior_returns": pr,
            "prior_return_rate": (pr + 0.1) / (po + 1), "log_prior_orders": math.log1p(po),
            "is_gift": int(str(rec.get("is_gift", "N")).upper() == "Y"), "shield": int(shield == "Y"), "pincode_default": default_pin}, errs, warn

def score(x):
    contrib = {}; z = MODEL["intercept"]
    for name, mu, sd, w in zip(MODEL["features"], MODEL["mean"], MODEL["scale"], MODEL["coef"]):
        if "=" in name: var, lvl = name.split("="); val = float(x[var] == lvl)
        else: var, val = name, float(x[name])
        c = w * (val - mu) / sd; contrib[var] = contrib.get(var, 0.0) + c
    z = MODEL["intercept"] + sum(contrib.values())
    return 1 / (1 + math.exp(-z)), contrib

def phrase(var, x, up):
    n = int(x["customer_prior_returns"])
    P = {"customer_prior_returns": f"Customer has returned {n} earlier order{'s' if n != 1 else ''}" if n else "Customer has never returned an order",
         "prior_return_rate": "Customer's past return rate is " + ("high" if up else "low"),
         "log_prior_orders": "Customer has " + ("a long" if x["customer_prior_orders"] >= 3 else "a short") + " order history",
         "shield": "Kestrel Shield member (free returns)" if x["shield"] else "Not a Shield member",
         "promised_delivery_days": f"Delivery promised in {int(x['promised_delivery_days'])} days" + (" (long wait)" if up else " (quick)"),
         "payment_mode": PAY[x["payment_mode"]], "discount_pct": f"{int(x['discount_pct'])}% discount" + (" (heavy)" if up else ""),
         "family": x["family"] + (" - returned more often than average" if up else " - returned less often than average"),
         "sales_channel": "Bought via " + x["sales_channel"].replace("_", " "), "is_gift": "Marked as a gift" if x["is_gift"] else "Not a gift",
         "pincode_default": "No delivery address captured" if x["pincode_default"] else "Delivery address captured"}
    return P.get(var, var)

@app.post("/predict")
def predict():
    rec = request.get_json(silent=True)
    if not isinstance(rec, dict): return jsonify(error="Send one order as a JSON object.", example="/static/example.json"), 400
    x, errs, warn = parse(rec)
    if errs: return jsonify(error="Could not score this order.", problems=errs), 422
    p, contrib = score(x); c = MODEL["costs"]; flag = p >= MODEL["threshold"]
    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    reasons = [{"effect": "raises risk" if v > 0 else "lowers risk", "text": phrase(k, x, v > 0), "weight": round(v, 2)}
               for k, v in ranked if abs(v) >= 0.08][:5]
    gain = c["effect_planning"] * p * c["return"] - c["call"]
    return jsonify(order_id=rec.get("order_id"), return_probability=round(p, 4), risk_flag=bool(flag),
                   recommended_action=("Phone the customer to confirm model and address BEFORE dispatch, then ship within 24h. Do not hold the parcel waiting."
                                       if flag else "Ship normally. A confirmation call would cost more than it is expected to save."),
                   expected_net_rs_if_called=round(gain), threshold=round(MODEL["threshold"], 4), reasons=reasons,
                   ignored_fields=sorted(IGNORED & set(rec)), warnings=warn,
                   caveat="Estimates risk, not certainty. Typical hit rate among flagged orders is about 3 in 10.",
                   model={"type": "logistic regression", "trained_through": MODEL["trained_through"][:10]})

@app.get("/health")
def health(): return jsonify(status="ok", trained_rows=MODEL["trained_rows"])

@app.get("/")
def home(): return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
