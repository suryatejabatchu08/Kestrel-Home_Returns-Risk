"""Train the Kestrel returns-risk model, validate it on time folds, price the decision, write predictions.

    python train.py --data /path/to/pack            # needs train.csv test_unlabelled.csv customers.csv products.csv
Outputs: model/model.json, predictions.csv, evidence/metrics.json
Only pre-dispatch columns are used (see LEAKY below). No paid API, no network.
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Columns the warehouse does NOT have at dispatch. They are filled in *after* a return is raised/approved, or are
# end-of-life service states, so they "predict" returns at 99% in train and are empty/different in the live test file.
LEAKY = ["pickup_scheduled_at", "last_service_event_type"]
NUM = ["discount_pct", "promised_delivery_days", "customer_prior_returns", "prior_return_rate",
       "log_prior_orders", "is_gift", "shield", "pincode_default"]
CATS = {"payment_mode": ["cod", "emi", "prepaid_card", "prepaid_upi"],
        "family": None, "sales_channel": ["app", "marketplace", "partner_outlet", "web"]}
RETURN_COST, CALL_COST, PILOT_EFFECT, CANCEL_RATE = 1150, 45, 0.35, 0.12   # ops-policy.pdf s4, s7
CONSERVATIVE_EFFECT = 0.25  # pilot was not a controlled trial -> plan on less than 35%

def load(dirpath):
    r = lambda f: pd.read_csv(f"{dirpath}/{f}")
    return r("train.csv"), r("test_unlabelled.csv"), r("customers.csv"), r("products.csv")

def prep(df, cust, prod):
    if "source" in df: df = df[df.source == "crm"]             # partner_feed rows are exact re-imports (651 dups)
    d = df.merge(prod[["sku", "family", "list_price_inr"]], on="sku", how="left").merge(cust[["customer_id", "shield_member"]], on="customer_id", how="left")
    d["order_t"] = pd.to_datetime(d.order_placed_at)
    exp = d.list_price_inr * d.qty * (1 - d.discount_pct / 100)
    d["order_value_fixed"] = np.where(d.order_value_inr / exp > 50, d.order_value_inr / 100, d.order_value_inr)  # Oct-25 gateway x100
    d["pincode_default"] = (d.delivery_pincode.astype(int) == 0).astype(int)
    d["prior_return_rate"] = (d.customer_prior_returns + 0.1) / (d.customer_prior_orders + 1)
    d["log_prior_orders"] = np.log1p(d.customer_prior_orders)
    d["is_gift"] = (d.is_gift == "Y").astype(int); d["shield"] = (d.shield_member == "Y").astype(int)
    return d

def design(d, families):
    X = d[NUM].astype(float).copy()
    cats = {**CATS, "family": families}
    for c, levels in cats.items():
        for l in levels: X[f"{c}={l}"] = (d[c] == l).astype(float)
    return X

def fit(X, y):
    sc = StandardScaler().fit(X); m = LogisticRegression(C=0.2, max_iter=3000).fit(sc.transform(X), y)
    return sc, m
def predict(sc, m, X): return m.predict_proba(sc.transform(X))[:, 1]

def call_net(p, y, thr, eff): s = p >= thr; return eff * y[s].sum() * RETURN_COST - CALL_COST * s.sum()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); a = ap.parse_args()
    tr, te, cust, prod = load(a.data)
    raw_n = len(tr); d = prep(tr, cust, prod).sort_values("order_t").reset_index(drop=True); t = prep(te, cust, prod)
    fams = sorted(prod.family.unique()); X = design(d, fams); y = d.returned.values
    M = {"rows_raw": raw_n, "rows_after_dedupe": len(d), "base_rate": float(y.mean())}

    # rolling-origin validation: always train on the past, score the next block
    folds = [("2025-12-01", "2026-01-01"), ("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]
    M["folds"] = []
    for s, e in folds:
        f = d.order_t < s; v = (d.order_t >= s) & (d.order_t < e)
        sc, m = fit(X[f], y[f]); p = predict(sc, m, X[v])
        M["folds"].append({"from": s, "to": e, "n": int(v.sum()), "base": float(y[v].mean()), "auc": roc_auc_score(y[v], p), "pr_auc": average_precision_score(y[v], p)})
    M["auc_mean"] = float(np.mean([x["auc"] for x in M["folds"]])); M["pr_auc_mean"] = float(np.mean([x["pr_auc"] for x in M["folds"]]))

    # headline hold-out = last quarter (same length/position as the real test file)
    f = d.order_t < "2026-04-01"; v = ~f
    sc, m = fit(X[f], y[f]); p = predict(sc, m, X[v]); yv = y[v]; Vv = d.order_value_fixed[v].values
    M["holdout"] = {"n": int(v.sum()), "returns": int(yv.sum()), "auc": roc_auc_score(yv, p), "pr_auc": average_precision_score(yv, p),
                    "acc_at_0.5": float(((p >= .5) == yv).mean()), "acc_predict_none": float(1 - yv.mean())}
    # leaky comparison (what a 95%-accuracy model would be doing)
    lk = pd.concat([X, d.pickup_scheduled_at.notna().astype(float).rename("pickup_set")], axis=1)
    sc2, m2 = fit(lk[f], y[f]); p2 = predict(sc2, m2, lk[v]); M["leaky_holdout"] = {"auc": roc_auc_score(yv, p2), "acc_at_0.5": float(((p2 >= .5) == yv).mean())}

    # decision economics
    be = lambda eff: CALL_COST / (eff * RETURN_COST)
    thr = be(CONSERVATIVE_EFFECT); M["threshold"] = thr; M["break_even_p_pilot"] = be(PILOT_EFFECT)
    sel = p >= thr
    M["call_policy"] = {"threshold": thr, "called": int(sel.sum()), "called_share": float(sel.mean()), "returns_in_called": int(yv[sel].sum()),
                        "recall": float(yv[sel].sum() / yv.sum()), "precision": float(yv[sel].mean()),
                        "net_at_35": call_net(p, yv, thr, .35), "net_at_25": call_net(p, yv, thr, .25), "net_at_15": call_net(p, yv, thr, .15),
                        "call_everyone_net_at_35": call_net(p, yv, 0, .35), "call_everyone_net_at_25": call_net(p, yv, 0, .25),
                        "shield_share_of_calls": float(d.shield[v].values[sel].mean()), "shield_share_overall": float(d.shield[v].mean())}
    rng = np.random.default_rng(0); nets = []
    for _ in range(2000):
        b = rng.integers(0, len(yv), len(yv)); nets.append(call_net(p[b], yv[b], thr, .25))
    M["call_policy"]["net_at_25_p5_p95"] = [float(x) for x in np.percentile(nets, [5, 95])]
    M["sensitivity"] = [{"effect": e, "break_even_p": be(e), "net_rs": call_net(p, yv, be(e), e)} for e in (.15, .20, .25, .35)]
    M["hold_policy"] = [{"margin": mg, "threshold": th, "n_held": int((p >= th).sum()),
                         "net_rs": float(CANCEL_RATE * ((yv[p >= th] * RETURN_COST).sum() - ((1 - yv[p >= th]) * mg * Vv[p >= th]).sum()))}
                        for mg in (.10, .20, .30) for th in (.157, .30, .50)]
    cal = pd.DataFrame({"p": p, "y": yv}); cal["q"] = pd.qcut(cal.p.rank(method="first"), 5, labels=False)
    M["calibration"] = cal.groupby("q").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(4).reset_index().to_dict("records")

    # final model on all labelled data -> model.json (plain coefficients: auditable, no pickle, no sklearn at serve time)
    sc, m = fit(X, y); pt = predict(sc, m, design(t, fams))
    sub = pd.DataFrame({"order_id": t.order_id, "score": pt.round(6)})
    ss = pd.read_csv(f"{a.data}/sample_submission.csv"); assert set(ss.order_id) == set(sub.order_id) and len(sub) == len(ss)
    sub.set_index("order_id").loc[ss.order_id].reset_index().to_csv("predictions.csv", index=False)
    M["test"] = {"n": len(sub), "mean_score": float(pt.mean()), "expected_returns": float(pt.sum()), "n_at_or_above_threshold": int((pt >= thr).sum())}
    json.dump({"features": list(X.columns), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
               "families": fams, "sku_family": dict(zip(prod.sku, prod.family)), "threshold": thr, "costs": {"return": RETURN_COST, "call": CALL_COST, "effect_planning": CONSERVATIVE_EFFECT},
               "num": NUM, "cats": {**CATS, "family": fams}, "trained_rows": len(d), "trained_through": str(d.order_t.max())}, open("model/model.json", "w"), indent=1)
    json.dump(M, open("evidence/metrics.json", "w"), indent=1, default=float); print(json.dumps(M, indent=1, default=float))
