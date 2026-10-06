import json, unittest, app as A
c = A.app.test_client()
BASE = json.load(open("static/example.json"))
class T(unittest.TestCase):
    def test_flags_risky(self):
        r = c.post("/predict", json=BASE).get_json(); self.assertTrue(r["risk_flag"]); self.assertGreater(r["return_probability"], .3); self.assertTrue(r["reasons"])
    def test_safe_order_ships(self):
        o = dict(BASE, sku="KH-CF-01", payment_mode="prepaid_upi", discount_pct=5, promised_delivery_days=3, customer_prior_returns=0, customer_prior_orders=0, shield_member="N")
        r = c.post("/predict", json=o).get_json(); self.assertFalse(r["risk_flag"]); self.assertLess(r["return_probability"], .1)
    def test_leaky_fields_ignored_not_used(self):
        a = c.post("/predict", json=BASE).get_json(); b = c.post("/predict", json=dict(BASE, pickup_scheduled_at="2026-01-01 10:00", last_service_event_type="REVERSE_PICKUP")).get_json()
        self.assertEqual(a["return_probability"], b["return_probability"]); self.assertEqual(len(b["ignored_fields"]), 2)
    def test_polite_errors(self):
        self.assertEqual(c.post("/predict", data="nope", content_type="text/plain").status_code, 400)
        r = c.post("/predict", json={"sku": "KH-XX-99"}); self.assertEqual(r.status_code, 422); self.assertIn("problems", r.get_json())
    @unittest.skipUnless(__import__("os").path.exists("data/customers.csv"), "needs the private data pack in data/")
    def test_matches_batch_file(self):  # service must reproduce predictions.csv for a real row
        import csv; row = next(csv.DictReader(open("data/test_unlabelled.csv"))); pred = next(csv.DictReader(open("predictions.csv")))
        cust = {r["customer_id"]: r["shield_member"] for r in csv.DictReader(open("data/customers.csv"))}
        rec = dict(row, shield_member=cust[row["customer_id"]]); r = c.post("/predict", json=rec).get_json()
        self.assertAlmostEqual(r["return_probability"], float(pred["score"]), places=3)
unittest.main() if __name__ == "__main__" else None
