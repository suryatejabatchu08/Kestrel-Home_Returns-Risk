# Kestrel Home - pre-dispatch return check (Variant A)

Scores an order *before dispatch* for its chance of coming back, and says what to do about it.
Plain logistic regression. **No model API, no key, no network, Rs 0 per prediction.**

## Run it (clean machine, ~1 minute, Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # flask + numpy only
python app.py                                          # open http://127.0.0.1:8000
python -m unittest tests/test_service.py               # 4 tests (5th runs only if the private data/ pack is present)
```
One endpoint: `POST /predict` with one order as JSON (example: `static/example.json`) ->
`return_probability`, `risk_flag`, `recommended_action`, `expected_net_rs_if_called`, `reasons[]`, `warnings[]`, `ignored_fields[]`.
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d @static/example.json
```
Bad input gets a 422 with a plain list of problems, not a stack trace. Post-return fields (`pickup_scheduled_at`,
`last_service_event_type`) are accepted but ignored, and the response says so.

## Retrain / regenerate predictions.csv (needs the private data pack)
Kestrel's policy (s10) forbids publishing customer/operational data, so **no data is in this repo**. Copy the five CSVs into `data/`, then:
```bash
pip install -r requirements-train.txt
python train.py --data data      # -> model/model.json, predictions.csv, evidence/metrics.json (about 10 s)
```

## What is where
`train.py` cleaning, time-fold validation, rupee model, final fit | `app.py` service | `static/index.html` screen |
`model/model.json` the whole model as readable coefficients | `evidence/` how I know it works | `memo/` note to Ritu | `submission-form.md`

## The three things to know on Monday
1. **Never feed it the service/pickup columns.** They are filled in after a return is raised; they make any model look 99% accurate in training and are empty or different at dispatch. `LEAKY` in `train.py` is the guard.
2. **The action is "phone before dispatch", not "hold".** Thresholds come from the policy costs (Rs 1,150 return, Rs 45 call). If those or the call effect change, change `RETURN_COST` / `CALL_COST` / `CONSERVATIVE_EFFECT` in `train.py` and retrain; the threshold moves itself.
3. **The call effect (25% planned, 35% pilot) is unproven.** Run flagged orders with a random half *not* called for 4 weeks to measure it; retrain monthly; check that `customer_prior_returns` is computed the way the data pack assumes.
