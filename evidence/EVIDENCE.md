# Evidence: does it work, and how often does it not

All numbers regenerate with `python train.py --data data` (`evidence/metrics.json`). Honest features only (no service/pickup columns).

## Validation design
Always train on the past, score the future (the real test file is the 3 months after training). Four rolling blocks, plus a last-quarter hold-out
(train Apr-25..Mar-26 -> score Apr-Jun-26, 2,126 orders, 245 returns), which mimics the real test.

| Block scored | Orders | Return rate | AUC | PR-AUC |
|---|---|---|---|---|
| Dec 2025 | 684 | 10.2% | 0.793 | 0.419 |
| Jan 2026 | 719 | 11.3% | 0.766 | 0.384 |
| Feb-Mar 2026 | 1,407 | 11.8% | 0.777 | 0.401 |
| **Apr-Jun 2026 (hold-out)** | 2,126 | 11.5% | **0.788** | **0.417** |
| mean | | | 0.781 | 0.405 |

Chosen over: gradient boosting (AUC 0.76-0.77, worse and less stable), logistic with all features (0.776), blend (0.778). Simple won.

## The 95% accuracy bar
| Model | Accuracy on hold-out |
|---|---|
| Predict "never returns" for everyone | 88.5% |
| **This model, 0.5 cut-off** | **89.5%** |
| Same model + `pickup_scheduled_at` (leak) | 98.5% (AUC 0.989) |

95% is only reachable by reading a field that exists *after* the customer returns. Not available at dispatch (the test file has none).

## Calibration (hold-out, five equal groups, low -> high risk)
| Predicted | 2.0% | 4.3% | 7.1% | 12.0% | 31.5% |
|---|---|---|---|---|---|
| Actual | 2.4% | 3.5% | 7.5% | 12.2% | 32.0% |

Tail: the 50 highest-scored orders were predicted 65%, actual 72%. Probabilities can be read as probabilities.

## Where it is wrong (flag = score >= 15.7%; 450 of 2,126 orders)
- Catches 138 of 245 returns (**56% recall**); 312 flagged orders did not return (**31% precision**, vs 11.5% base rate).
- **Misses 107 returns (44%)**: 93% are customers with no earlier returns, scoring ~9% on average. Nothing in the data at dispatch distinguishes them. This is the ceiling of the data, not a tuning problem.
- **False alarms**: mostly cash-on-delivery, long-delivery-promise, robot-vacuum/purifier orders from repeat or Shield customers who happened to keep the item.
- **Shield members are 22% of orders but 47% of flags** (return rate 20.5% vs 8.9%). The model finds them well (74% recall) which is exactly why holding them is risky.
- Orders from customers with no history: recall only 43%.

## Money (hold-out quarter; 2,126 orders ~ 3 months at 700/month)
Rs 1,150 per return, Rs 45 per call (policy s4). Call effect on returns: pilot said ~35%; I plan on 25% because the pilot is not a controlled trial.
| Call effect | Break-even score | Net, quarter | Net, per month |
|---|---|---|---|
| 15% | 26.1% | Rs 6.5k | ~Rs 2k |
| 20% | 19.6% | Rs 10.9k | ~Rs 3.6k |
| **25% (plan)** | **15.7%** | **Rs 19.4k (90% range 14.6k-24.4k)** | **~Rs 6.4k** |
| 35% (pilot) | 11.2% | Rs 38.8k | ~Rs 12.8k |

Per-month = quarter / 3.04. At the plan threshold the policy rows are: net Rs 19.4k at 25%, Rs 35.3k at 35%, Rs 3.6k at 15% (threshold held fixed at 15.7%).
Calling *everyone* at 25% loses Rs 25k per quarter: the model is what makes calling worth it.

**Holding** (12% of held orders cancel, policy s7), net per quarter: Rs -25k to -114k at the 15.7% threshold depending on margin (10-30%);
only the very top 48 orders (score >= 50%) at a 10% margin break even (+Rs 2k). A hold loses the sale on orders that would have been kept.
Margin is not in the pack; the conclusion holds across 10-30%.

## Honest limits
- Test-period AUC will vary by about +-0.02 from sampling alone (~250 returns). My expectation: AUC 0.78 (0.74-0.81).
- Train/test drift on the final features: adversarial AUC 0.507 (none detectable). Test set repeats 68.5% of known customers.
- The call effect, the margin, and "every flagged call completes" are assumptions, not measurements.
