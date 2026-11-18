# Model card: Start-Invest

## Summary

Start-Invest ranks startups for a venture fund. Two models are involved:

1. **Follow-on evaluator.** Estimates the probability that a startup raises another funding round within 24 months of the prediction date.
2. **Feed ranker.** Orders candidate startups for a specific fund using thesis similarity, mandate fit (stage, ticket size) and the evaluator's estimate.

Both are explained per prediction with SHAP values.

## Intended use

- A demonstration of ranking, calibration, explanation and evaluation practice for an investment-sourcing product.
- **Not** intended for real investment decisions. The models are trained only on synthetic data.

## Data

| | |
|---|---|
| Source | `start_invest.data.generate`, a seeded synthetic generator |
| Size (default) | 6,000 startups, ~15,000 funding rounds, 400 funds |
| Latent structure | Each startup has a hidden quality that drives funding cadence, traction and founder strength through noisy observables. A per-year funding climate (2021 boom, 2023–24 downturn) shifts outcomes over time. |
| Label | 1 if the startup raises any round in (snapshot, snapshot + 730 days] |
| Ranking ground truth | Graded 0–3 interest per (fund, startup) from the true sector preference, stage and ticket fit, hidden quality and the fund's risk appetite, plus noise. Uses variables the ranker never observes directly. |

## Evaluator

- **Model:** five LightGBM classifiers (different seeds, row and feature bagging), averaged.
- **Features:** 18 point-in-time features. Only events dated on or before the snapshot are used, which is enforced by `test_features_ignore_future_rounds`.
- **Validation:** time-based. Train on snapshots before 2022, early-stop and calibrate on 2022, test on 2023 to mid-2024.
- **Calibration:** Platt scaling on the log-odds, fit on the 2022 window.
- **Confidence:** the standard deviation across the five models, labelled by tercile across the candidate pool. This is model disagreement, not a full predictive interval.

| Test-period metric (n = 2,255, base rate 29%) | Value |
|---|---|
| ROC AUC | 0.770 |
| Brier score | 0.170 |
| Expected calibration error | 3.4 points raw, 3.1 points calibrated |
| Last-round recency alone (baseline) | AUC 0.679 |
| Random split, for comparison only | AUC 0.767 |

The raw ensemble is already close to calibrated, so calibration helps only a little here. It remains in the pipeline because it is cheap and guards against miscalibration when the data changes.

The random-split AUC is close to the time-split AUC because the generator's drift is mild. Real funding data would likely show a larger gap.

## Ranker

- **Model:** LightGBM LambdaMART (`lambdarank`, NDCG@10), early-stopped on validation funds.
- **Split:** by fund, 60/20/20. Test funds are never seen in training.
- **Candidates:** 60 per fund, drawn from test-period startups (about 40% from the fund's sectors), so the evaluator's inputs are out-of-sample.

| NDCG@10 on 80 held-out funds | Mean | 95% CI |
|---|---|---|
| **LambdaMART ranker** | **0.771** | 0.740 to 0.800 |
| Hand-tuned mandate score | 0.668 | 0.638 to 0.699 |
| Similarity × success | 0.526 | 0.494 to 0.558 |
| Follow-on odds only | 0.435 | 0.402 to 0.466 |
| Thesis similarity only | 0.372 | 0.345 to 0.400 |
| Popularity (total raised) | 0.289 | 0.226 to 0.351 |
| Random | 0.223 | 0.199 to 0.246 |

Paired bootstrap, ranker minus hand-tuned score: +0.103 (95% CI +0.077 to +0.130).

With few test funds the confidence interval widens. At 24 test funds, one of three seeds produced an interval that crossed zero. The test suite therefore evaluates with 50 held-out funds.

## Thesis matching

Investor theses and startup descriptions with founder bios are embedded and compared by cosine similarity. `all-MiniLM-L6-v2` is used when `sentence-transformers` is installed. Otherwise the pipeline falls back to TF-IDF + truncated SVD (96 dimensions), which needs no network access. With the fallback, similarity separates in-sector from out-of-sector pairs with AUC 0.893.

## Explanations

SHAP TreeExplainer values on both models are converted to short reasons. Two guards apply:

- Inputs the user did not provide are never shown as reasons.
- SHAP is relative to an average candidate, so a ticket outside the fund's range can still contribute positively if it is closer than most. The interface then says exactly that and never claims the ticket "matches".

## Limitations and risks

- **Synthetic data.** The ranking ground truth comes from the same generator, so the results show the method recovers structure of this kind. They are not evidence about real investor behaviour.
- **Feedback loops.** In production, ranked feeds shape which startups get attention, and therefore funding. Logged outcomes would be biased toward what the system showed, so off-policy evaluation would be needed.
- **Fairness.** Founder attributes such as experience and prior exits correlate with access to networks. In real data these can encode historical bias. Audit them before any real use, and consider excluding or constraining them.
- **Uncertainty.** Ensemble disagreement does not capture distribution shift. A large change in the funding climate can degrade accuracy without lowering reported confidence.
