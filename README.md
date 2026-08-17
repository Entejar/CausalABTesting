# Causal A/B Testing for Product & Marketing Impact
Data science project demonstrating rigorous A/B testing and causal inference on user-level, semi-continuous outcomes (e.g., revenue with many zeros), producing stakeholder-ready insights.

**Tech stack:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost. (TensorFlow/Keras optional, for the neural-learner extension.)

## Why this repo
- Show an end-to-end experimental workflow a Business / Product / Research DS would run in a modern growth, ads, or product analytics org.
- Balance **statistical rigor** (design, power, CUPED, AIPW) and **ML for measurement** (uplift / CATE modeling).
- Emphasize **stakeholder-ready insights** with clean visuals and clear recommendations: effect sizes, uncertainty, heterogeneity, and rollout recommendations.

## Highlights
- Synthetic DGP mimicking **zero-inflated revenue** and **treatment effect heterogeneity** (`src/simulate.py`).
- Baseline A/B testing: diff-in-means, Welch t, **CUPED** variance reduction, power/MDE (`src/ab_test.py`).
- Causal estimators: **AIPW/DR** for ATE; **T-learner** with tree / gradient boosting for uplift/CATE (`src/causal.py`, `src/metrics.py`).
- Evaluation: uplift deciles, **Qini** approximation, and clean plots.
- Scale & pitfalls: power/MDE, sequential-peeking cautions, and variance control (`src/ab_test.py`, `notebooks/`).

## Quickstart
```bash
pip install -r requirements.txt

# fast, reproducible run (matches Example Results below)
python scripts/run_experiment.py --n 30000 --tau 0.25 --zi 0.7 --seed 13 --model rf

# fuller run with the boosted learner
python scripts/run_experiment.py --n 200000 --tau 0.25 --zi 0.7 --seed 13 --model xgb
```

**Common flags:**
- `--n` : total samples (traffic)
- `--tau` : average treatment effect (signal strength)
- `--zi` : zero-inflation (sparsity level)
- `--seed` : random seed for reproducibility
- `--model` : uplift base learner, `rf` (default, no xgboost needed) or `xgb`

---

## Repo structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── simulate.py        # zero-inflated revenue + heterogeneity
│   ├── ab_test.py         # diff-in-means, Welch, CUPED, power/MDE
│   ├── causal.py          # AIPW/DR ATE, T-learner uplift (XGBoost / RF)
│   ├── metrics.py         # uplift curve + Qini approx
│   └── viz.py             # uplift plots
├── scripts/
│   └── run_experiment.py  # one-command pipeline
└── notebooks/
    ├── 01_simulation_experiment.ipynb
    ├── 02_ab_testing_baseline.ipynb
    └── 03_causal_uplift_modeling.ipynb
```

---

## What this demonstrates
- **Design:** unit of randomization, guardrail metrics, CUPED covariates, power & MDE planning.
- **Inference:** intent-to-treat vs treatment-on-treated, heteroskedasticity, and non-normal outcomes.
- **Heterogeneity:** uplift/CATE targeting for high-ROI segments with decile summaries & curves.
- **Robustness:** doubly-robust AIPW with overlap diagnostics for observational settings.
- **Communication:** translate ATE/CATE into business impact, risk ranges, and rollout recommendations.

---

## Example Results

Reproducible run (`python scripts/run_experiment.py --n 30000 --tau 0.25 --zi 0.7 --seed 13 --model rf`):

```
Oracle ATE (simulation truth) : 0.358
diff-in-means ATE             : 0.368   (95% CI [0.310, 0.426])
CUPED ATE                     : 0.365   (variance reduction 2.8%)
AIPW / doubly-robust ATE      : 0.371
T-learner CATE mean           : 0.387
Qini coefficient              : 0.312
```

**Interpretation.** All estimators recover the known simulation truth (oracle ATE 0.358), which is the point of validating on synthetic data. Uplift modeling reveals strong treatment-effect heterogeneity: the top decile shows a ~7.5-unit lift versus negative lift in the bottom deciles, so a targeted rollout to high-uplift users captures far more value than a blanket treatment.

---

## Extending this project
This repository is designed as a modular sandbox for experimentation. Possible next steps:
- **More meta-learners:** add S-learner and X-learner alongside the T-learner for a head-to-head uplift comparison.
- **Deep learners for uplift:** use `TensorFlow/Keras` to fit neural T-/X-learners or CEVAE-style models.
- **Real-world datasets:** apply to open ad-click, e-commerce, or engagement data (e.g., Criteo, Kaggle Ads).
- **Observational extensions:** add propensity-score modeling, `DoubleML`, or Difference-in-Differences modules.
- **Streaming experiments:** explore sequential A/B testing or Thompson sampling with Bayesian updates.
- **Dash or Streamlit app:** visualize experiment diagnostics and treatment heterogeneity interactively.

---

## Notes
- `XGBoost` on macOS may require OpenMP (`libomp`). The default `--model rf` path runs without xgboost.
- `TensorFlow/Keras` are optional and only needed for the neural-learner extension; they are not in `requirements.txt`.
- All functions are reproducible and tested under both macOS and Linux environments.

---

