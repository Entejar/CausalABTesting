# Causal A/B Testing for Product & Marketing Impact

Data science project demonstrating rigorous A/B testing and causal inference on user-level, semi-continuous outcomes (e.g., revenue with many zeros), producing stakeholder-ready insights.

**Tech stack:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow, Keras.

## Why this repo
- Show an end-to-end experimental workflow a Business / Product / Research DS would run in a modern growth, ads, or product analytics org.
- Balance **statistical rigor** (design, power, CUPED, AIPW) and **ML for measurement** (uplift models, meta-learners).
- Emphasize **stakeholder-ready insights** with clean visuals and clear recommendations: effect sizes, uncertainty, heterogeneity, and rollout recommendations.

## Highlights
- Synthetic DGP mimicking **zero-inflated revenue** and **treatment effect heterogeneity** (`src/simulate.py`).
- Baseline A/B testing: diff-in-means, Welch t, nonparametric bootstrap, **CUPED** (`src/ab_test.py`).
- Causal estimators: **AIPW/DR** for ATE; S-/T-learner with tree/gradient boosting for uplift/CATE (`src/causal.py`, `src/metrics.py`).
- Evaluation: uplift deciles, **Qini** approximation, and clean plots
- Scale & pitfalls: power/MDE, sequential peeking cautions, and variance control (`src/ab_test.py`, `notebooks/`).

## Quickstart
```bash
pip install -r requirements.txt
python scripts/run_experiment.py --n 200000 --tau 0.25 --zi 0.7 --seed 13

**Common flags:**
- `--n` : total samples (traffic)
- `--tau` : average treatment effect (signal strength)
- `--zi` : zero-inflation (sparsity level)
- `--seed` : random seed for reproducibility

---

## Repo structure
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── simulate.py        # zero-inflated revenue + heterogeneity
│   ├── ab_test.py         # diff-in-means, Welch, CUPED, power/MDE
│   ├── causal.py          # AIPW/DR ATE, S/T-learner uplift (XGBoost / RF)
│   ├── metrics.py         # uplift curve + Qini approx
│   ├── viz.py             # uplift plots
│   └── utils.py
├── scripts/
│   └── run_experiment.py  # one-command pipeline
├── notebooks/
│   ├── 01_simulation_experiment.ipynb
│   ├── 02_ab_testing_baseline.ipynb
│   └── 03_causal_uplift_modeling.ipynb
├── tests/
│   └── test_sanity.py
└── artifacts/             # created at runtime


---

## What this demonstrates
- **Design:** unit of randomization, guardrail metrics, CUPED covariates, power & MDE planning.  
- **Inference:** intent-to-treat vs treatment-on-treated, heteroskedasticity, and non-normal outcomes.  
- **Heterogeneity:** uplift/CATE targeting for high-ROI segments with decile summaries & curves.  
- **Robustness:** doubly-robust AIPW with overlap diagnostics for observational settings.  
- **Communication:** translate ATE/CATE into **business impact**, risk ranges, and rollout recommendations.

---

## Talking points (interviews)
- Why CUPED reduces variance and how you select covariates.  
- Interpreting **AIPW** vs. naïve diff-in-means; when doubly-robust methods help.  
- How uplift modeling changes **who** you treat, not just **whether** to treat.  
- Handling **zero-inflated** outcomes (e.g., revenue) and metric choice implications.  
- Avoiding **sequential peeking** and guarding statistical validity at scale.

---

## Example Results

**Average Treatment Effect (AIPW):**
ATE = 0.380 (± 0.018, 95% CI: [0.380, 0.451])


**Top uplift deciles (from T-learner):**
| Decile | Uplift |
|:-------|--------:|
| 9 | 3.419 |
| 8 | 1.147 |
| 7 | 0.674 |
| 6 | 0.346 |
| 5 | 0.107 |

**Interpretation:**  
The AIPW estimate suggests an average lift of ~0.38 units.  
Uplift modeling reveals strong treatment-effect heterogeneity: the top decile shows a 3.4-unit lift versus near-zero for lower deciles — ideal for **targeted rollout or personalization**.

---
## Extending this project
This repository is designed as a modular sandbox for experimentation. Possible next steps:
- **Real-world datasets:** apply to open ad click, e-commerce, or engagement data (e.g., Criteo, Kaggle Ads, Booking.com AB).  
- **Deep learners for uplift:** use `TensorFlow/Keras` to fit neural T-/X-learners or CEVAE-style models.  
- **Observational extensions:** add propensity-score modeling, `DoubleML`, or Difference-in-Differences modules.  
- **Streaming experiments:** explore sequential A/B testing or Thompson sampling with Bayesian updates.  
- **Dash or Streamlit app:** visualize experiment diagnostics and treatment heterogeneity interactively.

---

## Notes
- `XGBoost` on macOS may require OpenMP (`libomp`).   
- `TensorFlow/Keras` are included to demonstrate ML stack readiness and can be extended for deep-learner outcome models.  
- All functions are reproducible and tested under both macOS and Linux environments.

---

## License
Apache-2.0

