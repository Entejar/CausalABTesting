"""
One-command pipeline: simulate a zero-inflated revenue experiment, run A/B
baselines (diff-in-means, CUPED), estimate ATE (AIPW) and heterogeneity
(T-learner uplift + Qini), and print a stakeholder-style summary.

Usage:
    python scripts/run_experiment.py --n 200000 --tau 0.25 --zi 0.7 --seed 13
"""
from __future__ import annotations
import argparse
import os
import sys

# Make src/ importable whether run from repo root or elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import numpy as np  # noqa: E402

from simulate import make_synthetic  # noqa: E402
from ab_test import diff_in_means, cuped, mde_power  # noqa: E402
from causal import aipw_ate, t_learner_cate  # noqa: E402
from metrics import uplift_curve, qini_coefficient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal A/B testing demo pipeline.")
    p.add_argument("--n", type=int, default=200_000, help="total samples (traffic)")
    p.add_argument("--tau", type=float, default=0.25, help="avg treatment effect (signal)")
    p.add_argument("--zi", type=float, default=0.7, help="zero-inflation (sparsity)")
    p.add_argument("--seed", type=int, default=13, help="random seed")
    p.add_argument("--model", choices=["xgb", "rf"], default="rf",
                   help="uplift base learner (rf avoids the xgboost dependency)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = make_synthetic(n=args.n, tau=args.tau, zi=args.zi, seed=args.seed)
    feats = [c for c in df.columns if c.startswith("x")]

    # Oracle ATE is available because this is a simulation (both potential outcomes known).
    true_ate = float((df["y1"] - df["y0"]).mean())

    dm = diff_in_means(df)
    cp = cuped(df)
    mde = mde_power(args.n, sd=float(df["y"].std()))

    ate = aipw_ate(df, y="y", t="t", X=feats)
    cate = t_learner_cate(df, y="y", t="t", X=feats, model=args.model)
    qini = qini_coefficient(df, cate)
    deciles = uplift_curve(df, cate)

    print("=" * 60)
    print(f"Causal A/B Testing pipeline  (n={args.n:,}, tau={args.tau}, "
          f"zi={args.zi}, seed={args.seed})")
    print("=" * 60)
    print(f"Oracle ATE (simulation truth) : {true_ate:.4f}")
    print("-" * 60)
    print("A/B baselines")
    print(f"  diff-in-means ATE           : {dm.ate:.4f}  "
          f"(95% CI [{dm.ci_low:.4f}, {dm.ci_high:.4f}], p={dm.pvalue:.3g})")
    print(f"  CUPED ATE                   : {cp.ate:.4f}  "
          f"(variance reduction {cp.cuped_gain:.1%})")
    print(f"  MDE @ 80% power             : {mde:.4f}")
    print("-" * 60)
    print("Causal estimators")
    print(f"  AIPW / doubly-robust ATE    : {ate:.4f}")
    print(f"  T-learner CATE mean         : {cate.mean():.4f}  "
          f"(range {cate.min():.2f} to {cate.max():.2f})")
    print(f"  Qini coefficient            : {qini:.4f}")
    print("-" * 60)
    print("Uplift by decile (top = highest predicted lift)")
    print(deciles.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
