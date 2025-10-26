"""
Data generation for semi-continuous outcomes with zero-inflation and heterogeneous treatment effects.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic(n: int = 100_000, p: int = 10, zi: float = 0.6, tau: float = 0.2,
                   seed: int | None = 13) -> pd.DataFrame:
    """
    Create a user-level dataset with:
      - Covariates X ~ N(0,1)
      - Propensity ~ Bernoulli(0.5) (RCT), but effect heterogeneity via f(X)
      - Potential outcomes: Y0 = g0(X) + eps; Y1 = g1(X) + eps; many zeros
      - Semi-continuous outcome: zero-inflated lognormal-like revenue
    Args:
        n: sample size
        p: number of features
        zi: zero-inflation probability (baseline)
        tau: average treatment effect shift
        seed: RNG seed
    Returns:
        DataFrame with columns: ['y', 't', 'y0', 'y1', 'ps', 'user_id', features...]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    cols = [f"x{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    # non-linear signal
    s = 0.5*X[:,0] + 0.3*np.sin(X[:,1]) + 0.2*X[:,2]*X[:,3] - 0.4*(X[:,4] > 0).astype(float)
    # heterogeneity: larger effects for high s and x0>0
    hte = tau * (0.5 + 0.5*(s - s.min())/(s.max() - s.min())) * (X[:,0] > 0).astype(float)
    # zero inflation logits
    logit_zi0 = -np.log(1/zi - 1) + 0.6*X[:,1] - 0.4*X[:,2]
    logit_zi1 = logit_zi0 - 0.3  # treatment reduces zero-prob
    p0 = 1/(1+np.exp(-logit_zi0))
    p1 = 1/(1+np.exp(-logit_zi1))
    # positive part log-scale
    mu0 = 1.0 + 0.7*s
    mu1 = mu0 + hte
    # draw potential outcomes
    eps0 = rng.normal(scale=0.7, size=n)
    eps1 = rng.normal(scale=0.7, size=n)
    y0_pos = np.exp(mu0 + eps0)
    y1_pos = np.exp(mu1 + eps1)
    y0 = (rng.random(n) > p0) * y0_pos
    y1 = (rng.random(n) > p1) * y1_pos
    # assign treatment (RCT)
    t = rng.integers(0, 2, size=n)
    y = y0*(1-t) + y1*t
    ps = np.full(n, 0.5)
    df = pd.concat([df, pd.DataFrame({
        "user_id": np.arange(n),
        "t": t.astype(int),
        "y": y,
        "y0": y0,
        "y1": y1,
        "ps": ps,
        "s": s,
        "hte": hte
    })], axis=1)
    return df
