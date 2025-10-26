"""
Classical A/B testing utilities: diff-in-means, Welch t-test, CUPED, power.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ABResult:
    ate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    cuped_gain: float | None = None

def diff_in_means(df: pd.DataFrame, y: str = "y", t: str = "t", alpha: float = 0.05) -> ABResult:
    g = df.groupby(t)[y]
    y1, y0 = g.get_group(1), g.get_group(0)
    ate = y1.mean() - y0.mean()
    # Welch SE
    n1, n0 = y1.size, y0.size
    v1, v0 = y1.var(ddof=1), y0.var(ddof=1)
    se = np.sqrt(v1/n1 + v0/n0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    ci_low, ci_high = ate - z*se, ate + z*se
    # p-value
    pvalue = 2*(1 - norm.cdf(abs(ate)/se))
    return ABResult(ate, se, ci_low, ci_high, pvalue)

def cuped(df: pd.DataFrame, y: str = "y", t: str = "t", cov: str = "s", alpha: float = 0.05) -> ABResult:
    # pre-exposure proxy covariate 's' acts like a baseline metric
    y_adj = df[y] - (df[y].cov(df[cov]) / df[cov].var()) * df[cov]
    df2 = df.copy()
    df2["_y_adj"] = y_adj
    res = diff_in_means(df2, y="_y_adj", t=t, alpha=alpha)
    # variance reduction gain
    gain = 1 - (df2["_y_adj"].var() / df[y].var())
    res.cuped_gain = float(gain)
    return res

def mde_power(n: int, sd: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> float:
    """
    Minimum detectable effect (two-sided Z test) for equal variance, group ratio.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n1 = n * ratio / (1 + ratio)
    n0 = n - n1
    se = sd*np.sqrt(1/n1 + 1/n0)
    return (z_alpha + z_beta) * se
