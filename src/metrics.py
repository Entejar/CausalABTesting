"""
Metrics and curves for uplift evaluation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def uplift_curve(df: pd.DataFrame, uplift: np.ndarray, t: str = "t", y: str = "y", bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(uplift, q=bins, labels=False, duplicates="drop")
    d = df.copy()
    d["_decile"] = q
    agg = d.groupby("_decile").apply(
        lambda g: g.loc[g[t]==1, y].mean() - g.loc[g[t]==0, y].mean()
    ).reset_index()
    agg.columns = ["decile", "uplift"]
    return agg.sort_values("decile", ascending=False).reset_index(drop=True)

def qini_coefficient(df: pd.DataFrame, uplift: np.ndarray, t: str = "t", y: str = "y") -> float:
    # Simple trapezoidal approx of Qini (for demo)
    curve = uplift_curve(df, uplift, t=t, y=y)
    x = np.linspace(0, 1, len(curve))
    yvals = curve["uplift"].to_numpy()
    return float(np.trapz(yvals, x))
