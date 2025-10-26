"""
Causal estimators: AIPW/DR, S-learner, T-learner with sklearn / xgboost.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def _features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy()

def aipw_ate(df: pd.DataFrame, y: str, t: str, X: list[str]) -> float:
    # Propensity known (RCT = 0.5), but we still allow estimation for demo
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(df[X], df[t])
    ps = ps_model.predict_proba(df[X])[:,1]
    mu0 = Ridge(alpha=1.0).fit(df.loc[df[t]==0, X], df.loc[df[t]==0, y]).predict(df[X])
    mu1 = Ridge(alpha=1.0).fit(df.loc[df[t]==1, X], df.loc[df[t]==1, y]).predict(df[X])
    y_arr = df[y].to_numpy()
    t_arr = df[t].to_numpy()
    # AIPW influence function
    ipw1 = t_arr*(y_arr - mu1)/np.clip(ps,1e-3,1-1e-3)
    ipw0 = (1-t_arr)*(y_arr - mu0)/np.clip(1-ps,1e-3,1-1e-3)
    dr = (mu1 - mu0) + ipw1 - ipw0
    return float(dr.mean())

def t_learner_cate(df: pd.DataFrame, y: str, t: str, X: list[str], model="xgb") -> np.ndarray:
    if model == "xgb":
        f0 = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                          n_jobs=-1, random_state=13)
        f1 = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                          n_jobs=-1, random_state=13)
    else:
        f0 = RandomForestRegressor(n_estimators=300, random_state=13, n_jobs=-1)
        f1 = RandomForestRegressor(n_estimators=300, random_state=13, n_jobs=-1)
    f0.fit(df.loc[df[t]==0, X], df.loc[df[t]==0, y])
    f1.fit(df.loc[df[t]==1, X], df.loc[df[t]==1, y])
    mu0 = f0.predict(df[X])
    mu1 = f1.predict(df[X])
    return mu1 - mu0
