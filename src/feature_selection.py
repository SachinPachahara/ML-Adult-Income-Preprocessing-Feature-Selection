"""
From-scratch feature-selection utilities for the Adult Census Income assignment.
"""

import math
import numpy as np
import pandas as pd

def manual_variance(values):
    vals = np.array([float(v) for v in values if pd.notna(v)], dtype=float)
    mu = vals.mean()
    return float(np.mean((vals-mu)**2))

def manual_pearson(x, y):
    pairs = [(float(a), float(b)) for a,b in zip(x,y) if pd.notna(a) and pd.notna(b)]
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    xbar, ybar = xs.mean(), ys.mean()
    numerator = np.sum((xs-xbar)*(ys-ybar))
    denominator = math.sqrt(np.sum((xs-xbar)**2)*np.sum((ys-ybar)**2))
    return float(numerator/denominator) if denominator else 0.0

def contingency_table(x, y):
    return pd.crosstab(pd.Series(x, name="X"), pd.Series(y, name="Y"))

def manual_chi_square(table):
    observed = table.to_numpy(dtype=float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand_total = observed.sum()
    expected = row_totals @ col_totals / grand_total
    contributions = (observed-expected)**2 / expected
    chi2 = contributions.sum()
    df = (observed.shape[0]-1)*(observed.shape[1]-1)
    return float(chi2), int(df), pd.DataFrame(expected, index=table.index, columns=table.columns), pd.DataFrame(contributions, index=table.index, columns=table.columns)

def manual_anova(groups):
    arrays = [np.asarray(g, dtype=float) for g in groups if len(g)]
    all_values = np.concatenate(arrays)
    grand_mean = all_values.mean()
    k, n = len(arrays), len(all_values)
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in arrays)
    ss_within = sum(np.sum((g-g.mean())**2) for g in arrays)
    df_between = k-1
    df_within = n-k
    ms_between = ss_between/df_between if df_between else np.nan
    ms_within = ss_within/df_within if df_within else np.nan
    f = ms_between/ms_within if ms_within else np.nan
    return {
        "grand_mean": grand_mean, "ss_between": ss_between, "ss_within": ss_within,
        "df_between": df_between, "df_within": df_within, "f_statistic": f
    }

def entropy(values):
    s = pd.Series(values).dropna()
    probs = s.value_counts(normalize=True)
    return float(-sum(p*math.log2(p) for p in probs if p > 0))

def manual_mutual_information(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n == 0: return 0.0
    joint = df.groupby(["x","y"]).size() / n
    px = df["x"].value_counts(normalize=True)
    py = df["y"].value_counts(normalize=True)
    mi = 0.0
    for (a,b), pxy in joint.items():
        mi += pxy * math.log2(pxy/(px[a]*py[b]))
    return float(mi)
