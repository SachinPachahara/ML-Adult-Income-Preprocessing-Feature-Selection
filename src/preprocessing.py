"""
From-scratch preprocessing utilities for the Adult Census Income assignment.
These functions intentionally use basic Python/Pandas/NumPy logic rather than
ready-made sklearn preprocessing classes for the primary implementation.
"""
import math
import numpy as np
import pandas as pd

def manual_mean(values):
    vals = [float(v) for v in values if pd.notna(v)]
    return sum(vals) / len(vals) if vals else np.nan

def manual_median(values):
    vals = sorted(float(v) for v in values if pd.notna(v))
    n = len(vals)
    if n == 0: return np.nan
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid-1] + vals[mid]) / 2

def manual_mode(values):
    counts = {}
    for v in values:
        if pd.notna(v):
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else np.nan

def manual_variance(values):
    vals = [float(v) for v in values if pd.notna(v)]
    mu = manual_mean(vals)
    return sum((x-mu)**2 for x in vals) / len(vals) if vals else np.nan

def manual_std(values):
    return math.sqrt(manual_variance(values))

def manual_minmax(values):
    vals = [float(v) for v in values]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [0.0] * len(vals)
    return [(x-lo)/(hi-lo) for x in vals]

def manual_standardize(values, mean=None, std=None):
    vals = [float(v) for v in values]
    mu = manual_mean(vals) if mean is None else mean
    sigma = manual_std(vals) if std is None else std
    return [0.0 if sigma == 0 else (x-mu)/sigma for x in vals], mu, sigma

def manual_label_encode(series):
    mapping = {}
    encoded = []
    for value in series:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return pd.Series(encoded, index=series.index), mapping

def manual_one_hot(series):
    categories = sorted(series.dropna().unique().tolist())
    result = pd.DataFrame(index=series.index)
    for cat in categories:
        result[f"{series.name}_{cat}"] = (series == cat).astype(int)
    return result

def manual_iqr_bounds(values):
    s = pd.Series(values).dropna().sort_values()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3-q1
    return q1, q3, iqr, q1-1.5*iqr, q3+1.5*iqr

def manual_z_scores(values):
    vals = np.array(values, dtype=float)
    mu = vals.mean()
    sigma = vals.std()
    return np.zeros_like(vals) if sigma == 0 else (vals-mu)/sigma

def manual_train_test_split(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_test = int(round(len(df)*test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

def clean_category(series):
    return series.astype(str).str.strip().str.lower()
