from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.csv"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def describe_data() -> Dict[str, Any]:
    df = load_df()
    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "head": df.head(5).to_dict(orient="records"),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }

def summary_stats(columns: Optional[List[str]] = None) -> Dict[str, Any]:
    df = load_df()
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")
        df = df[columns]
    stats = df.describe(include="all").to_dict()

    def clean(v: Any) -> Any:
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v

    return {k: {kk: clean(vv) for kk, vv in v.items()} for k, v in stats.items()}

def detect_anomalies_zscore(column: str, z_thresh: float = 2.5) -> Dict[str, Any]:
    df = load_df()
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    series = pd.to_numeric(df[column], errors="coerce")
    mu = series.mean()
    sigma = series.std(ddof=0)

    if sigma == 0 or pd.isna(sigma):
        return {"column": column, "anomalies": [], "note": "std=0; no anomalies"}

    z = (series - mu) / sigma
    idx = z.abs() >= z_thresh

    anomalies = df[idx].copy()
    anomalies["zscore"] = z[idx]
    return {
        "column": column,
        "z_thresh": z_thresh,
        "count": int(anomalies.shape[0]),
        "anomalies": anomalies.to_dict(orient="records"),
    }

def plot_timeseries(date_col: str, value_col: str) -> Dict[str, Any]:
    df = load_df()
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Unknown columns: {date_col}, {value_col}")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col, value_col]).sort_values(date_col)

    fig = plt.figure()
    plt.plot(d[date_col], d[value_col])
    plt.xlabel(date_col)
    plt.ylabel(value_col)
    plt.tight_layout()

    out = PLOTS_DIR / f"{value_col}_timeseries.png"
    fig.savefig(out)
    plt.close(fig)

    return {"plot_path": str(out)}
