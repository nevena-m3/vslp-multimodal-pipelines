from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missingness(summary: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    if summary.empty or "missing_fraction" not in summary.columns:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.text(.5,.5,"No missingness data", ha="center", va="center"); ax.axis("off"); return _save(fig,path)
    s = summary.sort_values("missing_fraction", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(s))))
    ax.barh(s["feature"].astype(str), s["missing_fraction"])
    ax.invert_yaxis(); ax.set_xlabel("Missing fraction"); ax.set_title("Top feature missingness")
    ax.set_xlim(0, 1)
    return _save(fig, path)


def plot_feature_availability_heatmap(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 80) -> Path:
    cols = list(feature_cols)[:max_features]
    if not cols:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.text(.5,.5,"No feature columns detected", ha="center", va="center"); ax.axis("off"); return _save(fig,path)
    mat = (~df[cols].isna()).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(4, min(10, 0.18 * len(df)))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title("Feature availability matrix")
    ax.set_xlabel("Features"); ax.set_ylabel("Rows / recordings")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, label="Available")
    return _save(fig, path)


def plot_distribution_grid(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 12) -> Path:
    cols = list(feature_cols)[:max_features]
    if not cols:
        fig, ax = plt.subplots(figsize=(8,4)); ax.text(.5,.5,"No features to plot", ha="center", va="center"); ax.axis("off"); return _save(fig,path)
    n = len(cols); ncols = 3; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            ax.text(.5,.5,"all missing", ha="center", va="center"); ax.set_title(col); continue
        ax.hist(x, bins=min(20, max(5, int(np.sqrt(len(x))))), alpha=0.85)
        med = x.median(); ax.axvline(med, linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=9); ax.set_ylabel("count")
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Feature distributions: first selected features", y=1.01)
    return _save(fig, path)


def plot_corr_heatmap(corr: pd.DataFrame, path: Path, title: str = "Spearman correlation heatmap") -> Path:
    if corr is None or corr.empty:
        fig, ax = plt.subplots(figsize=(8,5)); ax.text(.5,.5,"Not enough numeric features for correlation", ha="center", va="center"); ax.axis("off"); return _save(fig,path)
    fig, ax = plt.subplots(figsize=(max(7, 0.22*len(corr.columns)), max(6, 0.22*len(corr.index))))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=6)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    return _save(fig, path)


def plot_outlier_counts(outliers: pd.DataFrame, path: Path) -> Path:
    if outliers.empty:
        fig, ax = plt.subplots(figsize=(8,4)); ax.text(.5,.5,"No robust outliers detected", ha="center", va="center"); ax.axis("off"); return _save(fig,path)
    counts = outliers["feature"].value_counts().head(30)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25*len(counts))))
    ax.barh(counts.index.astype(str), counts.values)
    ax.invert_yaxis(); ax.set_xlabel("Outlier count"); ax.set_title("Robust outliers by feature")
    return _save(fig, path)
