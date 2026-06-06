from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#071A33"
TEAL = "#2DB7B0"
GOLD = "#B68B2D"
RED = "#B42318"
MUTED = "#607089"
GRID = "#D9E2EF"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _empty(path: Path, message: str, title: str = "VSLP Feature Analysis") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.text(0.5, 0.54, title, ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, color=MUTED, wrap=True)
    ax.axis("off")
    return _save(fig, path)


def _style(ax, title: str | None = None) -> None:
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)


def plot_missingness(summary: pd.DataFrame, path: Path, top_n: int = 30) -> Path:
    if summary is None or summary.empty or "missing_fraction" not in summary.columns:
        return _empty(path, "No missingness data were available.", "Missingness")
    s = summary.copy()
    if "feature" not in s.columns:
        # Fallback: use index as feature name.
        s["feature"] = s.index.astype(str)
    s["missing_fraction"] = pd.to_numeric(s["missing_fraction"], errors="coerce")
    s = s.dropna(subset=["missing_fraction"]).sort_values("missing_fraction", ascending=False).head(top_n)
    if s.empty:
        return _empty(path, "No feature missingness values were available.", "Missingness")
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.32 * len(s))))
    colors = [RED if v >= 0.40 else GOLD if v >= 0.15 else TEAL for v in s["missing_fraction"]]
    ax.barh(s["feature"].astype(str), s["missing_fraction"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Missing fraction", color=MUTED)
    ax.set_xlim(0, 1)
    _style(ax, "Top missing feature columns")
    ax.axvline(0.15, color=GOLD, linestyle="--", linewidth=1, alpha=0.8)
    ax.axvline(0.40, color=RED, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(0.15, -0.8, "15%", color=GOLD, fontsize=8)
    ax.text(0.40, -0.8, "40%", color=RED, fontsize=8)
    return _save(fig, path)


def plot_feature_availability_heatmap(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 80, max_rows: int = 300) -> Path:
    cols = list(feature_cols)[:max_features]
    if df is None or df.empty or not cols:
        return _empty(path, "No feature columns were detected for the availability matrix.", "Feature availability")
    show = df.loc[:, cols].head(max_rows)
    mat = (~show.isna()).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(12, max(5, min(12, 0.055 * len(show) + 3.5))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Feature availability matrix", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(f"Feature columns shown: {len(cols)} / {len(feature_cols)}", color=MUTED)
    ax.set_ylabel(f"Rows shown: {len(show)} / {len(df)}", color=MUTED)
    tick_step = max(1, len(cols) // 35)
    xticks = list(range(0, len(cols), tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([cols[i] for i in xticks], rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["missing", "available"])
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_distribution_grid(df: pd.DataFrame, feature_cols: Sequence[str], path: Path, max_features: int = 12) -> Path:
    cols = list(feature_cols)[:max_features]
    if df is None or df.empty or not cols:
        return _empty(path, "No numeric features were available for distribution plotting.", "Feature distributions")
    n = len(cols); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            ax.text(.5, .5, "all missing", ha="center", va="center", color=MUTED)
            ax.set_title(col, fontsize=9, color=NAVY)
            ax.axis("off")
            continue
        ax.hist(x, bins=min(24, max(6, int(np.sqrt(len(x))))), color=TEAL, alpha=0.88)
        med = x.median(); ax.axvline(med, color=NAVY, linestyle="--", linewidth=1)
        ax.set_title(col, fontsize=9, color=NAVY)
        ax.set_ylabel("count", color=MUTED)
        _style(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions", fontsize=15, fontweight="bold", color=NAVY, y=1.01)
    return _save(fig, path)


def plot_corr_heatmap(corr: pd.DataFrame, path: Path, title: str = "Spearman correlation heatmap") -> Path:
    if corr is None or corr.empty:
        return _empty(path, "Not enough numeric features were available for correlation analysis.", title)
    fig, ax = plt.subplots(figsize=(max(8, 0.26 * len(corr.columns)), max(7, 0.26 * len(corr.index))))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6, color=MUTED)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index, fontsize=6, color=MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman rho", color=MUTED)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    return _save(fig, path)


def plot_outlier_counts(outliers: pd.DataFrame, path: Path) -> Path:
    if outliers is None or outliers.empty or "feature" not in outliers.columns:
        return _empty(path, "No robust outliers were detected or no outlier table was available.", "Robust outliers")
    counts = outliers["feature"].value_counts().head(30).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.32 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color=GOLD)
    ax.set_xlabel("Outlier count", color=MUTED)
    _style(ax, "Robust outliers by feature")
    return _save(fig, path)


def plot_role_counts(role_summary: pd.DataFrame, path: Path) -> Path:
    if role_summary is None or role_summary.empty or "role" not in role_summary.columns or "n_columns" not in role_summary.columns:
        return _empty(path, "No column-role summary was available.", "Column roles")
    df = role_summary.copy()
    df["n_columns"] = pd.to_numeric(df["n_columns"], errors="coerce").fillna(0)
    df = df.sort_values("n_columns", ascending=True)
    palette = {"Feature": TEAL, "Target": GOLD, "Identifier": NAVY, "QC": "#6B5DD3", "Covariate": "#4E7AA8", "Ignore": MUTED}
    colors = [palette.get(str(r), MUTED) for r in df["role"]]
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.44 * len(df))))
    ax.barh(df["role"].astype(str), df["n_columns"].astype(float), color=colors)
    ax.set_xlabel("Number of columns", color=MUTED)
    _style(ax, "Column roles after mapping")
    for i, v in enumerate(df["n_columns"].astype(float)):
        ax.text(v + max(0.2, df["n_columns"].max() * 0.01), i, str(int(v)), va="center", fontsize=9, color=MUTED)
    return _save(fig, path)


def plot_group_counts(group_counts: pd.DataFrame, path: Path, group_variable: str = "task") -> Path:
    if group_counts is None or group_counts.empty or not {"group_variable", "level", "n_rows"}.issubset(group_counts.columns):
        return _empty(path, "No recognized grouping columns were detected. Add metadata with task, diagnosis, severity, subject, session, or modality to enable group-count plots.", "Group counts")
    df = group_counts[group_counts["group_variable"].astype(str).eq(group_variable)].copy()
    if df.empty:
        preferred = ["task", "diagnosis", "severity_bin", "modality", "subject_id", "session_id"]
        available = list(group_counts["group_variable"].astype(str).unique())
        chosen = next((p for p in preferred if p in available), available[0])
        df = group_counts[group_counts["group_variable"].astype(str).eq(chosen)].copy()
        group_variable = chosen
    df["n_rows"] = pd.to_numeric(df["n_rows"], errors="coerce").fillna(0)
    df = df.sort_values("n_rows", ascending=True).tail(30)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.36 * len(df))))
    ax.barh(df["level"].astype(str), df["n_rows"].astype(float), color=TEAL)
    ax.set_xlabel("Rows / recordings", color=MUTED)
    _style(ax, f"Rows by {group_variable}")
    return _save(fig, path)


def plot_feature_family_counts(family_overview: pd.DataFrame, path: Path) -> Path:
    if family_overview is None or family_overview.empty or "family_or_subsystem" not in family_overview.columns or "n_features" not in family_overview.columns:
        return _empty(path, "No feature registry or subsystem labels were supplied. The plot will become informative after a feature registry / computation policy file is loaded.", "Feature families")
    df = family_overview.copy()
    df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce").fillna(0)
    df = df.sort_values("n_features", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.38 * len(df))))
    ax.barh(df["family_or_subsystem"].astype(str), df["n_features"].astype(float), color=TEAL)
    ax.set_xlabel("Features", color=MUTED)
    _style(ax, "Feature count by family / subsystem")
    return _save(fig, path)
