"""Plotting helpers for model performance and drift visualization."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns

from .evaluation import EvaluationResult

sns.set(style="whitegrid")


def plot_drift(report: Dict[int, EvaluationResult], output_path: Path | None = None) -> plt.Axes:
    """Plot per-season RMSE to visualize drift trends."""
    seasons = sorted(report.keys())
    rmses = [report[season].rmse for season in seasons]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seasons, rmses, marker="o")
    ax.set_xlabel("Season")
    ax.set_ylabel("RMSE")
    ax.set_title("Model performance over seasons (concept drift)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return ax
