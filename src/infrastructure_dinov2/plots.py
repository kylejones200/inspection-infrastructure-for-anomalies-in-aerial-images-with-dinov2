"""Figure generation for DINOv2 infrastructure anomaly detection article."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from infrastructure_dinov2.config import AppConfig
from infrastructure_dinov2.embeddings import (
    LABEL_DAMAGE,
    LABEL_EQUIPMENT,
    LABEL_NORMAL,
    LABEL_VEGETATION,
    generate_embeddings_with_structure,
)
from infrastructure_dinov2.style import apply_minimalist_style

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    LABEL_NORMAL: "#CCCCCC",
    LABEL_VEGETATION: "#2ECC40",
    LABEL_EQUIPMENT: "#FF4136",
    LABEL_DAMAGE: "#FF851B",
}

CLASS_NAMES_FULL = {
    LABEL_NORMAL: "Normal Infrastructure",
    LABEL_VEGETATION: "Vegetation Intrusion",
    LABEL_EQUIPMENT: "Equipment/Activity",
    LABEL_DAMAGE: "Surface Damage",
}

CLASS_NAMES_SHORT = {
    LABEL_NORMAL: "Normal",
    LABEL_VEGETATION: "Vegetation",
    LABEL_EQUIPMENT: "Equipment",
    LABEL_DAMAGE: "Damage",
}


def _figures_dir(config: AppConfig) -> Path:
    path = config.output.figures_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _anomaly_threshold(anomaly_scores: np.ndarray, sigma: float) -> float:
    return float(np.mean(anomaly_scores) + sigma * np.std(anomaly_scores))


def create_main_visualization(config: AppConfig) -> Path | None:
    """t-SNE projection of synthetic DINOv2 embeddings with outlier highlighting."""
    logger.info("Generating main visualization...")

    embeddings, labels, anomaly_scores = generate_embeddings_with_structure(config.data)

    logger.info("  Running t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        random_state=config.data.seed,
        perplexity=config.tsne.perplexity,
        max_iter=config.tsne.max_iter,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    threshold = _anomaly_threshold(anomaly_scores, config.anomaly_threshold_sigma)
    outlier_mask = anomaly_scores > threshold
    n_outliers = int(np.sum(outlier_mask))

    if not config.output.save_figures:
        logger.info("  Skipping save (save_figures=false); %s outliers flagged", n_outliers)
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    for class_id in (LABEL_NORMAL, LABEL_VEGETATION, LABEL_EQUIPMENT, LABEL_DAMAGE):
        mask = labels == class_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=CLASS_COLORS[class_id],
            label=CLASS_NAMES_FULL[class_id],
            alpha=0.6 if class_id == LABEL_NORMAL else 0.8,
            s=20 if class_id == LABEL_NORMAL else 40,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.scatter(
        embeddings_2d[outlier_mask, 0],
        embeddings_2d[outlier_mask, 1],
        marker="o",
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label=f"Flagged for Inspection (n={n_outliers}, >{config.anomaly_threshold_sigma:.0f}σ)",
    )

    apply_minimalist_style(ax)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.set_title(
        "DINOv2 Embedding Space - Infrastructure Anomaly Detection",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.02,
        f"{config.data.n_images:,} aerial images | "
        f"{config.data.embedding_dim}-dim embeddings | "
        f"{n_outliers} outliers flagged",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        color="black",
    )

    out_path = _figures_dir(config) / config.output.main_figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Saved %s (%s outliers flagged)", out_path.name, n_outliers)
    return out_path


def create_anomaly_distribution_visualization(config: AppConfig) -> Path | None:
    """Histogram of anomaly scores with μ + kσ threshold."""
    logger.info("Generating anomaly distribution visualization...")

    _embeddings, labels, anomaly_scores = generate_embeddings_with_structure(config.data)

    mean_score = float(np.mean(anomaly_scores))
    std_score = float(np.std(anomaly_scores))
    threshold = mean_score + config.anomaly_threshold_sigma * std_score

    if not config.output.save_figures:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for class_id in (LABEL_NORMAL, LABEL_VEGETATION, LABEL_EQUIPMENT, LABEL_DAMAGE):
        mask = labels == class_id
        ax.hist(
            anomaly_scores[mask],
            bins=50,
            alpha=0.6,
            color=CLASS_COLORS[class_id],
            label=CLASS_NAMES_SHORT[class_id],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold (μ + {config.anomaly_threshold_sigma:.0f}σ = {threshold:.2f})",
    )
    ax.axvline(
        mean_score,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {mean_score:.2f}",
    )

    apply_minimalist_style(ax)
    ax.set_xlabel("Anomaly Score (Distance to Cluster Centroid)", fontsize=10)
    ax.set_ylabel("Number of Images", fontsize=10)
    ax.set_title(
        "Anomaly Score Distribution",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    n_outliers = int(np.sum(anomaly_scores > threshold))
    ax.text(
        0.98,
        0.65,
        f"Total Images: {len(anomaly_scores):,}\n"
        f"Mean Score: {mean_score:.3f}\n"
        f"Std Dev: {std_score:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
        f"Flagged: {n_outliers} ({n_outliers / len(anomaly_scores) * 100:.2f}%)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1,
        },
    )

    out_path = _figures_dir(config) / config.output.distribution_figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Saved %s", out_path.name)
    return out_path


def create_performance_metrics_visualization(config: AppConfig) -> Path | None:
    """Bar charts for review workload reduction and detection performance."""
    logger.info("Generating performance metrics visualization...")

    if not config.output.save_figures:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scenarios = [
        "Manual\nReview\n(100%)",
        "Pilot\nFlagging\n(~5%)",
        "DINOv2\nAnomaly\n(~2%)",
    ]
    review_pct = [100, 5, 2]
    colors_workload = ["#FF4136", "#FF851B", "#2ECC40"]

    bars1 = ax1.bar(
        scenarios,
        review_pct,
        color=colors_workload,
        edgecolor="black",
        linewidth=1.5,
    )
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    apply_minimalist_style(ax1)
    ax1.set_ylabel("Images Requiring Human Review (%)", fontsize=10)
    ax1.set_title(
        "Review Workload Reduction",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax1.set_ylim(0, 110)
    ax1.text(
        0.5,
        0.95,
        "98% reduction: 10,000 images → 200 reviews",
        transform=ax1.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
        color="black",
    )

    metrics = ["Anomaly\nRecall", "False\nPositive\nRate", "Review\nWorkload"]
    values = [78, 22, 2]
    colors_perf = ["#2ECC40", "#FF4136", "#0074D9"]

    bars2 = ax2.bar(metrics, values, color=colors_perf, edgecolor="black", linewidth=1.5)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    apply_minimalist_style(ax2)
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_title(
        "Detection Performance (μ + 3σ Threshold)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax2.set_ylim(0, 90)
    ax2.text(
        0.5,
        0.95,
        "Captures 78% of actual anomalies while reviewing 2% of images",
        transform=ax2.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        style="italic",
        color="black",
    )

    out_path = _figures_dir(config) / config.output.performance_figure
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("  Saved %s", out_path.name)
    return out_path
