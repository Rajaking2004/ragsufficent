"""
Visualization utilities for Prob-RAG experiments.

Creates publication-quality figures for:
- Accuracy vs Coverage curves
- Routing distribution
- Calibration curves
- Threshold sensitivity analysis
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)


# Style configuration for publication-quality figures
STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

# Color palette
COLORS = {
    "prob_rag": "#2ecc71",      # Green
    "baseline": "#e74c3c",      # Red
    "red": "#e74c3c",           # Abstention
    "yellow": "#f39c12",        # Hedging
    "green": "#2ecc71",         # Standard
    "primary": "#3498db",       # Blue
    "secondary": "#9b59b6",     # Purple
}


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available")
        return
    
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_style("whitegrid")


def plot_accuracy_vs_coverage(
    prob_rag_curve: Dict[float, float],
    baseline_curve: Dict[float, float],
    title: str = "Selective Accuracy vs Coverage",
    save_path: Optional[str] = None
) -> None:
    """
    Plot accuracy vs coverage curves comparing Prob-RAG and baseline.
    
    This is the key figure showing improvement in selective accuracy.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib required for plotting")
        return
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prob-RAG curve
    coverages_pr = sorted(prob_rag_curve.keys())
    accuracies_pr = [prob_rag_curve[c] for c in coverages_pr]
    ax.plot(coverages_pr, accuracies_pr, 'o-', color=COLORS["prob_rag"], 
            label="Prob-RAG (Ours)", linewidth=2.5, markersize=8)
    
    # Baseline curve
    coverages_bl = sorted(baseline_curve.keys())
    accuracies_bl = [baseline_curve[c] for c in coverages_bl]
    ax.plot(coverages_bl, accuracies_bl, 's--', color=COLORS["baseline"],
            label="Binary Baseline", linewidth=2, markersize=7)
    
    # Fill between to highlight improvement
    if len(coverages_pr) == len(coverages_bl):
        ax.fill_between(
            coverages_pr, accuracies_bl, accuracies_pr,
            alpha=0.2, color=COLORS["prob_rag"],
            where=[a > b for a, b in zip(accuracies_pr, accuracies_bl)]
        )
    
    ax.set_xlabel("Coverage (Fraction of Questions Answered)")
    ax.set_ylabel("Selective Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for improvement
    if prob_rag_curve and baseline_curve:
        mid_coverage = 0.8
        if mid_coverage in prob_rag_curve and mid_coverage in baseline_curve:
            improvement = (prob_rag_curve[mid_coverage] - baseline_curve[mid_coverage]) * 100
            ax.annotate(
                f"+{improvement:.1f}pp",
                xy=(mid_coverage, (prob_rag_curve[mid_coverage] + baseline_curve[mid_coverage]) / 2),
                fontsize=12, fontweight='bold', color=COLORS["prob_rag"]
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.close()


def plot_routing_distribution(
    routing_stats: Dict[str, Any],
    title: str = "Traffic Light Routing Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of routing decisions.
    
    Shows breakdown of Red/Yellow/Green decisions.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    ax1 = axes[0]
    sizes = [
        routing_stats.get("red_ratio", 0),
        routing_stats.get("yellow_ratio", 0),
        routing_stats.get("green_ratio", 0)
    ]
    labels = ["🔴 Abstention", "🟡 Hedging", "🟢 Standard"]
    colors = [COLORS["red"], COLORS["yellow"], COLORS["green"]]
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        explode=(0.02, 0.02, 0.02)
    )
    ax1.set_title("Routing Decision Distribution")
    
    # Bar chart with accuracy per state
    ax2 = axes[1]
    states = ["Abstention\n(Red)", "Hedging\n(Yellow)", "Standard\n(Green)"]
    counts = [
        routing_stats.get("red_count", 0),
        routing_stats.get("yellow_count", 0),
        routing_stats.get("green_count", 0)
    ]
    
    bars = ax2.bar(states, counts, color=[COLORS["red"], COLORS["yellow"], COLORS["green"]])
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Samples per Routing State")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.close()


def plot_score_distribution(
    scores: List[float],
    tau_low: float,
    tau_high: float,
    title: str = "Sufficiency Score Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot histogram of sufficiency scores with threshold markers.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    
    # Color bars by region
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        bin_right = bins[i + 1]
        bin_center = (bin_left + bin_right) / 2
        
        if bin_center < tau_low:
            patch.set_facecolor(COLORS["red"])
        elif bin_center > tau_high:
            patch.set_facecolor(COLORS["green"])
        else:
            patch.set_facecolor(COLORS["yellow"])
    
    # Threshold lines
    ax.axvline(tau_low, color='darkred', linestyle='--', linewidth=2,
               label=f'τ_low = {tau_low}')
    ax.axvline(tau_high, color='darkgreen', linestyle='--', linewidth=2,
               label=f'τ_high = {tau_high}')
    
    # Shade regions
    ax.axvspan(0, tau_low, alpha=0.1, color=COLORS["red"])
    ax.axvspan(tau_low, tau_high, alpha=0.1, color=COLORS["yellow"])
    ax.axvspan(tau_high, 1, alpha=0.1, color=COLORS["green"])
    
    ax.set_xlabel("Sufficiency Score (S)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_calibration_curve(
    scores: List[float],
    labels: List[int],
    title: str = "Calibration Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve showing predicted vs actual accuracy.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute calibration curve
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    
    for i in range(n_bins):
        mask = (scores_arr >= bin_boundaries[i]) & (scores_arr < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(labels_arr[mask].mean())
            bin_counts.append(mask.sum())
    
    # Plot calibration curve
    ax.plot(bin_centers, bin_accuracies, 'o-', color=COLORS["primary"],
            label="Prob-RAG", linewidth=2, markersize=10)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration", alpha=0.7)
    
    # Fill gap
    ax.fill_between(bin_centers, bin_centers, bin_accuracies,
                   alpha=0.2, color=COLORS["primary"])
    
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Fraction of Correct Answers")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_threshold_sensitivity(
    sweep_results: List[Dict[str, Any]],
    metric: str = "selective_accuracy",
    title: str = "Threshold Sensitivity Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot how metrics change with different threshold configurations.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    configs = []
    values = []
    
    for result in sweep_results:
        tau_low = result["tau_low"]
        tau_high = result["tau_high"]
        value = result["comparison"]["prob_rag"][metric]
        
        configs.append(f"({tau_low}, {tau_high})")
        values.append(value)
    
    # Bar plot
    colors = [COLORS["primary"] if v == max(values) else COLORS["secondary"] 
              for v in values]
    bars = ax.bar(configs, values, color=colors)
    
    ax.set_xlabel("Thresholds (τ_low, τ_high)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    
    # Highlight best
    best_idx = values.index(max(values))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_multi_dataset_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["accuracy", "selective_accuracy", "coverage"],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison across multiple datasets.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    datasets = [d for d in results.keys() if "error" not in results[d]]
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        prob_rag_vals = [results[d]["prob_rag"][metric] for d in datasets]
        baseline_vals = [results[d]["baseline"][metric] for d in datasets]
        
        ax.bar(x - width/2, prob_rag_vals, width, label="Prob-RAG", color=COLORS["prob_rag"])
        ax.bar(x + width/2, baseline_vals, width, label="Baseline", color=COLORS["baseline"])
        
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.suptitle("Multi-Dataset Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def create_experiment_report(
    experiment_results: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Create a full visual report for an experiment.
    
    Returns path to output directory with all figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all figures
    if "selective_accuracy_curve" in experiment_results.get("prob_rag", {}):
        plot_accuracy_vs_coverage(
            experiment_results["prob_rag"]["selective_accuracy_curve"],
            experiment_results.get("baseline", {}).get("selective_accuracy_curve", {}),
            save_path=os.path.join(output_dir, "accuracy_vs_coverage.png")
        )
    
    if "prob_rag_routing" in experiment_results:
        plot_routing_distribution(
            experiment_results["prob_rag_routing"],
            save_path=os.path.join(output_dir, "routing_distribution.png")
        )
    
    logger.info(f"Report generated in {output_dir}")
    return output_dir


# For command-line usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results_json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        results = json.load(f)
    
    output_dir = os.path.dirname(sys.argv[1]) or "."
    create_experiment_report(results, output_dir)
