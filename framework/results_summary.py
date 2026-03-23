# framework/results_summary.py
"""Cross-dataset comparison tables and plots."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
              "#D55E00", "#F0E442", "#000000"]


def cross_dataset_table(results: pd.DataFrame) -> pd.DataFrame:
    """Main results table: one row per (dataset, model) with mean metrics."""
    grouped = results.groupby(["dataset", "model"]).agg(
        equipment_type=("equipment_type", "first"),
        prior_quality=("prior_quality", "first"),
        n_trajectories=("unit_id", "nunique"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_detection_lead_time=("detection_lead_time", "mean"),
        detection_success_rate=("detection_success", "mean"),
    ).reset_index()
    return grouped.sort_values(["dataset", "mean_rmse"])


def prior_quality_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Group PID+regime results by prior_quality."""
    pid_results = results[results["model"].isin(["pid_adaptive", "pid_regime"])]
    grouped = pid_results.groupby("prior_quality").agg(
        mean_rmse=("rmse", "mean"), std_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"),
        mean_detection_lead_time=("detection_lead_time", "mean"),
        n_trajectories=("unit_id", "nunique"),
    ).reset_index()
    return grouped


def regime_benefit_table(results: pd.DataFrame) -> pd.DataFrame:
    """For each dataset: PID MAE vs PID+regime MAE improvement."""
    pid = results[results["model"] == "pid_adaptive"].groupby("dataset")["mae"].mean()
    regime = results[results["model"] == "pid_regime"].groupby("dataset")["mae"].mean()
    combined = pd.DataFrame({"pid_mae": pid, "pid_regime_mae": regime})
    combined["improvement_pct"] = ((combined["pid_mae"] - combined["pid_regime_mae"])
                                    / combined["pid_mae"] * 100)
    return combined.reset_index()


def plot_cross_dataset_comparison(results: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generate publication-quality cross-dataset comparison figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Grouped bar chart: RMSE by model, grouped by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = results.groupby(["dataset", "model"])["rmse"].mean().reset_index()
    pivot_wide = pivot.pivot(index="dataset", columns="model", values="rmse")
    pivot_wide.plot(kind="bar", ax=ax, color=CB_PALETTE[:len(pivot_wide.columns)])
    ax.set_ylabel("Mean RMSE (life units)")
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/rmse_by_model_dataset.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Prior quality scatter
    pid_regime = results[results["model"] == "pid_regime"].copy()
    if not pid_regime.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (dataset, group) in enumerate(pid_regime.groupby("dataset")):
            ax.scatter(group["prior_quality"], group["rmse"],
                      color=CB_PALETTE[i % len(CB_PALETTE)], label=dataset, alpha=0.7, s=60)
        ax.set_ylabel("PID + Regime RMSE (life units)")
        ax.set_xlabel("Prior Quality Tier")
        ax.legend(title="Dataset")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/prior_quality_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Regime benefit waterfall
    benefit = regime_benefit_table(results)
    if not benefit.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(benefit))
        width = 0.35
        ax.bar(x - width/2, benefit["pid_mae"], width, label="PID", color=CB_PALETTE[0])
        ax.bar(x + width/2, benefit["pid_regime_mae"], width, label="PID + Regime", color=CB_PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(benefit["dataset"])
        ax.set_ylabel("Mean Absolute Error (life units)")
        ax.legend()
        for i, row in benefit.iterrows():
            if pd.notna(row["improvement_pct"]):
                ax.annotate(f'{row["improvement_pct"]:.0f}% better',
                           xy=(x[i] if isinstance(i, int) else i, min(row["pid_mae"], row["pid_regime_mae"])),
                           xytext=(0, -20), textcoords="offset points",
                           ha="center", fontsize=9, color=CB_PALETTE[2])
        plt.tight_layout()
        fig.savefig(f"{output_dir}/regime_benefit.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Detection lead time comparison
    det = results.dropna(subset=["detection_lead_time"])
    if not det.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        det_summary = det.groupby(["dataset", "model"])["detection_lead_time"].mean().reset_index()
        det_summary["label"] = det_summary["dataset"] + " / " + det_summary["model"]
        det_summary = det_summary.sort_values("detection_lead_time")
        ax.barh(det_summary["label"], det_summary["detection_lead_time"], color=CB_PALETTE[0])
        ax.set_xlabel("Mean Detection Lead Time (steps before failure)")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/detection_lead_time.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved figures to {output_dir}/")
