"""
data_visualization.py
---------------------
Module 6: Visualization

Creates the charts for the report using Matplotlib and Seaborn.
Charts are saved as PNG files in outputs/visualizations/.
"""

import sys

import matplotlib
matplotlib.use("Agg")          # draw to files only (no pop-up windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from src.data_collection import (PROCESSED_DIR, REPORTS_DIR, VISUALIZATION_DIR,
                                     create_project_directories)
except ModuleNotFoundError:
    from data_collection import (PROCESSED_DIR, REPORTS_DIR, VISUALIZATION_DIR,
                                 create_project_directories)

sns.set_theme(style="whitegrid", context="notebook")
DPI = 150
BLUE, ORANGE, GREEN, PURPLE, RED = "#1f78b4", "#d95f02", "#1b9e77", "#7570b3", "#e31a1c"


def save_figure(fig, filename):
    """Save a figure to outputs/visualizations/ and close it."""
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUALIZATION_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Data quality charts
# ---------------------------------------------------------------------------
def plot_missing_values(raw_missing):
    """Missing values per column in the RAW data."""
    data = raw_missing.copy()
    data["label"] = data["table"] + "." + data["column"]
    data = data.sort_values("missing_percent")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["label"], data["missing_percent"], color=ORANGE)
    for y, (pct, count) in enumerate(zip(data["missing_percent"], data["missing_count"])):
        ax.text(pct + 0.5, y, f"{int(count):,}", va="center", fontsize=8)
    ax.set_xlim(0, data["missing_percent"].max() * 1.15)
    ax.set_xlabel("Missing values (% of rows in the table)")
    ax.set_ylabel("Table.column")
    ax.set_title("Missing Values by Column in the Raw Olist Data")
    return save_figure(fig, "01_missing_values_raw.png")


def plot_before_after(before_after):
    """Missing cells and duplicate rows before vs after cleaning."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (metric, title) in zip(axes, [("missing_cells", "Missing cells"),
                                          ("duplicate_rows", "Duplicate rows")]):
        x = np.arange(len(before_after))
        width = 0.4
        ax.bar(x - width / 2, before_after[f"{metric}_before"], width,
               label="Before cleaning", color=ORANGE)
        ax.bar(x + width / 2, before_after[f"{metric}_after"], width,
               label="After cleaning", color=GREEN)
        ax.set_xticks(x)
        ax.set_xticklabels(before_after["table"], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} per table")
        ax.legend()
    fig.suptitle("Data Quality Before vs After Cleaning", fontsize=14)
    return save_figure(fig, "02_before_after_cleaning.png")


def plot_quality_status(results):
    """Validation check results (PASS / INFO / FLAGGED) per check category."""
    counts = results.groupby(["category", "status"]).size().unstack(fill_value=0)
    for status in ("PASS", "INFO", "FLAGGED"):
        if status not in counts.columns:
            counts[status] = 0
    counts = counts[["PASS", "INFO", "FLAGGED"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    counts.plot(kind="barh", stacked=True, ax=ax, color=[GREEN, PURPLE, ORANGE])
    ax.set_xlabel("Number of validation checks")
    ax.set_ylabel("Check category")
    ax.set_title("Data Validation Results by Check Category")
    ax.legend(title="Status")
    return save_figure(fig, "03_validation_results.png")


# ---------------------------------------------------------------------------
# Logistics pattern charts
# ---------------------------------------------------------------------------
def plot_order_status(orders):
    counts = orders["order_status"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color=BLUE)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Order status")
    ax.set_ylabel("Number of orders (log scale)")
    ax.set_title("Orders by Status")
    return save_figure(fig, "04_order_status.png")


def plot_monthly_orders(orders):
    monthly = orders["purchase_year_month"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly.index, monthly.values, marker="o", color=BLUE)
    ax.set_xlabel("Purchase month")
    ax.set_ylabel("Number of orders")
    ax.set_title("Monthly Order Volume")
    ax.tick_params(axis="x", rotation=90)
    return save_figure(fig, "05_monthly_orders.png")


def plot_delivery_time(orders):
    x = orders["delivery_time_days"].dropna()
    cap = x.quantile(0.99)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x[x <= cap], bins=50, color=BLUE, ax=ax)
    ax.axvline(x.mean(), color=RED, linestyle="--", label=f"Mean: {x.mean():.1f} days")
    ax.axvline(x.median(), color=GREEN, linestyle="--", label=f"Median: {x.median():.1f} days")
    ax.set_xlabel("Delivery time (days from purchase to customer delivery)")
    ax.set_ylabel("Number of orders")
    ax.set_title(f"Distribution of Delivery Time (axis cut at 99th percentile, {cap:.0f} days)")
    ax.legend()
    return save_figure(fig, "06_delivery_time_distribution.png")


def plot_delivery_delay(orders):
    x = orders["delivery_delay_days"].dropna()
    low, high = x.quantile(0.01), x.quantile(0.99)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x[(x >= low) & (x <= high)], bins=60, color=PURPLE, ax=ax)
    ax.axvline(0, color=RED, linestyle="--", label="Estimated delivery date")
    ax.set_xlabel("Delivery delay (days; negative = earlier than estimated)")
    ax.set_ylabel("Number of orders")
    ax.set_title("Actual vs Estimated Delivery Date (1st to 99th percentile shown)")
    ax.legend()
    return save_figure(fig, "07_delivery_delay_distribution.png")


def plot_late_by_state(orders):
    late = orders["is_late"].astype(float)
    rate = late.groupby(orders["customer_state"]).mean().mul(100).sort_values()
    overall = late.mean() * 100
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(rate.index, rate.values, color=BLUE)
    ax.axvline(overall, color=RED, linestyle="--", label=f"Overall: {overall:.1f}%")
    ax.set_xlabel("Late deliveries (% of delivered orders with measurable dates)")
    ax.set_ylabel("Customer state")
    ax.set_title("Late Delivery Rate by Customer State")
    ax.legend()
    return save_figure(fig, "08_late_delivery_by_state.png")


def plot_freight_vs_distance(items):
    data = items.dropna(subset=["shipping_distance_km", "freight_value"])
    r = data["shipping_distance_km"].corr(data["freight_value"])
    sample = data.sample(min(20000, len(data)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["shipping_distance_km"], sample["freight_value"], s=4, alpha=0.25, color=BLUE)
    ax.set_xlabel("Shipping distance between seller and customer (km)")
    ax.set_ylabel("Freight value")
    ax.set_title(f"Freight vs Shipping Distance (random sample of {len(sample):,} items; "
                 f"correlation on all items = {r:.2f})")
    return save_figure(fig, "09_freight_vs_distance.png")


def plot_review_by_lateness(orders):
    data = orders.dropna(subset=["is_late", "avg_review_score"]).copy()
    data["delivery"] = np.where(data["is_late"].astype(float) == 1, "Late", "On time")
    means = data.groupby("delivery")["avg_review_score"].mean().reindex(["On time", "Late"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(means.index, means.values, color=[GREEN, ORANGE])
    for i, value in enumerate(means.values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, 5)
    ax.set_xlabel("Delivery outcome")
    ax.set_ylabel("Average review score (1-5)")
    ax.set_title("Review Score: On-time vs Late Deliveries")
    return save_figure(fig, "10_review_score_by_lateness.png")


def plot_correlation(orders):
    cols = ["delivery_time_days", "delivery_delay_days", "handling_time_days",
            "carrier_transit_days", "total_price", "total_freight", "freight_ratio",
            "total_weight_g", "avg_shipping_distance_km", "avg_review_score"]
    corr = orders[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation of Key Logistics Variables (order level)")
    return save_figure(fig, "11_correlation_heatmap.png")


def create_all_visualizations(raw_missing, quality_results, before_after, orders, items):
    """Create every chart. Returns the list of saved file paths."""
    print("\nCreating visualizations...")
    return [
        plot_missing_values(raw_missing),
        plot_before_after(before_after),
        plot_quality_status(quality_results),
        plot_order_status(orders),
        plot_monthly_orders(orders),
        plot_delivery_time(orders),
        plot_delivery_delay(orders),
        plot_late_by_state(orders),
        plot_freight_vs_distance(items),
        plot_review_by_lateness(orders),
        plot_correlation(orders),
    ]


if __name__ == "__main__":
    # Standalone test (after main.py has run once):  python src\data_visualization.py
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    create_project_directories()
    needed = [REPORTS_DIR / "raw_missing_values.csv", REPORTS_DIR / "data_quality_summary.csv",
              REPORTS_DIR / "before_after_summary.csv",
              PROCESSED_DIR / "analysis_orders.csv", PROCESSED_DIR / "analysis_order_items.csv"]
    absent = [p.name for p in needed if not p.is_file()]
    if absent:
        print("Missing input files:", absent, "\nRun main.py first.")
    else:
        create_all_visualizations(pd.read_csv(needed[0]), pd.read_csv(needed[1]),
                                  pd.read_csv(needed[2]), pd.read_csv(needed[3]),
                                  pd.read_csv(needed[4]))