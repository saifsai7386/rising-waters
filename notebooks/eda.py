"""
eda.py — Data Visualization & Analysis
---------------------------------------
Univariate + multivariate analysis, distribution plots, box plots,
heat maps, and descriptive statistics for the rainfall/flood dataset.

Run: python3 notebooks/eda.py
Outputs PNG charts into notebooks/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "rainfall_data.csv")
PLOT_DIR = os.path.join(_BASE_DIR, "notebooks", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    return df


def descriptive_stats(df):
    print("\n=== Descriptive Statistics ===")
    print(df.describe(include="all").T)
    df.describe().to_csv(f"{PLOT_DIR}/descriptive_stats.csv")


def univariate_analysis(df):
    # Distribution of ANNUAL rainfall
    plt.figure(figsize=(8, 5))
    sns.histplot(df["ANNUAL"].dropna(), kde=True, color="steelblue")
    plt.title("Distribution of Annual Rainfall")
    plt.xlabel("Annual Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/univariate_annual_rainfall_dist.png", dpi=120)
    plt.close()

    # Countplot of target class
    plt.figure(figsize=(5, 5))
    sns.countplot(x="FLOODS", data=df, palette="Set2")
    plt.title("Flood vs No-Flood Class Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/univariate_flood_class_counts.png", dpi=120)
    plt.close()

    # Subdivision counts
    plt.figure(figsize=(8, 5))
    sns.countplot(y="SUBDIVISION", data=df, order=df["SUBDIVISION"].value_counts().index)
    plt.title("Records per Subdivision")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/univariate_subdivision_counts.png", dpi=120)
    plt.close()


def distribution_and_box_plots(df):
    # Monthly rainfall distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, month in zip(axes.flatten(), MONTHS):
        sns.histplot(df[month].dropna(), kde=True, ax=ax, color="teal")
        ax.set_title(month)
    plt.suptitle("Monthly Rainfall Distributions")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/distribution_monthly_rainfall.png", dpi=120)
    plt.close()

    # Box plots for outlier detection
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=df[MONTHS])
    plt.title("Box Plots of Monthly Rainfall (Outlier Detection)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/boxplot_monthly_rainfall.png", dpi=120)
    plt.close()

    # Annual rainfall by flood class
    plt.figure(figsize=(6, 5))
    sns.boxplot(x="FLOODS", y="ANNUAL", data=df, palette="Set3")
    plt.title("Annual Rainfall by Flood Outcome")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/boxplot_annual_by_flood.png", dpi=120)
    plt.close()


def multivariate_analysis(df):
    # Correlation heatmap
    numeric_df = df[MONTHS + ["ANNUAL"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Rainfall Features")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/heatmap_correlation.png", dpi=120)
    plt.close()

    # Pairplot of monsoon months colored by flood
    monsoon_cols = ["JUN", "JUL", "AUG", "SEP", "ANNUAL", "FLOODS"]
    sample = df[monsoon_cols].dropna()
    pp = sns.pairplot(sample, hue="FLOODS", diag_kind="kde", palette="husl")
    pp.fig.suptitle("Pairwise Relationships: Monsoon Months vs Flood", y=1.02)
    pp.savefig(f"{PLOT_DIR}/pairplot_monsoon_vs_flood.png", dpi=120)
    plt.close()

    # Average monthly rainfall trend by flood outcome
    means = df.groupby("FLOODS")[MONTHS].mean().T
    plt.figure(figsize=(10, 6))
    means.plot(marker="o", ax=plt.gca())
    plt.title("Average Monthly Rainfall: Flood vs No Flood Years")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/multivariate_monthly_trend_by_flood.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    descriptive_stats(df)
    univariate_analysis(df)
    distribution_and_box_plots(df)
    multivariate_analysis(df)
    print(f"\nAll plots saved to: {PLOT_DIR}")
