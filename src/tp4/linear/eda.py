"""Exploratory data analysis of the creators dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(df: pd.DataFrame, ncols: int = 4) -> None:
    """Plot a histogram for every feature column in df.

    Useful for spotting skewed distributions and outliers before
    running PCA.

    Args:
        df:    Clean numeric DataFrame (output of load_creators).
        ncols: How many histograms to show per row.
    """
    # Work out how many rows the grid needs
    nrows = int(np.ceil(len(df.columns) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()   # easier to loop over a 1-D list

    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=20, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=7)

    # Hide any empty subplot slots at the end
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame, title: str, xlabel: str) -> None:
    """Plot one horizontal boxplot per feature, all on the same axis.

    Drawn once on the raw data and once on the standardised data, this
    shows why PCA needs standardisation (course slides 5 - PCA, 43–44):
    on the raw scale a few large-valued features hide all the others.

    Box = middle 50 % of creators, line = median, whiskers = 1.5 × IQR,
    dots = outliers beyond the whiskers.

    Args:
        df:     Numeric DataFrame, one column per feature.
        title:  Plot title.
        xlabel: Label for the value axis (include the units).
    """
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(df.columns) + 1.5))
    ax.boxplot(
        [df[col] for col in df.columns],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", edgecolor="steelblue"),
        medianprops=dict(color="tomato", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="tomato", markeredgecolor="none"),
    )
    # Set the labels here rather than in boxplot(): the keyword was renamed in matplotlib 3.9
    ax.set_yticks(range(1, len(df.columns) + 1), df.columns)
    ax.invert_yaxis()   # first column at the top, same order as the DataFrame
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot a heatmap of the Pearson correlation matrix.

    Only the lower triangle is shown — the upper triangle is
    identical (correlation is symmetric) and would just be visual noise.

    Values close to +1 (dark red)  → strong positive linear relationship.
    Values close to -1 (dark blue) → strong negative linear relationship.
    Values close to  0 (white)     → no linear relationship.

    Args:
        df: Clean numeric DataFrame (output of load_creators).
    """
    corr = df.corr()

    # True in the upper triangle → seaborn will hide those cells
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        annot_kws={"size": 6},
        ax=ax,
    )
    ax.set_title("Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.show()


def high_correlation_pairs(
    df: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """Return every pair of features whose absolute correlation exceeds threshold.

    Strongly correlated pairs are candidates for dropping one of the two
    columns before running PCA, since they carry near-duplicate information.

    Args:
        df:        Clean numeric DataFrame (output of load_creators).
        threshold: Absolute correlation cut-off.  Default is 0.8.

    Returns:
        DataFrame with columns ['feature_1', 'feature_2', 'abs_corr'],
        sorted from highest to lowest correlation.
    """
    corr = df.corr().abs()

    # Keep only the lower triangle so each pair appears exactly once
    lower_triangle = np.tril(np.ones(corr.shape), k=-1).astype(bool)
    corr_lower = corr.where(lower_triangle)

    # Convert to a long-format table
    pairs = (
        corr_lower
        .stack()
        .reset_index()
    )
    pairs.columns = ["feature_1", "feature_2", "abs_corr"]

    return (
        pairs[pairs["abs_corr"] > threshold]
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
