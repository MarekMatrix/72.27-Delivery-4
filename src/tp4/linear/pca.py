"""Principal component analysis: loadings, explained variance, biplot.

PCA is implemented here with numpy, following the procedure from the
course slides (5 - PCA, slide 32):

    1. Build X with the variables as columns
    2. Standardise the variables            (done by standardise() in data.py)
    3. Compute the correlation matrix Sx
    4. Compute its eigenvalues and eigenvectors
    5. Sort the eigenvalues from largest to smallest
    6. Build V with the eigenvectors of the largest eigenvalues
    7. Compute the new variables Y as linear combinations of the originals
"""

from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PCA:
    """Result of fit_pca(): everything needed to interpret and apply the PCA.

    Attributes:
        components_:               Loadings, shape (n_components, n_features).
                                   Row i is the eigenvector of PC(i+1).
        explained_variance_:       Eigenvalues = variance of each component.
        explained_variance_ratio_: Share of the total variance per component.
        n_components_:             Number of components (= number of features).
        mean_:                     Feature means of the data PCA was fitted on.
    """

    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    n_components_: int
    mean_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Step 7: project observations onto the components (the scores Y = X·V)."""
        return (X - self.mean_) @ self.components_.T


def fit_pca(X_scaled: np.ndarray) -> PCA:
    """Run PCA keeping all components (steps 3–6 of the course procedure).

    We keep all components here so we can inspect the full scree plot
    and choose the right number of dimensions later.

    Args:
        X_scaled: Standardised feature matrix of shape (n_creators, n_features).
                  Must already be z-score scaled (use standardise() from data.py).

    Returns:
        PCA object.  Key attributes you will use:
            pca.explained_variance_ratio_  — fraction of variance per component
            pca.components_                — the loading vectors
    """
    # Step 3: correlation matrix (variables are columns, hence rowvar=False).
    # On standardised data this is the same matrix PCA would get from the
    # covariance — see slide 27, "covarianza vs correlación".
    corr = np.corrcoef(X_scaled, rowvar=False)

    # Step 4: eigenvalues and eigenvectors.  eigh is for symmetric matrices;
    # it returns real values and the eigenvectors as COLUMNS.
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Step 5: eigh sorts ascending, so reverse to largest-first.
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Numerical noise can make the exactly-zero eigenvalues (from columns that
    # sum to 100) come out as tiny negatives; a variance cannot be negative.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # The sign of an eigenvector is arbitrary (slide 46, "autovector opuesto").
    # Fix it so the largest loading of each component is positive; otherwise
    # the plots could flip from one run or machine to the next.
    largest = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[largest, np.arange(eigenvectors.shape[1])])
    eigenvectors = eigenvectors * signs

    # Step 6: V, one eigenvector per row (the layout the rest of this module uses)
    return PCA(
        components_=eigenvectors.T,
        explained_variance_=eigenvalues,
        explained_variance_ratio_=eigenvalues / eigenvalues.sum(),
        n_components_=len(eigenvalues),
        mean_=X_scaled.mean(axis=0),
    )


def loadings_df(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """Build a readable DataFrame of PCA loadings.

    A loading is how much a given original feature contributes to a
    principal component.  Large positive value → that feature pulls
    the component score up.  Large negative → pulls it down.

    Args:
        pca:           Fitted PCA object (output of fit_pca).
        feature_names: The original column names, in the same order they
                       were passed to fit_pca.  Use df.columns.tolist().

    Returns:
        DataFrame of shape (n_features, n_components).
        Rows = original features, columns = "PC1", "PC2", ...
    """
    # pca.components_ has shape (n_components, n_features)
    # We transpose so that rows = features, columns = components
    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )


def scree_plot(pca: PCA) -> None:
    """Plot individual and cumulative explained variance.

    Use this to decide how many principal components to keep.
    Look for the "elbow" — the point where adding another component
    gives diminishing returns.

    Vertical dotted lines mark the fewest components needed to
    explain 70 % and 80 % of total variance.

    Args:
        pca: Fitted PCA object (output of fit_pca).
    """
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Number of components needed to pass the 70 % and 80 % thresholds
    k70 = int(np.searchsorted(cumulative, 0.70)) + 1
    k80 = int(np.searchsorted(cumulative, 0.80)) + 1

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- Left axis: individual explained variance (bar chart) ---
    ax1.bar(
        range(1, len(explained) + 1),
        explained * 100,
        color="steelblue",
        alpha=0.7,
        label="Individual",
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # --- Right axis: cumulative explained variance (line chart) ---
    ax2 = ax1.twinx()
    ax2.plot(
        range(1, len(cumulative) + 1),
        cumulative * 100,
        "o-",
        color="tomato",
        label="Cumulative",
    )
    ax2.set_ylabel("Cumulative Explained Variance (%)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_ylim(0, 105)

    # Reference lines at 70 % and 80 %
    ax2.axhline(70, color="tomato", linestyle="--", alpha=0.5)
    ax2.axhline(80, color="darkred", linestyle="--", alpha=0.5)
    ax2.axvline(k70, color="tomato", linestyle=":", alpha=0.7)
    ax2.axvline(k80, color="darkred", linestyle=":", alpha=0.7)
    ax2.text(k70 + 0.2, 5,  f"{k70} PCs → 70 %", color="tomato",   fontsize=9)
    ax2.text(k80 + 0.2, 12, f"{k80} PCs → 80 %", color="darkred",  fontsize=9)

    plt.title("Scree Plot — Explained Variance per Principal Component")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    plt.show()

    print(f"Components needed for 70 % variance: {k70}")
    print(f"Components needed for 80 % variance: {k80}")


def plot_loadings(loadings: pd.DataFrame, components: list[str] = None) -> None:
    """Plot a horizontal bar chart of loadings for selected components.

    Bars are sorted by value so the most influential features are easy
    to read.  Blue = positive contribution, red = negative contribution.

    Args:
        loadings:   Output of loadings_df().
        components: Which PCs to plot.  Defaults to ['PC1', 'PC2'].
    """
    if components is None:
        components = ["PC1", "PC2"]

    fig, axes = plt.subplots(1, len(components), figsize=(8 * len(components), 7))

    # Make axes iterable even when there is only one component
    if len(components) == 1:
        axes = [axes]

    for ax, pc in zip(axes, components):
        sorted_vals = loadings[pc].sort_values()
        colors = ["tomato" if v < 0 else "steelblue" for v in sorted_vals]
        sorted_vals.plot(kind="barh", ax=ax, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{pc} Loadings", fontsize=12)
        ax.set_xlabel("Loading value")
        ax.tick_params(labelsize=8)

    plt.suptitle("Feature Loadings", fontsize=13)
    plt.tight_layout()
    plt.show()


def biplot(
    pca: PCA,
    X_scaled: np.ndarray,
    feature_names: list[str],
    labels: list[str] = None,
    arrow_scale: float = 3.5,
) -> None:
    """Scatter all creators in PC1–PC2 space and overlay loading arrows.

    Points = individual creators.
    Arrows = original features.  An arrow pointing right means that
    feature has a high positive loading on PC1 (creators with lots of
    that feature sit to the right).

    This plot is the reference for Exercise 1.2 (Kohonen) — we will
    compare whether the Kohonen grid groups creators the same way the
    biplot does.

    Args:
        pca:           Fitted PCA object.
        X_scaled:      Standardised feature matrix.
        feature_names: Original column names.
        labels:        Optional creator names for annotating each point.
        arrow_scale:   Multiply loading vectors by this number so the
                       arrows are visible relative to the point cloud.
    """
    # Project every creator onto the first two components
    scores = pca.transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    # --- Points: one dot per creator ---
    ax.scatter(scores[:, 0], scores[:, 1], s=50, alpha=0.7, color="steelblue", zorder=3)

    # Optionally label each point with the creator's name
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (scores[i, 0], scores[i, 1]), fontsize=5, alpha=0.6)

    # --- Arrows: one arrow per original feature ---
    for feature in feature_names:
        x = loadings.loc[feature, "PC1"] * arrow_scale
        y = loadings.loc[feature, "PC2"] * arrow_scale
        # Draw arrow from origin to (x, y)
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="tomato", lw=1.2),
        )
        # Label the arrow tip with the feature name
        ax.text(x * 1.08, y * 1.08, feature, fontsize=6.5, color="tomato")

    # Reference lines at the origin
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({explained[0] * 100:.1f} % variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({explained[1] * 100:.1f} % variance)", fontsize=11)
    ax.set_title("Biplot — Creators in PC1–PC2 Space", fontsize=13)
    plt.tight_layout()
    plt.show()


def bootstrap_stability(df: pd.DataFrame, n_components: int = 8,
                        n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """How stable is each component when the creators are resampled?

    Draws n_boot bootstrap samples (creators drawn with replacement), refits
    PCA on each and measures |cos| between every new component and the
    original one: 1 = same direction, 0 = unrelated.  With only 50 creators,
    components whose eigenvalues lie close together can swap or mix.

    Args:
        df:           Numeric feature DataFrame (output of load_creators).
        n_components: How many leading components to check.
        n_boot:       Number of bootstrap samples.
        seed:         Seed for the random resampling, so the result is reproducible.

    Returns:
        DataFrame indexed by "PC1", "PC2", ... with the eigenvalue and the
        median and 10th percentile of the similarity.
    """
    def standardised_pca(values: np.ndarray) -> PCA:
        return fit_pca((values - values.mean(axis=0)) / values.std(axis=0))

    values = df.to_numpy(dtype=float)
    reference = standardised_pca(values)
    rng = np.random.default_rng(seed)

    similarities = []
    while len(similarities) < n_boot:
        sample = values[rng.integers(0, len(values), len(values))]
        if np.any(sample.std(axis=0) == 0):   # a constant column cannot be standardised
            continue
        refit = standardised_pca(sample)
        similarities.append(np.abs(np.sum(reference.components_[:n_components]
                                          * refit.components_[:n_components], axis=1)))
    similarities = np.array(similarities)

    return pd.DataFrame(
        {
            "eigenvalue": reference.explained_variance_[:n_components],
            "median_similarity": np.median(similarities, axis=0),
            "p10_similarity": np.percentile(similarities, 10, axis=0),
        },
        index=[f"PC{i + 1}" for i in range(n_components)],
    )
