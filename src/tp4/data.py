"""Shared loading and preprocessing of creadores.csv.

Used by both the linear (EDA/PCA/Oja) and kohonen packages.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Columns to always remove before any analysis
# ------------------------------------------------------------------

# These are identifier columns — they label the row but carry no
# information about the creator's content or audience.
_ID_COLS = ["creator_id", "name"]

# These columns are calculated directly from other columns that we
# already keep.  Including them would double-count that information
# and distort PCA (they would appear as "extra" variance even though
# they add nothing new).
#
#   price_ars              = price_usd * 1500
#   organic_posts_pct      = 100 - sponsored_posts_pct
#   total_engagement_per_1000 = likes + comments + saves + shares (per 1000)
#
# Deliberately KEPT although derived from other columns:
#   estimated_argentina_reach = average_reach * argentina_audience_pct / 100
#     It is a product, not a linear combination, so PCA and Kohonen cannot
#     reconstruct it from its inputs. It measures how many Argentine accounts a
#     post actually reaches, which matters for clients targeting Argentina.
#   topic_*_pct and content_*_pct each sum to 100 within their group, so one
#     column per group is redundant. All four are kept so that every topic
#     (one per client) and every content type stays directly interpretable.
_REDUNDANT_COLS = [
    "price_ars",
    "organic_posts_pct",
    "total_engagement_per_1000",
]


# ------------------------------------------------------------------
# Public functions
# ------------------------------------------------------------------

def load_creators(path: str) -> pd.DataFrame:
    """Load creadores.csv and return a clean numeric feature DataFrame.

    Drops identifier and redundant columns so the result contains only
    the features that will be used for modelling.

    Args:
        path: Path to creadores.csv, e.g. "../data/creadores.csv".

    Returns:
        DataFrame with shape (n_creators, n_features).
        All columns are numeric; the index is reset to 0, 1, 2, ...
    """
    df = pd.read_csv(path)

    cols_to_drop = _ID_COLS + _REDUNDANT_COLS
    df = df.drop(columns=cols_to_drop)

    return df.reset_index(drop=True)


def standardise(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Scale every feature to zero mean and unit variance (z-score).

    PCA is sensitive to the scale of each feature.  Without this step
    a variable like `followers` (values up to 775 000) would dominate
    over `likes_per_1000` (values up to 55) even if both carry the
    same amount of useful information.

    Args:
        df: Clean numeric DataFrame — the output of load_creators().

    Returns:
        X_scaled : numpy array of shape (n_creators, n_features).
                   Each column has mean ≈ 0 and std ≈ 1.
        scaler   : The fitted StandardScaler.
                   Keep this — you need it to convert PCA results back
                   to the original units (inverse_transform).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler
