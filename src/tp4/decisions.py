"""Campaign decisions (exercise 1.4): turn the SOM, PCA and Oja results into creator selections.

Everything here works on the 28 variables from load_creators() and the scaling from
standardise(), so the numbers line up with exercises 1.1 – 1.3.

The selection pipeline is:

    1. Label every SOM neuron with the topic its weights lean towards (in original units).
    2. A client's candidates are the creators whose BMU sits in that topic's region.
    3. Rank the candidates by one campaign strategy (reach, discovery or depth).
    4. Compare the result with the agency's current rule: top creators by followers.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Clients and strategies
# ------------------------------------------------------------------

# One topic column per client of the agency.
CLIENT_TOPICS = {
    "dog_shelter": "topic_dogs_pct",
    "football_apparel": "topic_football_pct",
    "beauty": "topic_makeup_pct",
    "art_supplies": "topic_art_pct",
}

# The three strategies named in the statement for the football question
# ("alcance masivo, descubrimiento o profundidad educativa"). A strategy score is the
# mean z-score of its variables, so every variable weighs the same.
#
#   reach     : how many accounts a post reaches, in total and in Argentina.
#   discovery : how much of that reach comes from people who do not follow the
#               creator yet, and how much the content travels (shares, trends).
#   depth     : educational content that people keep and act on.
#
# followers is deliberately left out of every strategy: it is the baseline we compare against.
STRATEGIES = {
    "reach": ["average_reach", "average_impressions", "estimated_argentina_reach"],
    "discovery": ["non_follower_reach_pct", "shares_per_1000", "trend_posts_pct"],
    "depth": [
        "content_tutorial_pct",
        "saves_per_1000",
        "comments_per_1000",
        "profile_visits_per_1000",
        "average_watch_time_seconds",
    ],
}


# ------------------------------------------------------------------
# Loading the tuned SOM
# ------------------------------------------------------------------

def load_som(path: str | Path) -> dict:
    """Load the arrays saved by src/tp4/kohonen/run_kohonen_tuning.py.

    Args:
        path: Path to som_tuned.npz, e.g. "results/kohonen_models/som_tuned.npz".

    Returns:
        Dict with weights (width, height, n_features), bmu_indices (n_creators, 2),
        bmu_counts, u_matrix, feature_names, scaler_mean and scaler_scale.
    """
    with np.load(path) as saved:
        som = {key: saved[key] for key in saved.files}
    som["feature_names"] = som["feature_names"].tolist()
    return som


# ------------------------------------------------------------------
# SOM regions
# ------------------------------------------------------------------

def neuron_profiles(som: dict) -> pd.DataFrame:
    """Weights of every neuron in original units (percentages, USD, ...).

    The SOM is trained on z-scores, so the weights are converted back with the
    saved scaler before they are compared with each other.

    Returns:
        DataFrame indexed by (i, j) with one column per feature.
    """
    weights = som["weights"]
    width, height, n_features = weights.shape
    original = weights.reshape(-1, n_features) * som["scaler_scale"] + som["scaler_mean"]
    index = pd.MultiIndex.from_product([range(width), range(height)], names=["i", "j"])
    return pd.DataFrame(original, index=index, columns=som["feature_names"])


def neuron_topics(profiles: pd.DataFrame) -> pd.Series:
    """Label each neuron with the topic of largest weight (e.g. "topic_dogs_pct").

    Uses original units because the four topic columns are percentages of the
    same posts, so they are directly comparable before scaling.
    """
    return profiles[list(CLIENT_TOPICS.values())].idxmax(axis=1).rename("region")


def creator_table(raw: pd.DataFrame, som: dict, X_scaled: np.ndarray) -> pd.DataFrame:
    """One row per creator: name, main topic, BMU, SOM region and strategy scores.

    Args:
        raw: The unmodified creadores.csv (same row order as X_scaled).
        som: Output of load_som().
        X_scaled: Standardised features from standardise(load_creators(...)).
    """
    topics = list(CLIENT_TOPICS.values())
    regions = neuron_topics(neuron_profiles(som))

    table = raw[["name"]].copy()
    table["main_topic"] = raw[topics].idxmax(axis=1)
    table["i"], table["j"] = som["bmu_indices"][:, 0], som["bmu_indices"][:, 1]
    table["region"] = [regions.loc[(i, j)] for i, j in zip(table["i"], table["j"])]

    scores = strategy_scores(X_scaled, som["feature_names"])
    return pd.concat([table, scores], axis=1)


# ------------------------------------------------------------------
# Strategies and selections
# ------------------------------------------------------------------

def strategy_scores(X_scaled: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Score every creator on each strategy as the mean z-score of its variables."""
    column = {name: k for k, name in enumerate(feature_names)}
    return pd.DataFrame({
        strategy: X_scaled[:, [column[f] for f in features]].mean(axis=1)
        for strategy, features in STRATEGIES.items()
    })


def select_by_strategy(creators: pd.DataFrame, client: str, strategy: str, k: int,
                       on_topic_only: bool = True) -> pd.Index:
    """Top-k creators of the client's SOM region, ranked by one strategy score.

    With on_topic_only, creators the SOM placed in the region for their size or
    format but whose own main topic is a different one are left out (a large dogs
    creator can sit in the football region next to other large viral accounts).
    If fewer than k candidates remain, all of them are returned.
    """
    topic = CLIENT_TOPICS[client]
    candidates = creators[creators["region"] == topic]
    if on_topic_only:
        candidates = candidates[candidates["main_topic"] == topic]
    return candidates.nlargest(k, strategy).index


def select_mix(creators: pd.DataFrame, raw: pd.DataFrame, client: str,
               n_discovery: int = 3, n_depth: int = 2) -> pd.Index:
    """Discovery creators at mid-size prices plus a few depth creators.

    The discovery picks are the best discovery scores among the client's
    candidates priced at or below the candidates' median price, so the budget is
    not spent on the largest accounts. The depth picks are the best depth scores
    among the remaining candidates.
    """
    topic = CLIENT_TOPICS[client]
    candidates = creators[(creators["region"] == topic) & (creators["main_topic"] == topic)]
    prices = raw.loc[candidates.index, "price_usd"]
    affordable = candidates[prices <= prices.median()]
    discovery = affordable.nlargest(n_discovery, "discovery").index
    depth = candidates.drop(discovery).nlargest(n_depth, "depth").index
    return discovery.append(depth)


def select_by_followers(raw: pd.DataFrame, k: int) -> pd.Index:
    """The agency's current rule: the k creators with most followers, whatever the client."""
    return raw.nlargest(k, "followers").index


def select_by_index(index_values: np.ndarray, k: int) -> pd.Index:
    """Top-k creators by a single index (e.g. the Oja / PC1 score), whatever the client."""
    return pd.Index(np.argsort(-np.asarray(index_values))[:k])


def summarise_selection(rows: pd.Index, raw: pd.DataFrame, creators: pd.DataFrame,
                        client: str) -> dict:
    """What a selection buys for one client: topic fit, cost, reach and strategy profile.

    Totals assume one sponsored post per selected creator.
    """
    topic = CLIENT_TOPICS[client]
    picked = raw.loc[rows]
    cost = picked["price_usd"].sum()
    arg_reach = picked["estimated_argentina_reach"].sum()
    return {
        "creators": ", ".join(picked["name"]),
        "on_topic": float((creators.loc[rows, "main_topic"] == topic).mean()),
        "mean_topic_pct": picked[topic].mean(),
        "total_price_usd": cost,
        "total_argentina_reach": arg_reach,
        # Argentine accounts reached on the client's topic per dollar: the reach
        # is weighted by the share of the creator's posts that are on topic.
        "on_topic_arg_reach_per_usd": (picked["estimated_argentina_reach"]
                                       * picked[topic] / 100).sum() / cost,
        **{f"mean_{s}": creators.loc[rows, s].mean() for s in STRATEGIES},
    }


def compare_strategies(raw: pd.DataFrame, creators: pd.DataFrame, client: str, k: int,
                       oja_index: np.ndarray | None = None) -> pd.DataFrame:
    """Summaries of every SOM + strategy selection (and the mix) next to the followers baseline.

    Args:
        oja_index: Optional creator index from exercise 1.3; if given, its top-k is
                   added as one more baseline.

    Returns:
        DataFrame with one row per selection rule.
    """
    selections = {f"som_{s}": select_by_strategy(creators, client, s, k) for s in STRATEGIES}
    selections["som_mix"] = select_mix(creators, raw, client)
    selections["followers_only"] = select_by_followers(raw, k)
    if oja_index is not None:
        selections["oja_index"] = select_by_index(oja_index, k)
    return pd.DataFrame({
        rule: summarise_selection(rows, raw, creators, client)
        for rule, rows in selections.items()
    }).T


def region_purity(creators: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab of SOM region against each creator's main topic.

    A diagonal table means the SOM regions can be used as client segments.
    """
    return pd.crosstab(creators["region"], creators["main_topic"])


# ------------------------------------------------------------------
# Maintenance: new creators and monthly updates
# ------------------------------------------------------------------

def assign_new_creators(features: pd.DataFrame, som: dict) -> pd.DataFrame:
    """Place creators the SOM was not trained on, without retraining it.

    Uses the scaler saved with the SOM, so the new rows are scaled exactly like the
    training data. A large distance to the BMU means the creator does not resemble
    anything the map has seen, which is a sign that the map needs retraining.

    Args:
        features: Rows with the same 28 columns as load_creators(), in original units.

    Returns:
        DataFrame with the BMU (i, j), its region and the distance to the BMU.
    """
    X = (features[som["feature_names"]].to_numpy() - som["scaler_mean"]) / som["scaler_scale"]
    weights = som["weights"]
    flat = weights.reshape(-1, weights.shape[2])
    distances = np.linalg.norm(X[:, None, :] - flat[None, :, :], axis=2)
    best = distances.argmin(axis=1)
    i, j = np.unravel_index(best, weights.shape[:2])
    regions = neuron_topics(neuron_profiles(som))
    return pd.DataFrame({
        "i": i,
        "j": j,
        "region": [regions.loc[(a, b)] for a, b in zip(i, j)],
        "bmu_distance": distances[np.arange(len(X)), best],
    }, index=features.index)
