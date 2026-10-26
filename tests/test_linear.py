"""Area A: exploratory analysis, PCA and Oja's rule."""

from pathlib import Path

import numpy as np
import pytest

from tp4.data import load_creators, standardise
from tp4.linear.pca import fit_pca

DATA_PATH = Path(__file__).parent.parent / "data" / "creadores.csv"


@pytest.fixture
def X_scaled():
    X, _ = standardise(load_creators(DATA_PATH))
    return X


def test_explained_variance_sums_to_one_and_is_sorted(X_scaled):
    pca = fit_pca(X_scaled)
    ratio = pca.explained_variance_ratio_
    assert ratio.sum() == pytest.approx(1.0)
    assert np.all(np.diff(ratio) <= 1e-12)


def test_components_are_orthonormal(X_scaled):
    V = fit_pca(X_scaled).components_
    assert V @ V.T == pytest.approx(np.eye(len(V)), abs=1e-10)


def test_scores_have_the_component_variances(X_scaled):
    # The variance of each score column is its eigenvalue (slide 23)
    pca = fit_pca(X_scaled)
    scores = pca.transform(X_scaled)
    assert scores.var(axis=0) == pytest.approx(pca.explained_variance_, abs=1e-10)


def test_matches_sklearn(X_scaled):
    sklearn_decomposition = pytest.importorskip("sklearn.decomposition")
    ours = fit_pca(X_scaled)
    ref = sklearn_decomposition.PCA().fit(X_scaled)

    assert ours.explained_variance_ratio_ == pytest.approx(ref.explained_variance_ratio_, abs=1e-10)

    # Components agree up to sign; compare only those with non-zero variance,
    # since the zero-variance directions are not unique
    nonzero = ours.explained_variance_ > 1e-8
    dots = np.abs(np.sum(ours.components_[nonzero] * ref.components_[nonzero], axis=1))
    assert dots == pytest.approx(1.0, abs=1e-6)
