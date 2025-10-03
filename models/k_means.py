#models/k_means.py
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from .base.base import BaseModel


class _KMeansMajority(ClassifierMixin, BaseEstimator):
    """Helper class that clusters embeddings and maps clusters to labels.

    After fitting, each cluster is assigned the label that occurs most
    frequently among the samples assigned to that cluster.  Predictions
    output a one‑hot encoded matrix where the position of the 1 indicates
    the chosen label.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42, max_iter: int = 300) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans_: Optional[KMeans] = None
        self.label_map_: Optional[np.ndarray] = None
        self.n_labels_: Optional[int] = None

    def fit(self, X: np.ndarray, Y: np.ndarray | None = None) -> "_KMeansMajority":
        if Y is None:
            raise ValueError("Need Y to learn cluster‑label mapping.")
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init="auto",
        ).fit(X)
        clusters = self.kmeans_.labels_
        # Map each cluster to the label with the highest count
        self.label_map_ = np.zeros(self.n_clusters, dtype=int)
        for c in range(self.n_clusters):
            idx = (clusters == c)
            if not idx.any():
                self.label_map_[c] = 0
            else:
                # Sum Y over samples in the cluster and pick the most frequent label
                self.label_map_[c] = int(np.argmax(Y[idx].sum(axis=0)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans_ is None or self.label_map_ is None or self.n_labels_ is None:
            raise RuntimeError("Model not fitted.")
        cids = self.kmeans_.predict(np.asarray(X))
        Yp = np.zeros((len(cids), self.n_labels_), dtype=int)
        Yp[np.arange(len(cids)), self.label_map_[cids]] = 1
        return Yp


class KMeansModel(BaseModel):
    """K‑means based clustering model for multi‑label data.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters to form.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=300
        Maximum number of iterations of the k‑means algorithm.
    pca_components : int or None, optional
        Number of principal components to keep.  Passed to the base class.
    **kwargs : dict
        Additional parameters are stored but not used directly.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 5,
        random_state: int = 42,
        max_iter: int = 300,
        pca_components: int | None = None,
        **kwargs,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        super().__init__(pca_components=pca_components, **kwargs)

    def _build_estimator(self) -> _KMeansMajority:
        """Return a majority vote K‑means classifier."""
        return _KMeansMajority(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def _wrap_supervised(self, estimator: BaseEstimator) -> BaseEstimator:
        """Do not wrap k‑means in a one‑vs‑rest classifier."""
        return estimator