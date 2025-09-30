# models/kmeans.py
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from models.base.base import BaseModel 

class KMeansLabelMapper(BaseEstimator, ClassifierMixin):
    """KMeans + majority-label mapping → returns one-hot over n_labels."""
    def __init__(self, n_clusters: int = 5, random_state: int = 42, max_iter: int = 300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.kmeans_: Optional[KMeans] = None
        self.label_map_: Optional[np.ndarray] = None  # [n_clusters] -> label_idx
        self.n_labels_: Optional[int] = None

    def fit(self, X, Y=None):
        if Y is None:
            raise ValueError("KMeansLabelMapper requires Y (binary labels) to build cluster→label mapping.")
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.n_labels_ = Y.shape[1]

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init="auto",
            random_state=self.random_state, max_iter=self.max_iter
        ).fit(X)

        clusters = self.kmeans_.labels_
        self.label_map_ = np.zeros(self.n_clusters, dtype=int)
        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) == 0:
                self.label_map_[c] = 0
            else:
                counts = Y[idx].sum(axis=0)
                self.label_map_[c] = int(np.argmax(counts))
        return self

    def predict(self, X):
        if self.kmeans_ is None or self.label_map_ is None or self.n_labels_ is None:
            raise RuntimeError("Model not fitted.")
        cids = self.kmeans_.predict(X)  # [n_samples]
        Yp = np.zeros((len(cids), self.n_labels_), dtype=int)
        for i, c in enumerate(cids):
            j = int(self.label_map_[int(c)])
            Yp[i, j] = 1
        return Yp


class KMeansClassifier(BaseModel):
    """BaseModel wrapper so SBERT lives in the pipeline; no OvR wrapping."""
    def __init__(self, n_clusters: int = 5, random_state: int = 42, max_iter: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def _build_estimator(self):
        return KMeansLabelMapper(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter
        )

    def _wrap_supervised(self, estimator):
        # IMPORTANT: don't wrap KMeans in OneVsRest
        return estimator
#models/knn.py
from sklearn.neighbors import KNeighborsClassifier
from .base.base import BaseModel

class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _build_estimator(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
