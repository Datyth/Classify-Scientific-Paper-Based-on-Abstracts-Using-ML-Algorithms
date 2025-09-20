# src/paper_classification/models/classifiers/knn.py
from sklearn.neighbors import KNeighborsClassifier
from ..base import BaseModel

class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _build_estimator(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
