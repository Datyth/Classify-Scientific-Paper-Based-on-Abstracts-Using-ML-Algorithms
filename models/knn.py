from sklearn.neighbors import KNeighborsClassifier as _KNN
from .base.base import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors: int = 5, weights: str = "distance", **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _build_estimator(self):
        return _KNN(n_neighbors=self.n_neighbors, weights=self.weights)
