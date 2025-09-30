## models/base/factory.py

from models.knn import KNNModel
from models.decision_tree import DecisionTreeModel
from models.k_means import KMeansModel
from models.transformer import TransformerModel 
from .base import BaseModel

_REGISTRY = {
    "knn": KNNModel,
    "decision_tree": DecisionTreeModel,
    "kmeans": KMeansModel,
    #"transformer": TransformerModel, 
}

class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs):
        key = (name or "").lower()
        if key not in _REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Try one of: {', '.join(sorted(_REGISTRY))}")
        return _REGISTRY[key](**kwargs)

    @staticmethod
    def choices() -> tuple[str, ...]:
        return tuple(sorted(_REGISTRY.keys()))