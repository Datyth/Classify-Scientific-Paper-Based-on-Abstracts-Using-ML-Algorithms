## models/base/factory.py

from models.knn import KNNClassifier
from models.decision_tree import DecisionTreeClassifier
#from models.neural_net import NeuralNetClassifier
from models.k_means import KMeansClassifier
#from models.transformer import TransformerClassifier
from .base import BaseModel

_REGISTRY = {
    "knn": KNNClassifier,
    "decision_tree": DecisionTreeClassifier,
    "kmeans": KMeansClassifier,            # or "k_means": KMeansClassifier
    #"transformer": TransformerClassifier,  # if you have it
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