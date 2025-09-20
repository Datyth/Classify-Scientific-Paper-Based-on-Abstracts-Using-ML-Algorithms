# =============================================
# src/paper_classification/models/factory.py
# =============================================
from typing import Dict, Type
from .classifiers.knn import KNNClassifier
from .classifiers.decision_tree import DecisionTreeClassifier
from .classifiers.neural_net import NeuralNetClassifier
from .classifiers.kmeans import KMeansModel
from .classifiers.transformer import TransformerClassifier
from .base import BaseModel

_REGISTRY: Dict[str, Type[BaseModel]] = {
    "knn": KNNClassifier,
    "decision_tree": DecisionTreeClassifier,
    "mlp": NeuralNetClassifier,
    "kmeans": KMeansModel,
    "transformer": TransformerClassifier,
}

class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs) -> BaseModel:
        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Options: {list(_REGISTRY)}")
        return _REGISTRY[key](**kwargs)

