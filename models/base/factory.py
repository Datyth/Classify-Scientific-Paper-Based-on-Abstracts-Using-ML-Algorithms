
from typing import Dict, Type
from models.knn import KNNClassifier
from models.decision_tree import DecisionTreeClassifier
#from models.neural_net import NeuralNetClassifier
from models.k_means import KMeansClassifier
#from models.transformer import TransformerClassifier
from .base import BaseModel

_REGISTRY: Dict[str, Type[BaseModel]] = {
    "knn": KNNClassifier,
    "decision_tree": DecisionTreeClassifier,
    #"mlb":NeuralNetClassifier,
    "kmeans": KMeansClassifier,
    #"transformer": TransformerClassifier,
}

class ModelFactory:
    @staticmethod
    def create(name: str, **kwargs) -> BaseModel:
        key = name.lower()
        if key not in _REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Options: {list(_REGISTRY)}")
        return _REGISTRY[key](**kwargs)

