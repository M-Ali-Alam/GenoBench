from typing import Dict, Type, Any

# Central registry for all Genomic Foundation Models
MODEL_REGISTRY: Dict[str, Type[Any]] = {}

def register_model(name: str):
    """Decorator to dynamically register a model class into the framework."""
    def _register(cls: Type[Any]) -> Type[Any]:
        MODEL_REGISTRY[name] = cls
        return cls
    return _register

def get_model(name: str, **kwargs) -> Any:
    """Instantiates and returns a model, passing any configuration kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

from .base import BaseGFM

# Import modules here so the Python interpreter executes the decorators
from . import dummy
from . import hyenadna 
from . import dnabert2