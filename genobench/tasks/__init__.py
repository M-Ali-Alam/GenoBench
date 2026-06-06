from typing import Dict, Type, Any

TASK_REGISTRY: Dict[str, Type[Any]] = {}

def register_task(name: str):
    def _register(cls: Type[Any]) -> Type[Any]:
        TASK_REGISTRY[name] = cls
        return cls
    return _register

def get_task(name: str, **kwargs) -> Any:
    """Instantiates and returns a task class."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Task '{name}' not found. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name](**kwargs)

from . import dummy