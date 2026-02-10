from typing import Dict, Type, List
from .base import BaseCPMethod


# Global registry mapping method names to method classes
_METHODS_REGISTRY: Dict[str, Type[BaseCPMethod]] = {}


def register_method(name: str):
    """Decorator to register a continued pretraining method.

    This decorator adds a method class to the global registry, making it
    discoverable by the CLI and other tools.

    Example:
        >>> @register_method("my_method")
        >>> class MyMethod(BaseCPMethod):
        ...     def build_module(self, optim_config):
        ...         ...

    Args:
        name: Unique identifier for this method (used in CLI)

    Returns:
        Decorator function that registers the class

    Raises:
        ValueError: If the method name is already registered
    """

    def decorator(cls: Type[BaseCPMethod]) -> Type[BaseCPMethod]:
        if name in _METHODS_REGISTRY:
            raise ValueError(
                f"Method '{name}' is already registered. "
                f"Existing: {_METHODS_REGISTRY[name]}, New: {cls}"
            )
        if not issubclass(cls, BaseCPMethod):
            raise TypeError(f"Method class {cls} must inherit from BaseCPMethod")
        _METHODS_REGISTRY[name] = cls
        return cls

    return decorator


def get_method(name: str) -> Type[BaseCPMethod]:
    """Get a registered method class by name.

    Args:
        name: Method identifier

    Returns:
        The method class

    Raises:
        ValueError: If the method is not registered
    """
    if name not in _METHODS_REGISTRY:
        available = list(_METHODS_REGISTRY.keys())
        raise ValueError(f"Unknown method: '{name}'. Available methods: {available}")
    return _METHODS_REGISTRY[name]


def list_methods() -> List[str]:
    """List all registered method names.

    Returns:
        List of method identifiers
    """
    return sorted(_METHODS_REGISTRY.keys())


def is_method_registered(name: str) -> bool:
    """Check if a method is registered.

    Args:
        name: Method identifier

    Returns:
        True if the method is registered, False otherwise
    """
    return name in _METHODS_REGISTRY


def get_all_methods() -> Dict[str, Type[BaseCPMethod]]:
    """Get all registered methods.

    Returns:
        Dictionary mapping method names to method classes
    """
    return _METHODS_REGISTRY.copy()
