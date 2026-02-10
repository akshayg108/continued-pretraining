__version__ = "0.1.0"

from . import core
from . import data
from . import utils
from . import callbacks
from . import evaluation

# Make key APIs easily accessible
from .core import (
    # BaseCPMethod,  # TODO: Implement base classes
    # MethodConfig,  # TODO: Implement base classes
    register_method,
    get_method,
    list_methods,
)

__all__ = [
    # Submodules
    "core",
    "data",
    "utils",
    "callbacks",
    "evaluation",
    # Key exports
    # "BaseCPMethod",  # TODO: Implement
    # "MethodConfig",  # TODO: Implement
    "register_method",
    "get_method",
    "list_methods",
    "__version__",
]
