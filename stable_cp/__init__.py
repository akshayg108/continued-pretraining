__version__ = "0.1.0"

from . import core
from . import data
from . import utils
from . import callbacks
from . import evaluation

# Make key APIs easily accessible
from .core import (
    BaseCPMethod,
    MethodConfig,
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
    "BaseCPMethod",
    "MethodConfig",
    "register_method",
    "get_method",
    "list_methods",
    "__version__",
]
