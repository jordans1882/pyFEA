"""
The pyFEA library.

"""

__name__ = "pyfea"

# import importlib
from ._version import __version__
from .base_algos import *
from .benchmarks import *
from .fea import *

__all__ = [
    "__version__",
    "Function",
]


# TODO: replace all imports above with this below?
# importlib.import_module("._versoin", "__version__")
