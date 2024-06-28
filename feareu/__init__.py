"""
The feareu library.

A simple template Python library. Contains functionality for linear algebra
and statistics

Provides:
    - A matrix class

"""

__name__ = "feareu"

# import importlib
from ._version import __version__
from .base_algos import *
from .benchmarks import *
from .fea import *
from .experiments import *
from .function import Function
from .matrix import Matrix
from .template import Template

__all__ = [
    "__version__",
    "Template",
    "Matrix",
    "rastrigin__",
    "Function",
]


# TODO: replace all imports above with this below?
# importlib.import_module("._versoin", "__version__")
