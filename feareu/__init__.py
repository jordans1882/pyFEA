"""
The pytemplate library.

A simple template Python library. Contains functionality for linear algebra
and statistics

Provides:
    - A matrix class

"""

__name__ = "pytemplate"

# import importlib

from ._version import __version__

from .template import Template
from .matrix import Matrix


__all__ = [
    "__version__",
    "Template",
    "Matrix",
]


# TODO: replace all imports above with this below?
# importlib.import_module("._versoin", "__version__")
