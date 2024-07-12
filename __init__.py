"""from . import FEA
from . import pyfea"""

import os
import sys

from . import pyfea, tests

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
__all__ = ["pyfea", "tests"]
