__name__ = "benchmarks"
__name__ = "known_knots"

from .benchmarks import ackley__, elliptic__, rastrigin__, rosenbrock__, sphere__

# from .known_knots_fea import KnownKnotsFea

__all__ = [
    "sphere__",
    "elliptic__",
    "rastrigin__",
    "ackley__",
    "rosenbrock__",
    #     "KnownKnotsFea",
]
