from feareu.benchmarks.known_knots_fea import KnownKnotsFea
from feareu.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO
tester = KnownKnotsFea(number_of_knots=12, number_of_points=1000, max_error=0.01, delta=0.01, base_algo=BsplineFeaPSO, diagnostics_amount=1, generations=10, pop_size=15)
tester.run()