from feareu.benchmarks.known_knots_de import KnownKnotsDe
from feareu.benchmarks.known_knots_fea import KnownKnotsFea
from feareu.base_algos.bspline_specific.bspline_fea_pso import BsplineFeaPSO
from feareu.benchmarks.known_knots_ga import KnownKnotsGa
from feareu.benchmarks.known_knots_pso import KnownKnotsPso

fea_tester = KnownKnotsFea(number_of_knots=12, number_of_points=1000, overlap=1, factor_size=3, max_error=0.01, delta=0.01, base_algo=BsplineFeaPSO, diagnostics_amount=1, generations=10, pop_size=15)
fea_tester.run()

"""pso_tester = KnownKnotsPso(number_of_knots=10, factor_size=3, overlap=1, number_of_points=1000, max_error=0.01, delta=0.015, diagnostics_amount=1, pop_size=200)
pso_tester.run()"""

"""ga_tester = KnownKnotsGa(number_of_knots=10, factor_size=3, overlap=1, number_of_points=1000, max_error=0.01, delta=0.018, diagnostics_amount=1, tournament_options = 2, number_of_children = 20, pop_size=200)
ga_tester.run()"""

"""de_tester = KnownKnotsDe(number_of_knots=10, factor_size=3, overlap=1, number_of_points=1000, max_error=0.01, delta=0.018, diagnostics_amount=1, pop_size=200)
de_tester.run()"""