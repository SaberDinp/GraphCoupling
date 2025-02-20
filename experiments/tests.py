from formulations.RajakumarFormulation import RajakumarFormulation
from formulations.SaberFormulation import SaberFormulation
import numpy as np

from formulations.warm_start import WarmStart

if __name__ == '__main__':
    for n in range(20, 22):
        print("_______________________")
        A = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1  # Ensure symmetry
        warm_start = WarmStart()
        print("n", n)
        print(A)
        # rajakumar = RajakumarFormulation(A.tolist())
        # rajakumar.run(100)
        p, w = warm_start.get_union_of_stars_feasible_solution(A, remove_redundant_rows=True,
                                                               remove_rows_with_zero_weight=True,
                                                               choose_stars_greedily=False)
        saber = SaberFormulation(A, 11, start_p=p, start_w=w, lower_bound=n - 1)
        obj, vars = saber.run(relax=False, time_limit=100)
