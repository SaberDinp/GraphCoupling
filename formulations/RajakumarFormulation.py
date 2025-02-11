import gurobipy as gp
from gurobipy import GRB, quicksum
import itertools

"""
Wrapper for Rajakumar et al's MIP formulation based on their description in appendix B of their paper 
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.022606
"""
class RajakumarFormulation:
    def __init__(self, A):
        N = len(A[0])
        MAX_ROW = 2 ** (N - 1)
        M = 1000
        p = self.generate_rows(N)
        # Create a new model
        model = gp.Model("RajakumarMIP")
        # Create variables
        w = model.addVars(MAX_ROW, lb=-M, vtype=GRB.CONTINUOUS, name="w")
        b = model.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")

        model.update()

        # Set objective
        model.setObjective(quicksum(b[r] for r in range(MAX_ROW)), GRB.MINIMIZE)

        model.update()

        # constrs
        model.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)))
        model.addConstrs((-b[r] * M <= w[r] for r in range(MAX_ROW)))
        model.addConstrs((A[i][j] == quicksum(p[r][i] * p[r][j] * w[r] for r in range(MAX_ROW)) for i in range(N) for j in
                      range(i + 1, N)))

        model.update()
        self.model = model


    def generate_rows(self, n):
        """
        Generate all possible rows of length n with entries +1 and -1.
        The first element is always set to +1.
        """
        # Generate all combinations of length n-1 (since the first element is fixed to +1)
        combinations = list(itertools.product([1, -1], repeat=n - 1))

        # Add +1 as the first element to all combinations
        rows = [[1] + list(comb) for comb in combinations]

        return rows

    def run(self, time_limit=600):
        self.model.setParam('TimeLimit', time_limit)
        # Optimize model
        self.model.optimize()

        return self.model.ObjVal, self.model.getVars()


if __name__ == '__main__':
    A = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ]

    rajakumar = RajakumarFormulation(A)
    print(rajakumar.run())

