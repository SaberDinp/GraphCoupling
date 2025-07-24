import gurobipy as gp
from gurobipy import GRB, quicksum
import itertools

from samples import graph_loader

"""
Implementation of Rajakumar et al.'s MIP formulation for Graph Coupling Problem in Gurobi described in Appendix B of their paper:
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.022606
"""


class Baseline:
    def __init__(self, A):
        """
        Initialize the MIP model for a given adjacency matrix A.

        Parameters:
        A (list of list of int): Adjacency matrix of the input graph (assumed symmetric and unweighted).
        """
        N = len(A[0])  # Number of vertices
        MAX_ROW = 2 ** (N - 1)  # Number of rows of the matrix P
        M = sum(sum(A[i]) for i in range(N)) // 2  # Total number of edges used as the big M value

        # Generate all sign vectors r in {±1}^N with first entry fixed to +1
        p = self.generate_rows(N)

        # Create a new Gurobi model
        model = gp.Model("Baseline")

        # Continuous weights w_r, bounded by [-M, M]
        w = model.addVars(MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="w")

        # Binary activation variables b_r to control whether w_r is active
        b = model.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")

        # Objective: minimize the number of active variables
        model.setObjective(quicksum(b[r] for r in range(MAX_ROW)), GRB.MINIMIZE)

        # Constraints: enforce w[r] == 0 when b[r] == 0 (big-M encoding)
        model.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)))
        model.addConstrs((w[r] >= -b[r] * M for r in range(MAX_ROW)))

        # Reconstruction constraint:
        # A[i][j] == sum_r w[r] * p[r][i] * p[r][j] for all i < j
        model.addConstrs(
            (
                A[i][j] == quicksum(p[r][i] * p[r][j] * w[r] for r in range(MAX_ROW))
                for i in range(N) for j in range(i + 1, N)
            )
        )

        self.model = model

    def generate_rows(self, n):
        """
        Generate all ±1 vectors of length n with the first entry fixed to +1.

        Parameters:
        n (int): Length of each vector.

        Returns:
        list of list of int: All 2^(n-1) vectors in {±1}^n with first entry +1.
        """
        # Generate all 2^(n-1) combinations for the remaining n-1 entries
        combinations = list(itertools.product([1, -1], repeat=n - 1))

        # Prepend +1 to each combination to fix the first coordinate
        rows = [[1] + list(comb) for comb in combinations]

        return rows

    def run(self, time_limit=600, output_flag=1):
        """
        Solve the model with time limit and verbosity control.

        Parameters:
        time_limit (int): Maximum runtime in seconds (default: 600).
        output_flag (int): Gurobi output flag (1 for verbose, 0 for silent).

        Returns:
        tuple: (objective value, list of Gurobi variables)
        """
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('TimeLimit', time_limit)

        # Solve the optimization problem
        self.model.optimize()

        return self.model.ObjVal, self.model.getVars()


if __name__ == '__main__':
    # Load the adjacency matrix of a path graph with 15 nodes
    A = graph_loader.get_path_graphs(15)

    # Instantiate and solve the Rajakumar formulation
    baseline_model = Baseline(A)
    print(baseline_model.run(output_flag=0))
