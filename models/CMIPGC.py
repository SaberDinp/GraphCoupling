import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from collections import Counter
from samples import graph_loader


def max_eigenvalue_multiplicity(A):
    """
    Computes the maximum multiplicity of the eigenvalues of matrix A.

    Parameters:
        A (list of list of int): Input square matrix.

    Returns:
        int: Maximum multiplicity of an eigenvalue.
    """
    eigenvalues, _ = np.linalg.eig(A)
    # Round eigenvalues to mitigate floating point issues
    eigenvalue_counts = Counter(np.round(eigenvalues, decimals=8))
    return max(eigenvalue_counts.values())


class CMIPGC:
    """
    Compact Mixed Integer Programming formulation for the Graph Coupling Problem.

    It uses auxiliary variables to linearize
    bilinear products, enforces feasibility constraints, and applies symmetry-breaking
    and other cuts for improved performance.
    """

    def __init__(self, A, MAX_ROW, start_p=None, start_w=None):
        self.best_bound = 0
        N = len(A[0])        # Number of nodes in the graph
        M = 10               # Big-M value for bounding variables

        if start_p is not None:
            MAX_ROW = len(start_p)

        # Create a new Gurobi model
        m = gp.Model("CMIPGC")

        # Variables:
        # w[r]: weight of the r-th row
        w = m.addVars(MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="w")

        # p[r, i]: binary value for i-th element of r-th vector (used in decomposition)
        p = m.addVars(MAX_ROW, N, vtype=GRB.BINARY, name="p")

        # b[r]: binary indicator if the r-th term is active (non-zero weight)
        b = m.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")

        # z[i, j, r]: linearization of p[r, i] * p[r, j]
        z = m.addVars(N, N, MAX_ROW, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")

        # t[i, j, r]: linearization of z[i, j, r] * w[r]
        t = m.addVars(N, N, MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="t")

        # trace: sum of all weights
        trace = m.addVar(vtype=GRB.CONTINUOUS, lb=-M * MAX_ROW, ub=M * MAX_ROW, name="trace")

        # row_nums[r]: decimal encoding of p[r] used for symmetry-breaking
        row_nums = m.addVars(MAX_ROW, lb=1, ub=2 ** N - 1, vtype=GRB.CONTINUOUS, name="rownum")

        # Optional warm start from initial solution
        if start_p is not None and start_w is not None:
            for r in range(MAX_ROW):
                w[r].Start = start_w[r][r]
                for i in range(N):
                    p[r, i].Start = start_p[r][i]

        m.update()

        # Objective: minimize the number of active rows (terms used)
        m.setObjective(quicksum(b[r] for r in range(MAX_ROW)))
        m.update()

        # Big-M constraints to bind w[r] to b[r]
        m.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)), name="w1")
        m.addConstrs((-b[r] * M <= w[r] for r in range(MAX_ROW)), name="w2")

        # Linearization of z[i,j,r] = p[r,i] * p[r,j] via McCormick inequalities
        m.addConstrs((z[i, j, r] <= p[r, i] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp1")
        m.addConstrs((z[i, j, r] <= p[r, j] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp2")
        m.addConstrs((p[r, i] + p[r, j] - 1 <= z[i, j, r]
                      for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp3")

        # Linearization of t[i,j,r] = z[i,j,r] * w[r]
        m.addConstrs((t[i, j, r] <= z[i, j, r] * M
                      for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz1')
        m.addConstrs((t[i, j, r] >= -z[i, j, r] * M
                      for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz2')
        m.addConstrs((t[i, j, r] <= w[r] + (1 - z[i, j, r]) * M
                      for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz3')
        m.addConstrs((t[i, j, r] >= w[r] - (1 - z[i, j, r]) * M
                      for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz4')

        # Trace constraint: sum of all weights
        m.addConstr(trace == quicksum(w[r] for r in range(MAX_ROW)), 'trace')

        # Feasibility constraints: enforce target entries of A
        m.addConstrs((4 * quicksum(t[i, j, r] for r in range(MAX_ROW)) ==
                      A[i][j] + trace + A[0][i] + A[0][j]
                      for i in range(1, N) for j in range(i + 1, N)), name="main1")

        m.addConstrs((4 * quicksum(t[i, i, r] for r in range(MAX_ROW)) ==
                      A[i][i] + 2 * trace + A[0][i] + A[0][i]
                      for i in range(1, N)), name="main2")

        # Symmetry-breaking: fix p[r,0] = 1 and lex order constraints
        m.addConstrs((p[r, 0] == 1 for r in range(MAX_ROW)), 'pfirstcol')
        m.addConstrs((row_nums[r] == quicksum(2 ** i * p[r, i] for i in range(N))
                      for r in range(MAX_ROW)), 'rownum1')
        m.addConstrs((row_nums[r] <= row_nums[r + 1] - 2 for r in range(MAX_ROW - 1)), 'rownum2')

        # Row sum cut: prevent trivial solutions
        m.addConstrs((quicksum(p[r, i] for i in range(N)) >= 2 for r in range(1, MAX_ROW)), 'psum1')

        # Padberg triangle inequalities: tighten relaxation
        m.addConstrs((p[r, i] + p[r, j] + p[r, k]
                      - z[i, j, r] - z[i, k, r] - z[j, k, r] <= 1
                      for i in range(N) for j in range(i + 1, N)
                      for k in range(j + 1, N) for r in range(MAX_ROW)), name="padberg")

        # Spectral bound: number of terms ≥ n - λ_max multiplicity
        m.addConstr(quicksum(b[r] for r in range(MAX_ROW)) >= N - max_eigenvalue_multiplicity(A), name="spectral")

        m.update()
        self.model = m

    def run(self, relax=False, time_limit=600, output_flag=1,
            integrality_focus=0, bqp_cuts=-1, rlt_cuts=-1,
            zero_half_cuts=-1, heuristics=0.005):
        """
        Solve the CMIPGC model with optional solver tuning.

        Parameters:
            relax (bool): If True, solves the LP relaxation.
            time_limit (int): Time limit in seconds.
            output_flag (int): Verbosity of solver output.
            integrality_focus (int): Focus on integrality (0–3).
            bqp_cuts, rlt_cuts, zero_half_cuts (int): Cut strategy overrides.
            heuristics (float): Gurobi heuristic intensity.

        Returns:
            tuple: (objective value, list of variable objects)
        """
        if relax:
            self.model = self.model.relax()

        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('IntegralityFocus', integrality_focus)
        self.model.setParam('BQPCuts', bqp_cuts)
        self.model.setParam('RLTCuts', rlt_cuts)
        self.model.setParam('ZeroHalfCuts', zero_half_cuts)
        self.model.setParam('FlowCoverCuts', 0)
        self.model.setParam('MIRCuts', 0)
        self.model.setParam('Heuristics', heuristics)

        self.model.update()
        self.model.optimize()

        return self.model.ObjVal, self.model.getVars()


if __name__ == '__main__':
    instance = graph_loader.get_path_graphs(10)
    cmipgc = CMIPGC(instance, 30)
    obj, vars = cmipgc.run(relax=False, time_limit=1000, heuristics=0.005, output_flag=1)
    print("optimal val", obj)
    for var in vars:
        print(var)
