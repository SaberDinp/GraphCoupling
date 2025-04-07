
import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
import math

from math import floor, ceil
from formulations.warm_start import WarmStart
from samples import graph_loader
from collections import Counter

def max_eigenvalue_multiplicity(A):
    """
    Computes the maximum multiplicity of the eigenvalues of matrix A.

    Parameters:
        A (numpy.ndarray): Input square matrix.

    Returns:
        int: Maximum multiplicity of an eigenvalue.
    """
    eigenvalues, _ = np.linalg.eig(A)
    eigenvalue_counts = Counter(np.round(eigenvalues, decimals=8))  # Rounding to avoid numerical precision issues
    return max(eigenvalue_counts.values())


class SpinFormulation:

    def __init__(self, A, MAX_ROW, start_p, start_w):
        self.best_bound = 0
        print("start P", start_p)
        print("start w", start_w)
        N = len(A[0])
        M = 1

        if start_p is not None:
            MAX_ROW = len(start_p)

        # Create a new model
        m = gp.Model("SpinMIP")

        # Create variables
        w = m.addVars(MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="w")
        w_int = m.addVars(MAX_ROW, lb=-4, ub=4, vtype=GRB.INTEGER, name="w_int")
        p = m.addVars(MAX_ROW, N, vtype=GRB.BINARY, name="p")

        # b = m.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")

        z = m.addVars(N, N, MAX_ROW, vtype=GRB.INTEGER, lb=-1, ub=1, name="z")
        t = m.addVars(N, N, MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="t")
        absw = m.addVars(MAX_ROW, lb=0, ub=M, vtype=GRB.CONTINUOUS, name="abs_w")

        trace = m.addVar(vtype=GRB.CONTINUOUS, lb=-M * MAX_ROW, ub=M * MAX_ROW, name="trace")
        row_nums = m.addVars(MAX_ROW, lb=1, ub=2 ** N - 1, vtype=GRB.CONTINUOUS, name="rownum")
        if start_p is not None:
            for r in range(MAX_ROW):
                w[r].Start = start_w[r][r]
                for i in range(N):
                    p[r, i].Start = start_p[r][i]

        # for i in range(N):
        #     for r in range(MAX_ROW):
        #         p[r, i].BranchPriority = N - i

        m.update()

        # Set objective
        # m.setObjective(quicksum(absw[r] for r in range(MAX_ROW)) + quicksum(b[r] for r in range(MAX_ROW)), GRB.MINIMIZE)

        m.setObjective(quicksum(absw[r] for r in range(MAX_ROW)) + quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), GRB.MINIMIZE)
        # m.setObjective(quicksum(absw[r] for r in range(MAX_ROW)) + quicksum(row_nums[r] for r in range(MAX_ROW)), GRB.MINIMIZE)
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)))
        # m.setObjective(quicksum(column_num[i] for i in range(N)), GRB.MINIMIZE)
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)) + quicksum(p[r, 1] for r in range(MAX_ROW)), GRB.MINIMIZE)
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)) + quicksum(row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(p[r, i] for i in range(N) for r in range(MAX_ROW)), GRB.MINIMIZE)
        # m.setObjective(quicksum(2 ** r * row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)) + quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), GRB.MINIMIZE)

        m.update()

        # constrs
        # m.addConstrs((w[r] == w_int[r] / 4 for r in range(MAX_ROW)), name="w_int")
        # m.addConstr(quicksum(b[r] for r in range(MAX_ROW)) >= N - max_eigenvalue_multiplicity(A))

        # abs-w constrs
        m.addConstrs((absw[r] >= w[r] for r in range(MAX_ROW)), name="abs1")
        m.addConstrs((absw[r] >= -w[r] for r in range(MAX_ROW)), name="abs2")

        # m.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)), name="w1")
        # m.addConstrs((-b[r] * M <= w[r] for r in range(MAX_ROW)), name="w2")

        # # order of b's wrong!
        # m.addConstrs((b[r] >= b[r + 1] for r in range(MAX_ROW - 1)), name="b1")

        # m.addConstrs((z[i, j, r] <= p[r, i] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp1")
        # m.addConstrs((z[i, j, r] <= p[r, j] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp2")
        #
        # m.addConstrs(
        #     (p[r, i] + p[r, j] - 1 <= z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp3")
        #
        # m.addConstrs((t[i, j, r] <= z[i, j, r] * M for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz1')
        # m.addConstrs((-z[i, j, r] * M <= t[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz2')
        # m.addConstrs(
        #     (t[i, j, r] <= w[r] + (1 - z[i, j, r]) * M for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz3')
        # m.addConstrs(
        #     (w[r] - (1 - z[i, j, r]) * M <= t[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz4')
        m.addConstr(trace == quicksum(w[r] for r in range(MAX_ROW)), 'trace')

        # redundant because of the definition of trace
        # m.addConstr(4 * quicksum(t[0, 0, r] for r in range(MAX_ROW)) == A[0][0] + 4 * trace)
        # the first row constraint is redundant when we include the diagonal constraint
        # m.addConstrs((4 * quicksum(t[0, j, r] for r in range(MAX_ROW)) == A[0][j] + 2 * trace + A[0][0] + A[0][j]) for j in range(1, N))
        m.addConstrs((quicksum(t[i, j, r] for r in range(MAX_ROW)) == A[i][j] for i in
                      range(1, N) for j in range(i + 1, N)), name="main1")
        m.addConstrs(
            (quicksum(t[i, i, r] for r in range(MAX_ROW)) == A[i][i] + trace  for i in
             range(1, N)), name="main2")

        # symmetry breaking on p
        m.addConstrs((row_nums[r] == quicksum(2 ** i * p[r, i] for i in range(N)) for r in range(MAX_ROW)), 'rownum1')
        m.addConstrs((row_nums[r] <= row_nums[r + 1] - 2 for r in range(MAX_ROW - 1)), 'rownum2')
        m.addConstrs((p[r, 0] == 1 for r in range(MAX_ROW)), 'pfirstcol')


        # nonlinear constraints for t
        m.addConstrs((z[i, j, r] == (2 * p[r, i] - 1) * (2 * p[r, j] - 1) for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="z_lin")
        # m.addConstrs((t[i,j,r] == w[r] * p[r,i] * p[r,j] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="t_lin")
        m.addConstrs((t[i,j,r] == w[r] * z[i,j,r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="t_nonlin")
        # cut for p

        m.addConstrs((quicksum(p[r, i] for i in range(N)) >= 2 for r in range(1, MAX_ROW)), 'psum1')

        def callback(model, where):
            if where == GRB.Callback.MIP:
                self.best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                if self.best_bound - math.floor(self.best_bound) > 0.01:
                    self.best_bound = math.ceil(self.best_bound)
            if where == GRB.Callback.MIPNODE:
                model.cbCut(quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)) >= self.best_bound)
        self.call_back = callback
        m.update()

        self.model = m

    def run(self, relax=False, time_limit=600, output_flag=1, integrality_focus=0, bqp_cuts=-1, rlt_cuts=-1, zero_half_cuts=-1):
        if relax:
            self.model = self.model.relax()

        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam("OutputFlag", output_flag)
        self.model.setParam("IntegralityFocus", integrality_focus)
        self.model.setParam("BQPCuts", bqp_cuts)
        self.model.setParam("RLTCuts", rlt_cuts)
        self.model.setParam("ZeroHalfCuts", zero_half_cuts)
        self.model.setParam("FlowCoverCuts", 0)
        self.model.setParam("MIRCuts", 0)
        self.model.Params.LazyConstraints = 1
        self.model.Params.PreCrush = 1
        self.model.update()



        # Optimize model
        # self.model.setParam("MIPFocus", 3)
        # self.model.optimize(self.call_back)
        self.model.optimize()


        return self.model.ObjVal, self.model.getVars()

if __name__ == '__main__':
    # instance = graph_loader.load("samples/instances/7_2.txt")
    instance = graph_loader.get_path_graphs(9)
    print(instance)
    print("max eigenvalue multiplicity", max_eigenvalue_multiplicity(instance))
    # instance = A
    lower_bound = 0
    warm_start = WarmStart()
    p, w = warm_start.get_union_of_stars_feasible_solution(instance, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_stars_greedily=False)
    # p, w = warm_start.get_union_of_double_stars_feasible_solution(A_15, remove_redundant_rows=True, remove_rows_with_zero_weight=True)
    # saber = SaberFormulation(instance, None, start_p=p, start_w=w)

    saber = SpinFormulation(instance, 20, None, None)
    # saber.model.setParam("Heuristics", 0.10)
    saber.model.setParam("Heuristics", 0.005)
    obj, vars = saber.run(relax=False, time_limit=1000)
    print("optimal val", obj)
    for var in vars:
        print(var)


