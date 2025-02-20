
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from formulations.warm_start import WarmStart
import scipy as sp

class SaberMatrixFormulation:
    def __init__(self, A, MAX_ROW, start_p, start_w, lower_bound):
        N = len(A[0])
        M = 10

        COEF = np.ndarray(shape=(N, N))
        for i in range(N):
            for j in range(N):
                COEF[i][j] = 1
        COEF[0][0] = 4
        for i in range(1, N):
            COEF[i][0] = 2
            COEF[0][i] = 2
            COEF[i][i] = 2

        print(COEF)
        if start_p is not None:
            MAX_ROW = len(start_p)

        # Create a new model
        m = gp.Model("SaberMatrixMIP")

        # Create variables
        w = m.addMVar(MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="w")
        p = m.addMVar((MAX_ROW, N), vtype=GRB.BINARY, name="p")

        b = m.addMVar(MAX_ROW, vtype=GRB.BINARY, name="b")

        trace = m.addVar(vtype=GRB.CONTINUOUS, lb=-M * MAX_ROW, ub=M * MAX_ROW, name="trace")
        row_nums = m.addVars(MAX_ROW, vtype=GRB.CONTINUOUS, name="rownum")

        # warm start
        for r in range(MAX_ROW):
            w[r].Start = start_w[r][r]
            for i in range(N):
                p[r, i].Start = start_p[r][i]

        m.update()

        # Set objective

        m.setObjective(b.sum())

        m.update()

        # constrs
        m.addConstr(quicksum(b[r] for r in range(MAX_ROW)) >= lower_bound)

        m.addConstr(w <= M * b , name="w_upper")
        m.addConstr((-M * b <= w), name="w_lower")

        m.addConstr(trace == w.sum(), 'trace')

        print((p * w.reshape(MAX_ROW, 1)))
        # main constraints
        aux = (p * w.reshape(MAX_ROW, 1))
        m.addConstr(4 * p.T @ aux == A + trace * COEF)

        # symmetry breaking on p
        m.addConstrs((row_nums[r] == quicksum(2 ** i * p[r, i] for i in range(N)) for r in range(MAX_ROW)), 'rownum1')
        m.addConstrs((row_nums[r] <= row_nums[r + 1] - 2 for r in range(MAX_ROW - 1)), 'rownum2')
        m.addConstrs((p[r, 0] == 1 for r in range(MAX_ROW)), 'pfirstcol')

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
        self.model.update()

        # Optimize model
        self.model.optimize()

        return self.model.ObjVal, self.model.getVars()

if __name__ == '__main__':
    # A_12 = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]
    A = [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]]

    warm_start = WarmStart()
    p, w = warm_start.get_union_of_stars_feasible_solution(A, remove_redundant_rows=True,
                                                           remove_rows_with_zero_weight=True,
                                                           choose_stars_greedily=False)
    # p, w = warm_start.get_union_of_double_stars_feasible_solution(A_15, remove_redundant_rows=True, remove_rows_with_zero_weight=True)
    saber = SaberMatrixFormulation(A, 50, start_p=p, start_w=w, lower_bound=0)
    obj, vars = saber.run(relax=True, time_limit=1000)
    print("optimal val", obj)
    for var in vars:
        print(var)