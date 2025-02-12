
import gurobipy as gp
from gurobipy import GRB, quicksum

from formulations.warm_start import WarmStart


class SaberFormulation:
    def __init__(self, A, MAX_ROW, start_p, start_w):
        N = len(A[0])
        M = 10

        if start_p is not None:
            MAX_ROW = len(start_p)

        # Create a new model
        m = gp.Model("SaberMIP")

        # Create variables
        w = m.addVars(MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="w")
        p = m.addVars(MAX_ROW, N, vtype=GRB.BINARY, name="p")

        # for r in range(MAX_ROW):
        #     for i in range(N):
        #         p[r, i].BranchPriority = 1

        b = m.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")
        # for r in range(MAX_ROW):
        #     b[r].BranchPriority = 200
        z = m.addVars(N, N, MAX_ROW, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z")
        t = m.addVars(N, N, MAX_ROW, lb=-M, ub=M, vtype=GRB.CONTINUOUS, name="t")

        trace = m.addVar(vtype=GRB.CONTINUOUS, lb=-M * MAX_ROW, ub=M * MAX_ROW, name="trace")
        row_nums = m.addVars(MAX_ROW, vtype=GRB.CONTINUOUS, name="rownum")
        print("start_p", start_p)
        print("start_w", start_w)
        print((2 * start_p - 1).T @ start_w @ (2 * start_p - 1))
        for r in range(MAX_ROW):
            w[r].Start = start_w[r][r]
            for i in range(N):
                p[r, i].Start = start_p[r][i]
            # for i in range(N):
            #     for j in range(i, N):
            #         z[i, j, r].Start = start_p[r][i] * start_p[r][j]
            #         t[i, j, r].Start = start_p[r][i] * start_p[r][j] * start_w[r][r]
            # b[r].Start = 1
            # trace.Start = sum(start_w[r][r] for r in range(MAX_ROW))
            # row_nums[r].Start = sum(start_p[r][i] * 2 ** i for i in range(N))
            print("r", r, sum(start_p[r][i] * 2 ** i for i in range(N)))

        m.update()

        # Set objective

        m.setObjective(quicksum(b[r] for r in range(MAX_ROW)))
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)) + quicksum(row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(2 ** r * row_nums[r] for r in range(MAX_ROW)),
        #                GRB.MINIMIZE)
        # m.setObjective(quicksum(b[r] for r in range(MAX_ROW)) + quicksum(z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), GRB.MINIMIZE)

        m.update()

        # m.addConstrs((z[i, j, r] >= 0 for i in range(N) for j in range(N) for r in range(MAX_ROW)))
        # m.addConstrs((z[i, j, r] <= 1 for i in range(N) for j in range(N) for r in range(MAX_ROW)))
        m.addConstr(quicksum(b[r] for r in range(MAX_ROW)) >= 19)
        # constrs

        # m.addConstrs((b[r] == 1 for r in range(MAX_ROW)))

        m.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)), name="w1")
        m.addConstrs((-b[r] * M <= w[r] for r in range(MAX_ROW)), name="w2")

        m.addConstrs((z[i, j, r] <= p[r, i] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp1")
        m.addConstrs((z[i, j, r] <= p[r, j] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp2")
        # this constraint is no longer correct
        # m.addConstrs((quicksum(z[i, j, r] for j in range(i, N)) + quicksum(z[j, i, r] for j in range(i)) <= N * p[r, i] for i in range(N) for r in range(MAX_ROW)))
        m.addConstrs(
            (p[r, i] + p[r, j] - 1 <= z[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name="zp3")
        m.addConstrs((t[i, j, r] <= z[i, j, r] * M for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz1')
        m.addConstrs((-z[i, j, r] * M <= t[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz2')
        m.addConstrs(
            (t[i, j, r] <= w[r] + (1 - z[i, j, r]) * M for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz3')
        m.addConstrs(
            (w[r] - (1 - z[i, j, r]) * M <= t[i, j, r] for i in range(N) for j in range(i, N) for r in range(MAX_ROW)), name='tz4')
        m.addConstr(trace == quicksum(w[r] for r in range(MAX_ROW)), 'trace')
        # redundant because of the definition of trace
        # m.addConstr(4 * quicksum(t[0, 0, r] for r in range(MAX_ROW)) == A[0][0] + 4 * trace)
        # the first row constraint is redundant when we include the diagonal constraint
        # m.addConstrs((4 * quicksum(t[0, j, r] for r in range(MAX_ROW)) == A[0][j] + 2 * trace + A[0][0] + A[0][j]) for j in range(1, N))
        m.addConstrs((4 * quicksum(t[i, j, r] for r in range(MAX_ROW)) == A[i][j] + trace + A[0][i] + A[0][j] for i in
                      range(1, N) for j in range(i + 1, N)), name="main1")
        m.addConstrs(
            (4 * quicksum(t[i, i, r] for r in range(MAX_ROW)) == A[i][i] + 2 * trace + A[0][i] + A[0][i] for i in
             range(1, N)), name="main2")

        # symmetry breaking on p
        m.addConstrs((row_nums[r] == quicksum(2 ** i * p[r, i] for i in range(N)) for r in range(MAX_ROW)), 'rownum1')
        m.addConstrs((row_nums[r] <= row_nums[r + 1] - 2 for r in range(MAX_ROW - 1)), 'rownum2')
        m.addConstrs((p[r, 0] == 1 for r in range(MAX_ROW)), 'pfirstcol')

        # cuts
        m.addConstrs((p[r, i] + p[r, j] + p[r, k] - z[i, j, r] - z[i, k, r] - z[j, k, r] <= 1 for i in range(N) for j in
                      range(i + 1, N) for k in range(j + 1, N) for r in range(MAX_ROW)))
        # m.addConstrs((p[r, i] + p[r, j] + p[r, k] + p[r, l] - z[i, j, r] - z[i, k, r] - z[i, l, r] - z[j, k, r] - z[j, l, r] - z[k, l, r] <= 1 for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N) for l in range(k + 1, N) for r in range(MAX_ROW)))
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
    # A = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]

    # A = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]
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
    A_15 = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ]
    # A_25 = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]
    # A_20 = [
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # ]

    # A = [[0, 0, 1, 0],
    #      [0, 0, 0, 0],
    #      [1, 0, 0, 0],
    #      [0, 0, 0, 0]]
    warm_start = WarmStart()
    p, w = warm_start.get_union_of_stars_feasible_solution(A_15, remove_redundant_rows=True, remove_rows_with_zero_weight=True, choose_stars_greedily=False)
    # p, w = warm_start.get_union_of_double_stars_feasible_solution(A_15, remove_redundant_rows=True, remove_rows_with_zero_weight=True)
    saber = SaberFormulation(A_15, 11, start_p=p, start_w=w)
    obj, vars = saber.run(relax=False, time_limit=1000)
    print("optimal val", obj)
    for var in vars:
        print(var)
