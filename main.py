#!/usr/bin/env python3.11

# Copyright 2024, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB, quicksum

MAX_ROW = 40
N = 10
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


# A = [
#     [0, 1, 1],
#     [1, 0, 0],
#     [1, 0, 0]
# ]
M = 10

try:
    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    w = m.addVars(MAX_ROW, lb= -M, vtype=GRB.CONTINUOUS, name="w")
    p = m.addVars(MAX_ROW, N, vtype=GRB.BINARY, name="p")
    b = m.addVars(MAX_ROW, vtype=GRB.BINARY, name="b")
    z = m.addVars(N, N, MAX_ROW, vtype=GRB.BINARY, name="z")
    t = m.addVars(N, N, MAX_ROW, lb=-M, vtype=GRB.CONTINUOUS, name="t")
    trace = m.addVar(vtype=GRB.CONTINUOUS, name="trace")

    m.update()


    # Set objective
    m.setObjective(quicksum(b[r] for r in range(MAX_ROW)), GRB.MINIMIZE)

    m.update()

    # constrs
    m.addConstrs((w[r] <= b[r] * M for r in range(MAX_ROW)))
    m.addConstrs((-b[r] * M <= w[r] for r in range(MAX_ROW)))
    m.addConstrs((z[i, j, r] <= p[r, i] for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((z[i, j, r] <= p[r, j] for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((p[r, i] + p[r, j] - 1 <= z[i, j, r] for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((t[i, j, r] <= z[i, j, r] * M for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((-z[i, j, r] * M <= t[i, j, r] for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((t[i, j, r] <= w[r] + (1 - z[i, j, r]) * M for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstrs((w[r] - (1 - z[i, j, r]) * M <= t[i, j, r] for i in range(N) for j in range(N) for r in range(MAX_ROW)))
    m.addConstr(trace == quicksum(w[r] for r in range(MAX_ROW)))
    m.addConstr(4 * quicksum(t[0, 0, r] for r in range(MAX_ROW)) == A[0][0] + 4 * trace)
    m.addConstrs((4 * quicksum(t[0, j, r] for r in range(MAX_ROW)) == A[0][j] + 2 * trace + A[0][0] + A[0][j]) for j in range(1, N))
    m.addConstrs((4 * quicksum(t[i, j, r] for r in range(MAX_ROW)) == A[i][j] + trace + A[0][i] + A[0][j] for i in range(1, N) for j in range(i + 1, N)))
    m.addConstrs((4 * quicksum(t[i, i, r] for r in range(MAX_ROW)) == A[i][i] + 2 * trace + A[0][i] + A[0][i] for i in range(1, N)))

    m.addConstrs((p[r, 0] == 1 for r in range(MAX_ROW)))



    m.update()

    m.setParam('TimeLimit', 600)
    # Optimize model
    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")