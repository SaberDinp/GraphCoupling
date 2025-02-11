import gurobipy as gp

if __name__ == '__main__':
    m = gp.Model("test")
    x1 = m.addVar(vtype= gp.GRB.CONTINUOUS, name="x1", lb=0, ub=1)
    x2 = m.addVar(vtype= gp.GRB.CONTINUOUS, name="x2", lb=0, ub=1)
    x3 = m.addVar(vtype= gp.GRB.CONTINUOUS, name="x3", lb=0, ub=1)
    x4 = m.addVar(vtype= gp.GRB.CONTINUOUS, name="x4", lb=0, ub=1)
    x5 = m.addVar(vtype= gp.GRB.CONTINUOUS, name="x5", lb=0, ub=1)
    m.addConstr(x1 + x2 <= 1)
    m.addConstr(x2 + x3 <= 1)
    m.addConstr(x1 + x3 <= 1 )
    m.addConstr(4 * x3 + 3 * x4 + 5 * x5 <= 10)
    m.addConstr(x1 + 2 * x4 <= 2)
    m.addConstr(3 * x2 + 4 * x5 <= 5)
    m.setObjective(3 * x1 + 2 * x2 + x3 + 2 * x4 + x5, gp.GRB.MAXIMIZE)

    # cuts
    m.addConstr(x3 + x4 + x5 <= 2)
    m.addConstr(x1 + x4 <= 1)
    m.addConstr(x2 + x5 <= 1)
    m.addConstr(x1 + x2 + x3 <= 1)
    m.update()
    m.optimize()
    for var in m.getVars():
        print(var)