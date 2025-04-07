from formulations.RajakumarFormulation import RajakumarFormulation
from formulations.SaberFormulation import SaberFormulation
import numpy as np

from formulations.warm_start import WarmStart
from samples import graph_loader
import os
import pandas as pd

if __name__ == '__main__':
    time_limit = 10
    directory = "../samples/instances"
    saber_results = {}
    rajakumar_results = {}
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
    else:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(filepath):
                print("file", filename)
                instance = graph_loader.load(filepath)
                warm_start = WarmStart()
                p, w = warm_start.get_union_of_stars_feasible_solution(instance, remove_redundant_rows=True,
                                                                       remove_rows_with_zero_weight=True,
                                                                       choose_stars_greedily=False)
                try:
                    saber = SaberFormulation(instance, None, start_p=p, start_w=w)
                    obj, vars = saber.run(relax=False, time_limit=time_limit, heuristics=0.005, output_flag=0)
                    gap = saber.model.MIPGap
                    best_bound = saber.model.ObjBound
                    runtime = saber.model.Runtime
                    print("Saber:", "obj", obj, "bound", best_bound, "gap", gap, "runtime", runtime)
                    saber_results[filename] = {"obj": obj, "gap": gap, "best_bound": best_bound, "runtime": runtime}
                except:
                    print("exception for saber formulation in graph", filepath)

                try:
                    rajakumar = RajakumarFormulation(instance)
                    obj, vars = rajakumar.run(time_limit=time_limit, output_flag=0)
                    best_bound = rajakumar.model.ObjBound
                    gap = rajakumar.model.MIPGap
                    runtime = rajakumar.model.Runtime
                    rajakumar_results[filename] = {"obj": obj, "gap": gap, "best_bound": best_bound, "runtime": runtime}
                    print("rajakumar:", "obj", obj, "bound", best_bound, "gap", gap, "runtime", runtime)
                except:
                    print("exception for rajakumar formulation in graph", filepath)

    saber_df = pd.DataFrame(saber_results)

    # Save as CSV
    saber_df.to_csv('saber_results.csv', index=True)

    rajakumar_df = pd.DataFrame(rajakumar_results)
    rajakumar_df.to_csv('rajakumar_results.csv', index=True)

