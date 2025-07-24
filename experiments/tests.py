from models.baseline_MIP import Baseline
from models.CMIPGC import CMIPGC
from models.warm_start import WarmStart
from samples import graph_loader

import os
import pandas as pd

# ============================================================
# Evaluate both Baseline and CMIPGC formulations on all graphs
# in the ../samples/instances directory, with a time limit of 3600 seconds.
# Results (objective, gap, bound, runtime) are saved to CSV.
# ============================================================

if __name__ == '__main__':
    time_limit = 3600  # Gurobi time limit for each model (in seconds)
    instances_directory = "../samples/instances"  # Directory containing input graph instances
    cmipgc_results = {}     # Dictionary to store CMIPGC results
    baseline_results = {}   # Dictionary to store Baseline results

    # Check if instance directory exists
    if not os.path.isdir(instances_directory):
        print(f"Directory '{instances_directory}' does not exist.")
    else:
        # Loop through all .txt graph instance files
        for filename in os.listdir(instances_directory):
            filepath = os.path.join(instances_directory, filename)

            # Ensure we are only processing files (not subdirectories)
            if os.path.isfile(filepath) and filename.endswith(".txt"):
                print(f"Reading '{filepath}'")

                # Load adjacency matrix from file
                instance = graph_loader.load(filepath)

                # Construct warm-start solution using union-of-stars heuristic
                warm_start = WarmStart()
                p, w = warm_start.get_union_of_stars_feasible_solution(
                    instance,
                    remove_redundant_rows=True,
                    remove_rows_with_zero_weight=True,
                    choose_stars_greedily=False
                )

                # Try solving using CMIPGC formulation
                try:
                    cmipgc_model = CMIPGC(instance, None, start_p=p, start_w=w)
                    obj, vars = cmipgc_model.run(
                        relax=False,
                        time_limit=time_limit,
                        heuristics=0.005,
                        output_flag=0
                    )
                    gap = cmipgc_model.model.MIPGap
                    best_bound = cmipgc_model.model.ObjBound
                    runtime = cmipgc_model.model.Runtime
                    print("CMIPGC:", "obj", obj, "bound", best_bound, "gap", gap, "runtime", runtime)

                    cmipgc_results[filename] = {
                        "obj": obj,
                        "gap": gap,
                        "best_bound": best_bound,
                        "runtime": runtime
                    }
                except:
                    print("Exception for CMIPGC formulation in graph", filepath)

                # Try solving using Baseline formulation
                try:
                    baseline_model = Baseline(instance)
                    obj, vars = baseline_model.run(
                        time_limit=time_limit,
                        output_flag=0
                    )
                    best_bound = baseline_model.model.ObjBound
                    gap = baseline_model.model.MIPGap
                    runtime = baseline_model.model.Runtime
                    print("Baseline:", "obj", obj, "bound", best_bound, "gap", gap, "runtime", runtime)

                    baseline_results[filename] = {
                        "obj": obj,
                        "gap": gap,
                        "best_bound": best_bound,
                        "runtime": runtime
                    }
                except:
                    print("Exception for Baseline formulation in graph", filepath)

    # Convert CMIPGC results to DataFrame and save to CSV
    cmipgc_df = pd.DataFrame(cmipgc_results)
    cmipgc_df.to_csv('cmipgc_results.csv', index=True)

    # Convert Baseline results to DataFrame and save to CSV
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv('baseline_results.csv', index=True)
