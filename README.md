# Graph Coupling Problem: MIP Formulations and Heuristics

This project provides code and experiments for solving the Graph Coupling Problem using mixed-integer programming (MIP) formulations, including a new compact formulation and heuristic warm-starts.

---

## 📁 Project Structure

### `samples/` — Data Generation and Loading
This directory contains everything related to graph instance generation and loading.

- **`samples/instances/`**  
  Contains the Erdős–Rényi graph instances used in experiments.

- **`samples/graph_generator.py`**  
  Generates random Erdős–Rényi graphs and saves them as `.txt` files.

- **`samples/graph_loader.py`**  
  Reads a given graph file and returns its adjacency matrix.

---

### `models/` — Solving Methods
This directory contains different MIP formulations and heuristics for solving the Graph Coupling Problem.

- **`models/baseline_MIP.py`**  
  Implements the MIP formulation proposed by Rajakumar et al. (Phys. Rev. A, 2022).

- **`models/CMIPGC.py`**  
  Contains the new compact MIP formulation proposed in this project.

- **`models/warm_start.py`**  
  Implements two heuristic constructions:
  - **Union of Stars**: from Rajakumar et al.
  - **Union of Double-Stars**: our novel method

---

### `experiments/tests.py` — Experimental Evaluation
Runs both MIP formulations (`baseline_MIP.py` and `CMIPGC.py`) on all graph instances in `samples/instances` using a time limit of **3600 seconds** per instance.

Outputs:
- `experiments/baseline_results.csv`  
- `experiments/cmipgc_results.csv`  

Each CSV file stores the objective value, best bound, MIP gap, and runtime for each instance and formulation.

---

## 📝 Citation

If you use this codebase in your research, please consider citing our work (citation to be added).

---

## 🔧 Requirements

All required Python packages, along with their specific versions, are listed in the `requirements.txt` file.

> **Note**: Gurobi is a commercial solver and requires a valid license. Please refer to the [Gurobi website](https://www.gurobi.com/documentation/) for installation instructions and license setup.


Feel free to open issues or contribute if you'd like to extend the project!
