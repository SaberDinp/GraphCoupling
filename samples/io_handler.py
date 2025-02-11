import itertools
import numpy as np
import os
import networkx as nx

"""Input/Output handler for input graphs"""
class IO:
    """Constructor"""
    def __init__(self):
        self.SAMPLES_DIRECTORY = "../samples"

    """Function to produce all of the non-isomorphic graphs up to 7 vertices"""
    def generate_non_isomorphic_graphs(self, max_vertices=7):
        os.makedirs(self.SAMPLES_DIRECTORY, exist_ok=True)

        for n in range(1, max_vertices + 1):
            num_edges = n * (n - 1) // 2  # Number of edges in upper triangular part
            seen_graphs = set()

            for index, bits in enumerate(itertools.product([0, 1], repeat=num_edges)):
                adj_matrix = np.zeros((n, n), dtype=int)
                upper_tri_indices = zip(*np.triu_indices(n, k=1))

                for (i, j), bit in zip(upper_tri_indices, bits):
                    adj_matrix[i, j] = bit
                    adj_matrix[j, i] = bit  # Ensure symmetry

                G = nx.from_numpy_array(adj_matrix)

                new_g = True
                for g in seen_graphs:
                    if nx.is_isomorphic(G, g):
                        new_g = False
                        break
                if new_g:
                    seen_graphs.add(G)
                    filename = os.path.join("", f"graph_{n}_vertices_{len(seen_graphs)}.txt")
                    np.savetxt(filename, adj_matrix, fmt="%d")

    def read_adjacency_matrix(self, filename):
        filepath = os.path.join("", filename)
        if os.path.exists(filepath):
            return np.loadtxt(filepath, dtype=int).tolist()
        else:
            raise FileNotFoundError(f"File {filename} not found in samples directory.")


if __name__ == '__main__':
