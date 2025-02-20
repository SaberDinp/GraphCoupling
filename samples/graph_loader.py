import numpy as np


def load(file_name):
    """
    Reads a graph file and returns its adjacency matrix.

    Parameters:
    file_name (str): Path to the graph file.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    with open(file_name, "r") as f:
        # Read the first line: it contains the number of nodes and edges
        first_line = f.readline().strip()
        num_nodes, num_edges = map(int, first_line.split())

        # Initialize an empty adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Read the edges and fill the adjacency matrix
        for line in f:
            if line.strip():  # Ignore empty lines
                u, v = map(int, line.split())
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1  # Since the graph is undirected

    return adj_matrix


if __name__ == '__main__':
    adj = load("instances/20_10.txt")
    print(type(adj))
    print(type(adj.tolist()))
    print(adj.tolist())