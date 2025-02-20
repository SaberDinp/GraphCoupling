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

    return adj_matrix.tolist()

def get_path_graphs(n):
    return [[1 if abs(i - j) == 1 else 0 for j in range(n)] for i in range(n)]

def get_cycle_graphs(n):
    return [[1 if abs(abs(i - j) % n) in [1, n-1] else 0 for j in range(n)] for i in range(n)]


def get_perfect_matching(n):
    """
    Generates the adjacency matrix of a perfect matching graph with n vertices.

    Parameters:
    n (int): Number of vertices in the graph. Must be even.

    Returns:
    numpy.ndarray: Adjacency matrix of the perfect matching graph.
    """
    if n % 2 != 0:
        raise ValueError("Number of vertices must be even for a perfect matching graph.")

    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)

    # Connect vertex i to i+1 in pairs (1-2, 3-4, ...)
    for i in range(0, n, 2):
        adj_matrix[i, i + 1] = 1
        adj_matrix[i + 1, i] = 1  # Since the graph is undirected

    return adj_matrix


if __name__ == '__main__':
    adj = load("instances/20_10.txt")
    print(type(adj))
    print(adj)