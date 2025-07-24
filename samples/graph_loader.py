import numpy as np

def load(file_name):
    """
    Reads a graph file and returns its adjacency matrix.

    Parameters:
    file_name (str): Path to the graph file. The expected format is:
                     - First line: "<number_of_nodes> <number_of_edges>"
                     - Subsequent lines: "<u> <v>" for each edge (0-based indexing)

    Returns:
    list of list of int: Adjacency matrix of the graph as a nested Python list.
    """
    with open(file_name, "r") as f:
        # Read the first line containing the number of nodes and edges
        first_line = f.readline().strip()
        num_nodes, num_edges = map(int, first_line.split())

        # Initialize a zero adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Populate the adjacency matrix from the list of edges
        for line in f:
            if line.strip():  # Skip any empty lines
                u, v = map(int, line.split())
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1  # Because the graph is undirected

    # Return the matrix as a regular Python list of lists
    return adj_matrix.tolist()


def get_path_graphs(n):
    """
    Generates the adjacency matrix of a path graph with n vertices.

    Parameters:
    n (int): Number of vertices.

    Returns:
    list of list of int: Adjacency matrix of the path graph.
    """
    return [[1 if abs(i - j) == 1 else 0 for j in range(n)] for i in range(n)]


def get_cycle_graphs(n):
    """
    Generates the adjacency matrix of a cycle graph with n vertices.

    Parameters:
    n (int): Number of vertices.

    Returns:
    list of list of int: Adjacency matrix of the cycle graph.
    """
    return [[1 if abs(abs(i - j) % n) in [1, n - 1] else 0 for j in range(n)] for i in range(n)]


def get_perfect_matching(n):
    """
    Generates the adjacency matrix of a perfect matching graph with n vertices.

    Parameters:
    n (int): Number of vertices. Must be even.

    Returns:
    numpy.ndarray: Adjacency matrix of the perfect matching graph.
    """
    if n % 2 != 0:
        raise ValueError("Number of vertices must be even for a perfect matching graph.")

    # Initialize a zero adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)

    # Match vertex i with i+1 in each pair (i = 0, 2, 4, ...)
    for i in range(0, n, 2):
        adj_matrix[i, i + 1] = 1
        adj_matrix[i + 1, i] = 1  # Symmetric entry for undirected edge

    return adj_matrix.tolist()


if __name__ == '__main__':
    # Example usage: Load and print the adjacency matrix of a saved graph instance
    adj = load("instances/20-1.txt")
    print("graph 20-1:")
    for item in adj:
        print(item)
    print("________")
    print("graph C_10")
    adj = get_cycle_graphs(10)
    for item in adj:
        print(item)
    print("________")
    print("graph P_10")
    adj = get_path_graphs(10)
    for item in adj:
        print(item)
    print("________")
    print("graph PM_10")
    adj = get_perfect_matching(10)
    for item in adj:
        print(item)