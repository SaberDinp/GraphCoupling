# This script generates random Erdős–Rényi graphs of varying sizes,
# which are used as input instances for our experimental evaluations.
# For each graph size (number of nodes), it creates multiple instances
# with randomly chosen edge probabilities, and saves them in text files
# under the './instances' directory.


# Import necessary libraries
import networkx as nx  # NetworkX for graph creation and manipulation
import random  # For generating random probabilities
import os  # For handling file paths and directories

# Define the range of graph sizes (number of nodes)
number_of_nodes = range(4, 21)  # Generates graphs with node counts from 4 to 20

# Define how many graphs to generate per graph size
number_of_graphs_per_size = 2

# Directory where the generated graph instances will be saved
save_path = "./instances2"

# Loop over each desired graph size
for i in number_of_nodes:
    # Generate multiple graphs of the same size
    for j in range(number_of_graphs_per_size):
        # Randomly choose the edge probability for the Erdős–Rényi model
        p = random.uniform(0, 1)

        # Generate an Erdős–Rényi random graph G(n=p, p)
        G = nx.erdos_renyi_graph(i, p)

        # Create the output filename (e.g., "4-1.txt", "4-2.txt", ...)
        filename = os.path.join(save_path, str(i) + "-" + str(j + 1) + ".txt")

        # Write graph data to file
        with open(filename, "w") as f:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()

            # Write the number of nodes and edges as the first line
            f.write(f"{num_nodes} {num_edges}\n")

            # Write each edge as a pair of node indices (one edge per line)
            if num_edges > 0:
                buffer = "\n".join(f"{u} {v}" for u, v in G.edges())
                f.write(buffer + "\n")

            # Write an extra newline at the end (could be useful for parsing)
            f.write("\n")

# Notify user that graph generation and saving is complete
print("graphs saved")

