import networkx as nx
import random
import os

number_of_nodes = range(4, 21)  # range of number of nodes
number_of_graphs_per_size = 20


save_path = "./instances"

for i in number_of_nodes:
    for j in range(number_of_graphs_per_size):
        p = random.uniform(0, 1)
        G = nx.erdos_renyi_graph(i, p)
        filename = os.path.join(save_path, str(i) + "_" + str(j) + ".txt")
        with open(filename, "w") as f:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            f.write(f"{num_nodes} {num_edges}\n")

            if num_edges > 0:
                buffer = "\n".join(f"{u} {v}" for u, v in G.edges())
                f.write(buffer + "\n")

            f.write("\n")

print("graphs saved")
