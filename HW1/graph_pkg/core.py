import networkx as nx
import numpy as np

class GraphTools:
    def __init__(self, edge_list=None, adjacency_matrix=None, weighted=False):

        self.is_weighted = weighted
        self.G = nx.Graph()

        if edge_list is not None:
            if weighted:
                self.G.add_weighted_edges_from(edge_list)
            else:
                self.G.add_edges_from(edge_list)

        elif adjacency_matrix is not None:
            adjacency_matrix = np.array(adjacency_matrix)
            size = adjacency_matrix.shape[0]
            self.G.add_nodes_from(range(size))

            for i in range(size):
                for j in range(i + 1, size):
                    if adjacency_matrix[i][j] != 0:
                        self.G.add_edge(i, j, weight=adjacency_matrix[i][j])
                        
        else:
            raise ValueError("Neither edge list nor adjacency matrix was provided.")

    def show_info(self):
        print("----- Graph Info -----")
        print(f"Weighted: {self.is_weighted}")
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        print(f"Number of edges: {self.G.number_of_edges()}")
