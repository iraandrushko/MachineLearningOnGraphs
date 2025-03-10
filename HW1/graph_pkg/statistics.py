import networkx as nx
import numpy as np
from scipy.sparse import csgraph

def compute_degree_matrix(graph):
    n = graph.number_of_nodes()
    degrees = [val for (_, val) in graph.degree()]
    D = np.diag(degrees)
    return D

def compute_adjacency_matrix(graph):
    A = nx.to_numpy_array(graph)
    return A

def compute_laplacian(graph, normalized=False):
    if normalized:
        L = nx.normalized_laplacian_matrix(graph).toarray()
    else:
        L = nx.laplacian_matrix(graph).toarray()
    return np.array(L)

def compute_eigens(graph, matrix_type='adjacency'):
    if matrix_type == 'adjacency':
        M = compute_adjacency_matrix(graph)
    elif matrix_type == 'laplacian':
        M = compute_laplacian(graph)
    else:
        raise ValueError("matrix_type has to be 'adjacency' or 'laplacian'")
    
    eigenvalues, eigenvectors = np.linalg.eig(M)
    return eigenvalues, eigenvectors

def compute_local_overlap(graph):
    overlap = {}
    for u, v in graph.edges():
        neighbors_u = set(graph.neighbors(u)) - {v}
        neighbors_v = set(graph.neighbors(v)) - {u}
        common = neighbors_u.intersection(neighbors_v)
        overlap[(u, v)] = len(common)
    return overlap
