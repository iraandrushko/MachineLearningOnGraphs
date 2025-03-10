import networkx as nx
from collections import deque

def dfs(graph, start):
    visited = set()
    order = []

    def _dfs(v):
        visited.add(v)
        order.append(v)
        for neighbor in graph.neighbors(v):
            if neighbor not in visited:
                _dfs(neighbor)
    
    _dfs(start)
    return order

def bfs(graph, start):
    visited = set([start])
    order = []
    queue = deque([start])
    
    while queue:
        v = queue.popleft()
        order.append(v)
        for neighbor in graph.neighbors(v):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

def find_shortest_path(graph, source, target, weight=None):
    try:
        return nx.shortest_path(graph, source=source, target=target, weight=weight)
    except nx.NetworkXNoPath:
        return None

def find_all_paths(graph, source, target, path_length):
    def _find_paths(current, target, path, all_paths):

        if len(path) > path_length + 1:
            return
        if current == target and len(path) == path_length + 1:
            all_paths.append(path.copy())
            return
        for neighbor in graph.neighbors(current):
            if neighbor not in path:
                path.append(neighbor)
                _find_paths(neighbor, target, path, all_paths)
                path.pop()
    
    all_paths = []
    _find_paths(source, target, [source], all_paths)
    return all_paths

def detect_cycles(graph):
    cycles = nx.cycle_basis(graph)
    if len(cycles) == 0:
        return None
    else:
        return cycles
