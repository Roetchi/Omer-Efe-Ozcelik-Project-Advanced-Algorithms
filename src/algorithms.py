from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush
from graph_model import GraphData, build_weighted_adjacency, haversine_m, hospital_nodes

@dataclass
class PathResult:
    algorithm: str
    start: str
    goal: str
    distance_m: float
    path: list[str]
    expanded_nodes: int

def reconstruct_path(parents: dict[str, str], end: str) -> list[str]:
    path = [end]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    return list(reversed(path))

def get_cached_adjacency(graph_data: GraphData) -> dict[str, list[tuple[str, float, str]]]:
    if "_adjacency_cache" not in graph_data:
        graph_data["_adjacency_cache"] = build_weighted_adjacency(graph_data)
    return graph_data["_adjacency_cache"]

def get_cached_hospitals(graph_data: GraphData) -> list[str]:
    if "_hospital_cache" not in graph_data:
        graph_data["_hospital_cache"] = hospital_nodes(graph_data)
    return graph_data["_hospital_cache"]

def build_nearest_hospital_heuristic(graph_data: GraphData) -> dict[str, float]:
    nodes = graph_data["nodes"]
    hospitals = get_cached_hospitals(graph_data)
    heuristic: dict[str, float] = {}
    for node_id, node_data in nodes.items():
        heuristic[node_id] = min(
            haversine_m(node_data["lat"], node_data["lon"], nodes[h]["lat"], nodes[h]["lon"])
            for h in hospitals
        )
    return heuristic

def get_cached_heuristic(graph_data: GraphData) -> dict[str, float]:
    if "_nearest_hospital_heuristic" not in graph_data:
        graph_data["_nearest_hospital_heuristic"] = build_nearest_hospital_heuristic(graph_data)
    return graph_data["_nearest_hospital_heuristic"]

def dijkstra_to_nearest_hospital(graph_data: GraphData, start: str) -> PathResult:
    adjacency = get_cached_adjacency(graph_data)
    hospital_set = set(get_cached_hospitals(graph_data))
    queue: list[tuple[float, str]] = [(0.0, start)]
    distances: dict[str, float] = {start: 0.0}
    parents: dict[str, str] = {}
    expanded_nodes = 0
    while queue:
        current_distance, current_node = heappop(queue)
        if current_distance != distances[current_node]:
            continue
        expanded_nodes += 1
        if current_node in hospital_set:
            return PathResult("Dijkstra", start, current_node, current_distance, reconstruct_path(parents, current_node), expanded_nodes)
        for neighbour, weight, _road_name in adjacency[current_node]:
            candidate = current_distance + weight
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                parents[neighbour] = current_node
                heappush(queue, (candidate, neighbour))
    raise ValueError(f"No hospital reachable from {start!r}")

def a_star_to_nearest_hospital(graph_data: GraphData, start: str) -> PathResult:
    adjacency = get_cached_adjacency(graph_data)
    hospital_set = set(get_cached_hospitals(graph_data))
    heuristic = get_cached_heuristic(graph_data)
    queue: list[tuple[float, float, str]] = [(heuristic[start], 0.0, start)]
    g_score: dict[str, float] = {start: 0.0}
    parents: dict[str, str] = {}
    expanded_nodes = 0
    while queue:
        _f_score, current_distance, current_node = heappop(queue)
        if current_distance != g_score[current_node]:
            continue
        expanded_nodes += 1
        if current_node in hospital_set:
            return PathResult("A*", start, current_node, current_distance, reconstruct_path(parents, current_node), expanded_nodes)
        for neighbour, weight, _road_name in adjacency[current_node]:
            candidate = current_distance + weight
            if candidate < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = candidate
                parents[neighbour] = current_node
                heappush(queue, (candidate + heuristic[neighbour], candidate, neighbour))
    raise ValueError(f"No hospital reachable from {start!r}")
