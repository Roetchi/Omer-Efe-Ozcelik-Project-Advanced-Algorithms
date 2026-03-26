from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, TypedDict

class NodeData(TypedDict, total=False):
    label: str
    type: str
    lat: float
    lon: float
    address: str
    source: str

class EdgeData(TypedDict):
    source: str
    target: str
    road_factor: float
    road_name: str

class GraphData(TypedDict):
    city: str
    country: str
    nodes: Dict[str, NodeData]
    edges: List[EdgeData]
    scenarios: List[dict]

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_graph_data(path: Path | None = None) -> GraphData:
    if path is None:
        path = project_root() / "data" / "potsdam_graph.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def build_weighted_adjacency(graph_data: GraphData) -> dict[str, list[tuple[str, float, str]]]:
    nodes = graph_data["nodes"]
    adjacency: dict[str, list[tuple[str, float, str]]] = {node_id: [] for node_id in nodes}
    for edge in graph_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        source_data = nodes[source]
        target_data = nodes[target]
        base_distance = haversine_m(source_data["lat"], source_data["lon"], target_data["lat"], target_data["lon"])
        weight = base_distance * edge["road_factor"]
        adjacency[source].append((target, weight, edge["road_name"]))
        adjacency[target].append((source, weight, edge["road_name"]))
    return adjacency

def hospital_nodes(graph_data: GraphData) -> list[str]:
    return [node_id for node_id, node_data in graph_data["nodes"].items() if node_data["type"] == "hospital"]

def start_nodes(graph_data: GraphData) -> list[str]:
    return [scenario["start"] for scenario in graph_data["scenarios"]]
