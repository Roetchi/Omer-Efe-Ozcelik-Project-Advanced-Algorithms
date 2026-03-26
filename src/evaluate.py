from __future__ import annotations
import os, time, tracemalloc
from pathlib import Path
import networkx as nx
import pandas as pd
from algorithms import a_star_to_nearest_hospital, dijkstra_to_nearest_hospital
from graph_model import build_weighted_adjacency, hospital_nodes, load_graph_data

def build_networkx_graph(graph_data: dict) -> nx.Graph:
    graph = nx.Graph()
    for node_id, node_data in graph_data["nodes"].items():
        graph.add_node(node_id, **node_data)
    adjacency = build_weighted_adjacency(graph_data)
    for source, neighbours in adjacency.items():
        for target, weight, road_name in neighbours:
            if graph.has_edge(source, target):
                continue
            graph.add_edge(source, target, weight=weight, road_name=road_name)
    return graph

def benchmark_runtime_ms(function, graph_data: dict, start: str, repeats: int = 12000) -> tuple[float, object]:
    for _ in range(150):
        function(graph_data, start)
    result = None
    start_time_ns = time.perf_counter_ns()
    for _ in range(repeats):
        result = function(graph_data, start)
    elapsed_ms = (time.perf_counter_ns() - start_time_ns) / repeats / 1_000_000
    return elapsed_ms, result

def benchmark_peak_kib(function, graph_data: dict, start: str) -> tuple[float, object]:
    tracemalloc.start()
    result = function(graph_data, start)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024, result

def atomic_to_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.csv")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)

def evaluate_project(output_dir: Path | None = None) -> pd.DataFrame:
    graph_data = load_graph_data()
    graph = build_networkx_graph(graph_data)
    hospitals = hospital_nodes(graph_data)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_distances = nx.floyd_warshall(graph, weight="weight")
    rows: list[dict] = []
    for scenario in graph_data["scenarios"]:
        start = scenario["start"]
        oracle_goal = min(hospitals, key=lambda hospital: oracle_distances[start][hospital])
        oracle_distance = oracle_distances[start][oracle_goal]
        for name, function in (("Dijkstra", dijkstra_to_nearest_hospital), ("A*", a_star_to_nearest_hospital)):
            runtime_ms, timing_result = benchmark_runtime_ms(function, graph_data, start)
            peak_kib, _memory_result = benchmark_peak_kib(function, graph_data, start)
            rows.append({
                "scenario_node": start,
                "scenario": graph_data["nodes"][start]["label"],
                "scenario_description": scenario["description"],
                "algorithm": name,
                "nearest_hospital": graph_data["nodes"][timing_result.goal]["label"],
                "distance_m": round(timing_result.distance_m, 1),
                "runtime_ms": runtime_ms,
                "peak_kib": peak_kib,
                "expanded_nodes": timing_result.expanded_nodes,
                "oracle_hospital": graph_data["nodes"][oracle_goal]["label"],
                "oracle_distance_m": round(oracle_distance, 1),
                "hospital_matches_oracle": timing_result.goal == oracle_goal,
                "distance_matches_oracle": abs(timing_result.distance_m - oracle_distance) < 1e-6,
                "path": " → ".join(graph_data["nodes"][node]["label"] for node in timing_result.path),
            })
    results = pd.DataFrame(rows)
    atomic_to_csv(results, output_dir / "benchmark_results.csv")
    summary = (
        results.groupby("algorithm", as_index=False)
        .agg(avg_runtime_ms=("runtime_ms", "mean"), avg_peak_kib=("peak_kib", "mean"), avg_expanded_nodes=("expanded_nodes", "mean"))
        .sort_values("avg_runtime_ms")
    )
    atomic_to_csv(summary, output_dir / "benchmark_summary.csv")
    return results

if __name__ == "__main__":
    dataframe = evaluate_project()
    pd.set_option("display.max_columns", None)
    print(dataframe.to_string(index=False))
