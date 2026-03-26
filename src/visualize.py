from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from graph_model import build_weighted_adjacency, load_graph_data

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

def create_graph_figure(output_dir: Path) -> Path:
    graph_data = load_graph_data()
    graph = build_networkx_graph(graph_data)
    positions = {node_id: (node_data["lon"], node_data["lat"]) for node_id, node_data in graph_data["nodes"].items()}
    pretty_labels = {
        "potsdam_hbf": "Central\nStation","alter_markt": "Alter\nMarkt","ernst_von_bergmann": "Ernst von\nBergmann",
        "dutch_quarter": "Dutch\nQuarter","nauener_tor": "Nauener\nTor","brandenburg_gate": "Brandenburg\nGate",
        "st_josef": "St. Josefs\nHospital","sanssouci_palace": "Sanssouci\nPalace","schiffbauergasse": "Schiffbauergasse",
        "glienicke_bridge": "Glienicke\nBridge","babelsberg_park": "Babelsberg\nPark","oberlinklinik": "Oberlin-\nklinik",
    }
    plt.figure(figsize=(10, 6.8))
    nx.draw_networkx_edges(graph, positions, width=1.7)
    hospitals = [node for node, data in graph.nodes(data=True) if data["type"] == "hospital"]
    starts = [node for node, data in graph.nodes(data=True) if data["type"] == "start"]
    landmarks = [node for node, data in graph.nodes(data=True) if data["type"] == "landmark"]
    nx.draw_networkx_nodes(graph, positions, nodelist=landmarks, node_size=260, node_shape="o", label="Landmark")
    nx.draw_networkx_nodes(graph, positions, nodelist=starts, node_size=330, node_shape="o", label="Scenario start")
    nx.draw_networkx_nodes(graph, positions, nodelist=hospitals, node_size=420, node_shape="s", label="Hospital")
    label_offsets = {
        "potsdam_hbf": (0.0000, -0.00065), "alter_markt": (0.0000, -0.00075), "ernst_von_bergmann": (0.0009, 0.0005),
        "dutch_quarter": (0.0003, 0.00075), "nauener_tor": (0.0002, 0.00075), "brandenburg_gate": (-0.0012, -0.0004),
        "st_josef": (-0.0010, -0.0005), "sanssouci_palace": (-0.0002, 0.0008), "schiffbauergasse": (0.0010, 0.00015),
        "glienicke_bridge": (0.0008, 0.00075), "babelsberg_park": (0.0010, -0.0004), "oberlinklinik": (0.0011, -0.0001),
    }
    label_positions = {node: (positions[node][0] + label_offsets[node][0], positions[node][1] + label_offsets[node][1]) for node in positions}
    nx.draw_networkx_labels(graph, label_positions, labels=pretty_labels, font_size=8)
    plt.title("Curated Potsdam graph used for hospital pathfinding")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(frameon=True, loc="lower left")
    plt.tight_layout()
    path = output_dir / "graph_overview.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path

def create_runtime_figure(output_dir: Path) -> Path:
    results = pd.read_csv(output_dir / "benchmark_results.csv")
    pivot = results.pivot(index="scenario", columns="algorithm", values="runtime_ms")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Average runtime per run (ms)")
    ax.set_xlabel("Scenario")
    ax.set_title("Dijkstra vs A* average runtime")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    path = output_dir / "runtime_comparison.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path

def create_expansion_figure(output_dir: Path) -> Path:
    results = pd.read_csv(output_dir / "benchmark_results.csv")
    pivot = results.pivot(index="scenario", columns="algorithm", values="expanded_nodes")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Expanded nodes")
    ax.set_xlabel("Scenario")
    ax.set_title("Search effort by scenario")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    path = output_dir / "node_expansions.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return path

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    create_graph_figure(out)
    create_runtime_figure(out)
    create_expansion_figure(out)
