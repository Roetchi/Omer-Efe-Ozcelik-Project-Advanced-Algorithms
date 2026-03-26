from __future__ import annotations
from pathlib import Path
from evaluate import evaluate_project
from visualize import create_expansion_figure, create_graph_figure, create_runtime_figure

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    evaluate_project(output_dir)
    create_graph_figure(output_dir)
    create_runtime_figure(output_dir)
    create_expansion_figure(output_dir)
    print(f"Outputs written to: {output_dir}")

if __name__ == "__main__":
    main()
