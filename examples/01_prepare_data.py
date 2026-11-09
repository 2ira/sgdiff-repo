"""Load the benchmark graphs and compare them with the paper's Table 3.

    python examples/01_prepare_data.py
"""

from sgdiff.api import available_datasets, describe_dataset
from sgdiff.data.datasets import load_graph

for name in available_datasets():
    if name == "router":
        print(f"{name:<10} skipped (manual download, see data/README.md)")
        continue
    if name == "synthetic":
        continue
    graph = load_graph(name, root="data")
    spec = describe_dataset(name)
    print(
        f"{name:<10} nodes={graph.num_nodes:<6} edges={graph.num_edges:<6} "
        f"features={graph.num_features:<5} | paper nodes={spec['num_nodes']:<6} "
        f"edges={spec['num_edges']:<6}"
    )
    print(f"           domain={spec['domain']}, auto_download={spec['auto_download']}")
