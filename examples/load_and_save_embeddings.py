"""
This is a very basic example of an openEO process graph.
It shows how to load Google AEF embeddings with load_embeddings with a spatial and
temporal bbox, and then stores them with the save_embeddings process.
"""
import os
from pathlib import Path

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")

from minibackend import execute_graph_dict

process_graph = {
    "load_embeddings": {
        "process_id": "load_embeddings",
        "arguments": {
            "url": "https://raw.githubusercontent.com/Open-EO/openeo-processes-dask-ml/refs/heads/main/examples/embeddings/Google_AlphaEarth_embeddings_stac_item.json",
            "spatial_extent": {"west": 8.2, "east": 8.5, "south": 48.9, "north": 49.1},
            "temporal_extent": ["2024-01-01", "2024-12-31"],
            "asset_name": "embeddings",
        },
        # "result": True
    },
    "save_embeddings": {
        "process_id": "save_embeddings",
        "arguments": {"data": {"from_node": "load_embeddings"}},
        "result": True,
    },
}

x = execute_graph_dict(process_graph)
print(x)

x = x.compute()
print(x)
