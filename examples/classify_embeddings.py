"""
This is basic example of an openEO process graph.
It shows how to classify AEF embeddings to map crop types in Breizh, France.
This is the same as in examples/process_graphs/classify_embeddings.json
"""
import json
import os
from pathlib import Path

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")


from minibackend import execute_graph_dict

with open("examples/training_data/train_data.json") as file:
    geoms = json.load(file)


process_graph = {
    # init random forest
    "mlmclassrandomforest1": {
        "process_id": "mlm_class_random_forest",
        "arguments": {
            "max_variables": "onethird",
            "num_trees": 200,
            "seed": 42,
            "dimension": "embedding",
            "use_timeseries": False,
        },
    },
    # 2) datacube for training
    "load_embeddings_train": {
        "process_id": "load_embeddings",
        "arguments": {
            "url": "https://raw.githubusercontent.com/Open-EO/openeo-processes-dask-ml/refs/heads/main/examples/embeddings/Google_AlphaEarth_embeddings_stac_item.json",
            "spatial_extent": {
                "west": -4.02,
                "south": 48.20,
                "east": -3.74,
                "north": 48.30,
                "crs": 4326,
            },
            "temporal_extent": ["2017-01-01", "2017-12-31"],
        },
    },
    "aggregatespatial1": {
        "process_id": "aggregate_spatial",
        "arguments": {
            "data": {"from_node": "load_embeddings_train"},
            "geometries": geoms,
            "reducer": {
                "process_graph": {
                    "median1": {
                        "process_id": "median",
                        "arguments": {"data": {"from_parameter": "data"}},
                        "result": True,
                    }
                }
            },
        },
    },
    "mlfit1": {
        "process_id": "ml_fit",
        "arguments": {
            "model": {"from_node": "mlmclassrandomforest1"},
            "target": "class_name",
            "training_set": {"from_node": "aggregatespatial1"},
        },
    },
    "mlpredict1": {
        "process_id": "ml_predict",
        "arguments": {
            "data": {"from_node": "load_embeddings_train"},
            "model": {"from_node": "mlfit1"},
        },
    },
    "saveresult1": {
        "process_id": "save_result",
        "arguments": {
            "data": {"from_node": "mlpredict1"},
            "format": "GTiff",
            "options": {},
        },
        "result": True,
    },
}

# out = execute_graph_dict(process_graph)
# print(out)

import json

with open("examples/process_graphs/classify_embeddings.json", "w") as file:
    json.dump(process_graph, file)
