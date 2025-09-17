import argparse
import os
from pathlib import Path

from minibackend import execute_graph_dict

if Path.cwd().resolve().name == "examples":
    os.chdir("..")
if Path.cwd().resolve().name != "openeo-processes-dask-ml":
    raise Exception("Current CWD is not the Project root (openeo-processes-dask-ml)")


def run_openeo_ml_predict(model_url: str):
    process_graph = {
        "process_graph": {
            "load_data": {
                "process_id": "load_stac",
                "arguments": {
                    "url": "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
                    "spatial_extent": {
                        "west": 8.2,
                        "east": 8.5,
                        "south": 48.9,
                        "north": 49.1,
                    },
                    "temporal_extent": ["2024-08-05", "2024-08-12"],
                    "bands": [
                        "coastal",
                        "blue",
                        "green",
                        "red",
                        "rededge1",
                        "rededge2",
                        "rededge3",
                        "nir",
                        "nir08",
                        "nir09",
                        "swir16",
                        "swir22",
                    ],
                    "resolution": 10,
                },
            },
            "load_model": {
                "process_id": "load_ml_model",
                "arguments": {
                    "uri": model_url,
                    "model_asset": "weights",
                },
            },
            "predict": {
                "process_id": "ml_predict",
                "arguments": {
                    "data": {"from_node": "load_data"},
                    "model": {"from_node": "load_model"},
                },
                "result": True,
            },
        },
        "parameters": [],
    }

    out_datacube = execute_graph_dict(process_graph)  # output datacube is lazy
    out_datacube = out_datacube.compute()  # compute the datacube
    print(out_datacube)


if __name__ == "__main__":
    # call this script like this:
    # python make_embeddings.py --uri https://link-to-stac/mlm-item.json

    parser = argparse.ArgumentParser(
        prog="MLonDatacube", description="Script to apply an ML model on a datacube."
    )

    parser.add_argument(
        "--uri",
        default="./examples/mlm_items/prithvi_v2_item.json",
        type=str,
        help="URI to the STAC-MLM file that catalogs the ML model to apply to the datacube.",
    )
    args = parser.parse_args()
    uri = args.uri
    run_openeo_ml_predict(uri)
