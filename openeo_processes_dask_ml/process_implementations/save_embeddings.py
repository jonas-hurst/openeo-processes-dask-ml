import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import xarray as xr
from dask.delayed import Delayed, delayed
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.constants import (
    OPENEO_RESULTS_PATH,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils, zip_utils


def _get_stac_item_template(_id: str) -> dict:
    d = {
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/embeddings/v0.0.1/schema.json"
        ],
        "type": "Feature",
        "id": _id,
        "collection": None,
        "links": [{"rel": "self", "href": f"./{_id}.json"}],
        "bbox": None,  # will be set later,
        "geometry": None,  # will be set later,
        "properties": {
            "datetime": None,
            "start_datetime": None,
            "end_datetime": None,
            # "gsd": None,
            "title": "EO-Embeddings",
            "description": "EO embeddings produced using openeo-processes-dask-ml",
            "emb:type": None,  # will be set later
            "emb:dimensions": None,  # will be set later
            "emb:chip_layout": {"layout_type": None},
            "data_type": None,  # will be set later
        },
        "assets": {
            "embeddings": {
                "href": None,
                "title": "embeddings",
                "type": None,
                "roles": ["embedding"],
            }
        },
    }
    return d


def _save_as_zarr(datacube: xr.DataArray, result_dir: Path, zarr_dir: Path) -> Delayed:
    saved = datacube.to_zarr(zarr_dir, compute=False)
    zip_path = delayed(zip_utils.create_zip_archive)(
        result_dir, zarr_dir, "results.zarr.zip", saved
    )
    return zip_path


def _set_stac_spatial_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    x_dim, y_dim = dim_utils.get_spatial_dim_names(datacube)

    # todo: convert coords to wgs84
    xmin = float(min(datacube.coords[x_dim].data))
    ymin = float(min(datacube.coords[y_dim].data))
    xmax = float(max(datacube.coords[x_dim].data))
    ymax = float(max(datacube.coords[y_dim].data))
    bbox = [xmin, ymin, xmax, ymax]

    geom = {
        "type": "Polygon",
        "coordinates": [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ],
    }

    stac_metadata["bbox"] = bbox
    stac_metadata["geometry"] = geom


def _set_stac_time_metadata(stac_metadata: dict, datacube: xr.DataArray):
    try:
        time_dim = dim_utils.get_time_dim_name(datacube)
        if len(datacube.coords[time_dim]) > 1:
            tmin = min(datacube.coords[time_dim].data)
            tmax = max(datacube.coords[time_dim].data)
            stac_metadata["properties"]["start_datetime"] = str(tmin)
            stac_metadata["properties"]["end_datetime"] = str(tmax)
            del stac_metadata["properties"]["datetime"]
        else:
            t = datacube.coords[time_dim].data[0]
            stac_metadata["properties"]["datetime"] = str(t)
    except DimensionMissing:
        dt = str(datetime.now())
        stac_metadata["properties"]["datetime"] = dt
        del stac_metadata["properties"]["start_datetime"]
        del stac_metadata["properties"]["end_datetime"]


def _set_stac_embedding_metadata(stac_metadata: dict, datacube: xr.DataArray):
    emb_dim = dim_utils.get_embedding_dim_name(datacube)
    stac_metadata["properties"]["emb:type"] = "patch"
    stac_metadata["properties"]["emb:dimensions"] = len(datacube.coords[emb_dim].data)
    stac_metadata["properties"]["data_type"] = str(datacube.dtype)


def _set_stac_embedding_metadata_raster(stac_metadata: dict, datacube: xr.DataArray):
    _set_stac_embedding_metadata(stac_metadata, datacube)
    stac_metadata["properties"]["emb:chip_layout"]["layout_type"] = "regular_grid"


def _set_stac_embedding_asset_metadata_raster(
    stac_metadata: dict, out_path: Path
) -> dict:
    stac_metadata["assets"]["embeddings"]["href"] = str(out_path.absolute())
    stac_metadata["assets"]["embeddings"]["type"] = "application/vnd.zarr"
    return stac_metadata


def _update_stac_metadata_raster_cube(
    stac_metadata: dict, datacube: xr.DataArray, out_path: Path
):
    _set_stac_spatial_metadata_raster(stac_metadata, datacube)
    _set_stac_time_metadata(stac_metadata, datacube)
    _set_stac_embedding_metadata_raster(stac_metadata, datacube)


def _save_as_parquet(datacube: xr.DataArray, path: Path) -> bool:
    raise NotImplementedError("Saving of irregular embedding grids not implemented")


def _update_stac_metadata_vector_cube(stac_metadata: dict, datacube: xr.DataArray):
    pass


def _save_metadata_file(stac_metadata: dict, metadata_path: str) -> bool:
    try:
        with open(metadata_path, "w") as file:
            json.dump(stac_metadata, file, indent=4)
        return True
    except Exception as e:
        raise Exception("Failed saving the metadata file.")


def save_embeddings(data: xr.DataArray) -> Delayed:
    # you can call this method form your project-specific save-results process
    # if this method returns True, saving was successful, you can skip your own save-result code
    # if it returns False, saving was unsuccessful (i.e. no embeddings DC) and you can run your own save-result code

    if "embedding" not in data.dims:
        raise DimensionMissing(
            "Datacube does not contain an embedding dimension. It therefore can not "
            "be used in the save_embeddings process"
        )

    _id = str(uuid4())
    result_dir = Path(OPENEO_RESULTS_PATH) / _id
    zarr_out_path = result_dir / "result.zarr"
    metadata_path = result_dir / "result.json"

    stac_metadata = _get_stac_item_template(_id)

    spatial_dims = dim_utils.get_spatial_dim_names(data)
    if len(spatial_dims) == 2:
        # this implies embeddings in a regular raster -> save as zarr
        data.name = "embeddings"
        _update_stac_metadata_raster_cube(stac_metadata, data, result_dir)
        zipped_zarr_path = _save_as_zarr(data, result_dir, zarr_out_path)
        stac_metadata = delayed(_set_stac_embedding_asset_metadata_raster)(
            stac_metadata, zipped_zarr_path
        )

    if "geometry" in data.dims or "geom" in data.dims:
        # this implieds embeddings in irregular raster -> save as geo-parquet
        _update_stac_metadata_vector_cube(stac_metadata, data)
        _save_as_parquet(data, result_dir)

    saved = delayed(_save_metadata_file)(stac_metadata, metadata_path)
    return saved
