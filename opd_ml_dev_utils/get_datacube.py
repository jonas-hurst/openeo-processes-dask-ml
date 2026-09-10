"""
Some helper function to obtain datacubes (helpful for development)
"""

import hashlib
import os
import pickle
from typing import Optional, Union

import pystac_client
import stackstac
import xarray as xr
from dask import array as da
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from openeo_processes_dask.process_implementations.cubes import load_stac

from openeo_processes_dask_ml.process_implementations.constants import (
    DATACUBE_CACHE_DIR,
)


def _secure_hash_objects(*args):
    """Computes a stable, cryptographic hash from arbitrary objects."""
    hasher = hashlib.sha256()
    for obj in args:
        # Use repr() to get a more stable string representation than str()
        obj_bytes = repr(obj).encode("utf-8")
        hasher.update(obj_bytes)
    return hasher.hexdigest()


def _write_datacube_to_cache(datacube: xr.DataArray, path: str):
    if not os.path.exists(DATACUBE_CACHE_DIR):
        os.makedirs(DATACUBE_CACHE_DIR)
    with open(path, "wb") as file:
        pickle.dump(datacube, file)


def get_random_datacube(shape: tuple[int, ...], dims: tuple[str, ...]) -> xr.DataArray:
    if len(shape) != len(dims):
        raise ValueError("Length of shape and dim attributes must be the same")

    coords = {dim_name: range(dim_len) for dim_name, dim_len in zip(dims, shape)}

    dc = xr.DataArray(da.random.random(shape), dims=dims, coords=coords)

    return dc


def get_datacube_from_pickle_file(path: str) -> xr.DataArray:
    with open(path, "rb") as file:
        dc = pickle.load(file)
    if not isinstance(dc, xr.DataArray):
        raise TypeError("The provided file is not an xarray DataArray")
    return dc


def get_datacube_from_stackstac(
    bbox: tuple[float, float, float, float],
    time_period: str,  # Format: 'YYYY-MM-DD/YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ'
    collection: str = "sentinel-2-l2a",
    stac_url: str = "https://earth-search.aws.element84.com/v1",
    assets: list[str] = ["nir", "red"],  # Blue, Green, Red, NIR
    epsg: int = None,  # Target EPSG code, defaults to the most common for the area
    resolution: int = 10,  # Target resolution in meters
    cloud_cover_max: float = None,  # Maximum cloud cover percentage (0-100)
    chunksize: int = 2048,  # Spatial chunk size for Dask (useful for larger areas/times)
) -> xr.DataArray:
    """
    Forms an xarray datacube of Sentinel-2 images for a given bounding box
    and time period from the Element 84 STAC catalog using stackstac.

    Args:
        bbox (list[float]): Bounding box as [minx, miny, maxx, maxy].
        time_period (str): Time range as 'YYYY-MM-DD/YYYY-MM-DD'.
        collection (str): The STAC collection ID (default: 'sentinel-2-l2a').
        stac_url (str): The URL of the STAC catalog (default: Element 84).
        assets (list[str]): List of band names (asset keys) to include.
                             Defaults to B02, B03, B04 (10m visible) and B08 (10m NIR).
        epsg (int): The target EPSG code for the output datacube. If None,
                    stackstac attempts to find a suitable UTM zone.
        resolution (int): The target spatial resolution in meters.
        cloud_cover_max (float): Maximum percentage of cloud cover allowed for items.
        chunksize (int): Spatial chunk size for Dask. Set to None for no chunking
                         (loads everything into memory).

    Returns:
        xr.DataArray: An xarray DataArray containing the stacked Sentinel-2 imagery,
                      or None if no items were found.
    """

    hash_val = _secure_hash_objects(
        bbox,
        time_period,
        collection,
        stac_url,
        assets,
        epsg,
        resolution,
        cloud_cover_max,
        chunksize,
    )

    filename = "stackstac_" + hash_val + ".pickle"
    path = os.path.join(DATACUBE_CACHE_DIR, filename)

    if os.path.exists(path):
        return get_datacube_from_pickle_file(path)
    else:
        client = pystac_client.Client.open(stac_url)
        search_params = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": time_period,
            "query": {},
        }
        if cloud_cover_max is not None:
            search_params["query"]["eo:cloud_cover"] = {"lt": cloud_cover_max}
        search = client.search(**search_params)

        # Get the item collection (lazy loading)
        item_collection = search.item_collection()

        if not item_collection:
            raise Exception("No items found for the specified criteria.")

        dc_lazy = stackstac.stack(
            item_collection,
            assets=assets,
            epsg=epsg,
            resolution=resolution,
            chunksize=chunksize,
            bounds_latlon=bbox,
        )

        dc = dc_lazy.compute()
        _write_datacube_to_cache(dc, path)

        return dc


def load_stac_without_cache(
    url: str,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    bands: list[str] | None = None,
    properties: dict | None = None,
    resolution: float | None = None,
    projection: int | str | None = None,
    resampling: str | None = None,
) -> xr.DataArray:
    if "https://stac.dataspace.copernicus.eu/v1" in url:
        # set CDSE AWS S3 envs
        os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
        os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
        os.environ["AWS_HTTPS"] = "YES"
        os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
        os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
        cdse_key = os.environ.get("CDSE_S3_ACCESS_KEY")
        cdse_key_secret = os.environ.get("CDSE_S3_SECRET_KEY")

        if cdse_key is None or cdse_key_secret is None:
            raise Exception(
                "CDSE Credentials are missing. "
                "Set them using ENVs CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY"
            )

        os.environ["AWS_ACCESS_KEY_ID"] = cdse_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = cdse_key_secret

        collection_id = url.split("/")[-1]

        if collection_id == "sentinel-2-l2a":
            band_conversion = {
                "coastal": "B01_20m",
                "blue": "B02_10m",
                "green": "B03_10m",
                "red": "B04_10m",
                "rededge1": "B05_20m",
                "rededge2": "B06_20m",
                "rededge3": "B07_20m",
                "nir": "B08_10m",
                "nir08": "B8A_20m",
                "nir09": "B09_60m",
                "swir16": "B11_20m",
                "swir22": "B12_20m",
            }
        elif collection_id == "sentinel-2-global-mosaics":
            band_conversion = {
                "blue": "B02",
                "green": "B03",
                "red": "B04",
                "nir": "B08",
            }
        else:
            band_conversion = {}

        old_bands = bands
        bands = [band_conversion[b] if b in band_conversion else b for b in bands]
    elif "https://planetarycomputer.microsoft.com/api/stac/v1" in url:
        band_conversion = {
            "coastal": "B01",
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "rededge1": "B05",
            "rededge2": "B06",
            "rededge3": "B07",
            "nir": "B08",
            "nir08": "B8A",
            "nir09": "B09",
            "swir16": "B11",
            "swir22": "B12",
            "scl": "SCL",
        }
        old_bands = bands
        bands = [band_conversion[b] if b in band_conversion else b for b in bands]
    else:
        old_bands = None

    dc_lazy = load_stac(
        url,
        spatial_extent,
        temporal_extent,
        bands,
        properties,
        resolution,
        projection,
        resampling,
    )

    if old_bands is not None:
        dc_lazy.coords["bands"] = old_bands

    return dc_lazy


def load_stac_with_cache(
    url: str,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    bands: list[str] | None = None,
    properties: dict | None = None,
    resolution: float | None = None,
    projection: int | str | None = None,
    resampling: str | None = None,
) -> xr.DataArray:
    hash_val = _secure_hash_objects(
        url,
        spatial_extent,
        temporal_extent,
        bands,
        properties,
        resolution,
        projection,
        resampling,
    )

    filename = hash_val + ".pickle"
    path = os.path.join(DATACUBE_CACHE_DIR, filename)

    if os.path.exists(path):
        dc = get_datacube_from_pickle_file(path)
        dc = dc.chunk({"time": 1, "bands": -1, "x": 224, "y": 224})
        return dc
    else:
        dc_lazy = load_stac_without_cache(
            url,
            spatial_extent,
            temporal_extent,
            bands,
            properties,
            resolution,
            projection,
            resampling,
        )
        dc = dc_lazy.compute()

        _write_datacube_to_cache(dc, path)

        return dc
