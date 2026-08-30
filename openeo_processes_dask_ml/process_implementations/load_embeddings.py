import hashlib
import json
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as Datetime
from pathlib import Path
from urllib.parse import urlparse

import dask
import dask_geopandas
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pyproj
import pystac
import pystac_client
import rioxarray  # dont remove (even if your IDE says it is unused)
import shapely
import xarray as xr
import xvec  # dont remove (even if your IDE says it is unused)
from dask import array as da
from dask.delayed import delayed
from numpy.typing import NDArray
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from openeo_processes_dask.process_implementations.cubes._filter import _reproject_bbox
from openeo_processes_dask.process_implementations.exceptions import (
    DimensionMissing,
    OpenEOException,
)

from openeo_processes_dask_ml.process_implementations import constants
from openeo_processes_dask_ml.process_implementations.utils import (
    dim_utils,
    download_utils,
    stac_utils,
    zarr_utils,
    zip_utils,
)


def _get_item_time(stac_item: pystac.Item) -> Datetime:
    """
    Extracts the time from a STAC item. Returns datetime property if present,
    and start_datetime if datetime is not present
    :param stac_item: The pystac.Item object
    :return: datetime object
    """
    if isinstance(stac_item.datetime, Datetime):
        return stac_item.datetime

    dt_start = stac_item.properties.get("start_datetime")
    if isinstance(dt_start, str):
        start = Datetime.fromisoformat(dt_start)
        return start

    # in theory this should never occur as it would violate the STAC spec
    raise ValueError("Could not determine the item's datetime")


def _match_geom_in_list(
    geometry_list: list[shapely.geometry.base.BaseGeometry],
    geometry: shapely.geometry.base.BaseGeometry,
    tolerance: float,
) -> int | None:
    """
    Check weather a geometry is already present in a list of geometries, considering a
    coordinate matching tolerance to accomate for float rounding errors
    :param geometry_list: List of NORMALIZED geometries to check
    :param geometry: the NORMALIZED geometry to check
    :param tolerance: Tolerance to check
    :return: index of polygon in list, None if polygon is not present in list
    """
    for i, p in enumerate(geometry_list):
        if p.equals_exact(geometry, tolerance=tolerance):
            return i
    return None


def _monotonic_bounds_slice(coord_values, lo, hi):
    """
    Return a contiguous slice selecting coords within [lo, hi],
    working for both ascending and descending coordinates.
    """
    coord_values = np.asarray(coord_values)

    # normalize the requested range
    if lo > hi:
        lo, hi = hi, lo

    mask = (coord_values >= lo) & (coord_values <= hi)
    idx = np.flatnonzero(mask)

    if idx.size == 0:
        return None  # no overlap

    # idx is contiguous for monotonic coords (either direction)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _snap_slice_to_chunks(
    slc: slice | None,
    dim_size: int,
    chunks: int | Sequence[int] | None,
) -> slice | None:
    """Expand a slice outwards so that it covers only *whole* chunks.

    Parameters
    ----------
    slc
        The slice to expand (``step`` must be ``None`` or ``1``).
    dim_size
        Length of the dimension the slice refers to.
    chunks
        Either a single chunk length (uniform chunking, as reported by
        ``encoding["preferred_chunks"]``) or a sequence of per-chunk lengths
        (as reported by ``DataArray.chunksizes``).

    Returns
    -------
    slice
        A slice whose ``start`` sits on a chunk boundary and whose ``stop``
        sits on a chunk boundary (or at ``dim_size``), and which always
        contains the original slice.
    """
    if slc is None or not chunks:
        return slc

    start, stop, step = slc.indices(dim_size)  # normalises None / negatives
    if step != 1:
        raise ValueError("Chunk snapping is only supported for contiguous slices.")
    if start >= stop:  # empty selection -> nothing to snap
        return slice(start, stop)

    if isinstance(chunks, (int, np.integer)):
        # ---- uniform chunking ------------------------------------------
        chunk = int(chunks)
        if chunk <= 0:
            return slice(start, stop)
        new_start = (start // chunk) * chunk
        new_stop = min(((stop + chunk - 1) // chunk) * chunk, dim_size)
    else:
        # ---- irregular chunking (e.g. dask chunksizes) -------------------
        bounds = np.concatenate(([0], np.cumsum(np.asarray(chunks, dtype=np.int64))))
        # largest boundary <= start
        i = int(np.searchsorted(bounds, start, side="right")) - 1
        # smallest boundary >= stop
        j = int(np.searchsorted(bounds, stop, side="left"))
        new_start = int(bounds[max(i, 0)])
        new_stop = int(min(bounds[min(j, len(bounds) - 1)], dim_size))

    return slice(new_start, new_stop)


def _load_zarr(
    path: str, bbox: BoundingBox | None, temporal_extent: TemporalInterval
) -> xr.DataArray:
    """
    Load a zarr store into an xarray.DataArray.

    The `path` may point to a local file or a remote URL. If it ends in
    ".zip" the archive is (downloaded and) extracted into the cache dir
    (``constants.DATA_CACHE_DIR``). The `path` acts as a stable ID: if the
    corresponding extracted store already exists in the cache it is reused
    without re-downloading or re-extracting.
    """
    cache_dir = Path(constants.DATA_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if zip_utils.is_zip(path):
        store_path = _get_extracted_store(path, cache_dir)
    else:
        # Not a zip: open directly (local path or remote store).
        store_path = path

    ds = zarr_utils.open_zarr_auto(store_path)

    # inspect for CRS:
    crs = ds.rio.crs
    if crs is not None:
        if "spatial_ref" not in ds.coords:
            ds.rio.write_crs(crs, inplace=True)
    else:
        x_dim, y_dim = dim_utils.get_spatial_dim_names(ds)
        x_coords = ds.coords[x_dim].values
        y_coords = ds.coords[y_dim].values

        # check if x and y coords could be WGS-84 geogrpahic coords
        x_is_between = np.all((x_coords >= -181) & (x_coords <= 181))
        y_is_between = np.all((y_coords >= -91) & (y_coords <= 91))

        if x_is_between and y_is_between:
            warnings.warn(
                "Could not detect a CRS in the zarr store. Assuming EPSG:4326"
            )
            crs = None
            ds.rio.write_crs("EPSG:4326", inplace=True)
        else:
            raise Exception("Could not determine CRS of data in zarr store.")

    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        var_name = data_vars[0]
    elif "embeddings" in ds.data_vars:
        var_name = "embeddings"
    elif "embedding" in ds.data_vars:
        var_name = "embedding"
    else:
        raise KeyError(
            f"Store has multiple variables {data_vars}, and no 'embedding' or "
            f"'embeddings' variable."
        )

    embedding_datacube = ds[var_name]

    preferred_chunks = embedding_datacube.encoding.get("preferred_chunks")
    # e.g. {'time': 1, 'band': 64, 'y': 256, 'x': 256}

    # rename band dimension to embedding dimension
    try:
        band_dim = dim_utils.get_band_dim_name(embedding_datacube)
        embedding_datacube = embedding_datacube.rename({band_dim: "embedding"})
        preferred_chunks["embedding"] = preferred_chunks[band_dim]
        del preferred_chunks[band_dim]
    except DimensionMissing:
        pass

    # make proper temporal dimension
    try:
        time_dim = dim_utils.get_time_dim_name(embedding_datacube)
    except DimensionMissing:
        time_dim = None
    if time_dim is not None:
        time_coords = embedding_datacube.coords[time_dim].values
        time_dim_dtype = embedding_datacube.coords[time_dim].dtype

        if not np.issubdtype(time_dim_dtype, np.datetime64):
            if np.issubdtype(time_dim_dtype, np.integer) and max(time_coords) < 3000:
                # if temp dim is type int and below 3000, we assume they are years
                new_time_coords = pd.to_datetime(time_coords, format="%Y")
            elif np.issubdtype(time_dim_dtype, np.integer) and max(time_coords) >= 3000:
                # if temp dim is type int and above 3000, we assume unix epochs
                # i.e. seconds since Jan 1, 1970
                new_time_coords = pd.to_datetime(time_coords, unit="s")
            else:
                raise NotImplementedError(
                    f"No time decodeing implemented for temporal dimension of"
                    f" dtype {time_dim_dtype}"
                )
            embedding_datacube = embedding_datacube.assign_coords(
                {time_dim: new_time_coords}
            )

    # todo: prevent loading huge zarr stores without bbox

    if bbox is not None:
        if crs is not None:
            crs_str = crs.to_string()
        else:
            crs_str = "EPSG:4326"

        bbox_reproj = _reproject_bbox(bbox, target_crs=crs_str)
        x_start, x_end = bbox_reproj.east, bbox_reproj.west
        y_start, y_end = bbox_reproj.south, bbox_reproj.north

        x_coord_name, y_coord_name = dim_utils.get_spatial_dim_names(embedding_datacube)

        x_coord_values = embedding_datacube.coords[x_coord_name].values
        y_coord_values = embedding_datacube.coords[y_coord_name].values

        x_slice = _monotonic_bounds_slice(x_coord_values, x_start, x_end)
        y_slice = _monotonic_bounds_slice(y_coord_values, y_start, y_end)

        if x_slice is None or y_slice is None:
            raise ValueError("Bounding box does not intersect the datacube")

        # --- Snap slices to chunk boundaries ---

        if preferred_chunks:
            x_dim = embedding_datacube[x_coord_name].dims[0]
            y_dim = embedding_datacube[y_coord_name].dims[0]

            x_slice = _snap_slice_to_chunks(
                x_slice,
                embedding_datacube.sizes[x_dim],
                preferred_chunks.get(x_dim),
            )
            y_slice = _snap_slice_to_chunks(
                y_slice,
                embedding_datacube.sizes[y_dim],
                preferred_chunks.get(y_dim),
            )
        # --------------------------------------------

        embedding_datacube = embedding_datacube.isel(
            {x_coord_name: x_slice, y_coord_name: y_slice}
        )

    if temporal_extent is not None and time_dim is not None:
        t_start = temporal_extent.start.to_numpy()
        t_end = temporal_extent.end.to_numpy()

        # 1-D time coord is tiny -> pull it into memory (this is cheap)
        time_vals = np.asarray(embedding_datacube.coords[time_dim].values)

        # boolean mask for the requested range, then integer positions
        mask = (time_vals >= t_start) & (time_vals <= t_end)
        idx = np.flatnonzero(mask)
        embedding_datacube = embedding_datacube.isel({time_dim: idx})

    if preferred_chunks:
        embedding_datacube = embedding_datacube.chunk(preferred_chunks)
        # drop shard encoding to not cause problems further down the line
        embedding_datacube.encoding.pop("shards", None)
    else:
        embedding_datacube = embedding_datacube.chunk()

    return embedding_datacube.chunk()


def _get_extracted_store(path: str, cache_dir: Path) -> Path:
    """
    Ensure the zip referenced by `path` is extracted in the cache dir and
    return the path to the extracted zarr store. Uses `path` as the ID so
    work is not repeated.
    """
    # Derive a stable, unique directory name from the full path (the "ID").
    path_id = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    zip_stem = Path(urlparse(path).path).stem  # nice human-readable suffix
    extract_dir = cache_dir / f"{zip_stem}_{path_id}"

    # If already extracted, reuse it.
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return _find_store(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    if _is_local_path(path):
        local_zip = Path(path)
        if not local_zip.is_file():
            raise FileNotFoundError(f"Local zip not found: {path}")
        zip_utils.extract_zip_archive(local_zip, extract_dir)
        # Original local file is left untouched.
    else:
        # Remote: download into cache, extract, then delete the downloaded zip.
        downloaded_zip = cache_dir / f"{path_id}.zip"
        try:
            # _download(path, downloaded_zip)
            # _extract_zip(downloaded_zip, extract_dir)
            download_utils.download(path, target_path=downloaded_zip)
            zip_utils.extract_zip_archive(downloaded_zip, extract_dir)
        finally:
            if downloaded_zip.exists():
                downloaded_zip.unlink()

    return _find_store(extract_dir)


def _is_local_path(path: str) -> bool:
    """Return True if `path` refers to a file on the local machine."""
    parsed = urlparse(path)
    # No scheme, or a file:// scheme, or a Windows drive letter -> local.
    if parsed.scheme in ("", "file"):
        return True
    if len(parsed.scheme) == 1:  # e.g. "C:\..." parsed as scheme "c"
        return True
    return False


def _find_store(extract_dir: Path) -> Path:
    """
    Locate the zarr store inside `extract_dir`. Handles the common case
    where a zip contains a single top-level folder (the store itself).
    """
    # A .zarr directory anywhere near the top is the store.
    zarr_dirs = list(extract_dir.glob("*.zarr")) + list(extract_dir.glob("*/*.zarr"))
    if zarr_dirs:
        return zarr_dirs[0]

    # Otherwise, if there's a single top-level entry, assume it's the store.
    entries = [p for p in extract_dir.iterdir() if not p.name.startswith("__")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]

    # Fall back to the extraction dir itself.
    return extract_dir


def _load_tiff(path: str) -> xr.DataArray:
    dc = rioxarray.open_rasterio(path, chunks=True)
    dc = dc.rename({"band": "embedding"})  # rename bands dim to embedding

    # squeeze x and y if its only 1 (i.e. tiff only has one pixel
    if len(dc.coords["x"]) == 1 and len(dc.coords["y"]) == 1:
        dc = dc.squeeze(dim=["x", "y"], drop=True)
    else:
        # we assume that we have (re-alligned) ViT patch embeddings
        # as vit patch embeddings do not contain content about their (spatial) location
        # we drop the x and y coords
        dc = dc.drop_vars(["x", "y"])
        dc = dc.rename({"x": "patch_x", "y": "patch_y"})
    # todo: what if tiff contains embeddins that are NOT ViT patch embeddings?

    dc.attrs.clear()  # remove rioxarray stats
    return dc


def _prepare_geoparquet(
    path: str,
    bbox: BoundingBox | None,
    geom_column_name: str,
    emb_column_name: str,
    emb_size: int,
    emb_dtype: np.dtype,
    to_epsg_4326: bool = False,
) -> tuple[NDArray[shapely.Geometry], da.Array]:
    @delayed
    def _stack_partition(series):
        # series is a pandas Series of numpy arrays -> 2D array
        return np.stack(series.to_numpy())

    gdf: dask_geopandas.GeoDataFrame = dask_geopandas.read_parquet(
        path, columns=[geom_column_name, emb_column_name]
    )

    if bbox is not None:
        if bbox.crs != "EPSG:4326":
            bbox = _reproject_bbox(bbox, "EPSG:4326")

        xmin, ymin, xmax, ymax = (
            bbox.west,
            bbox.south,
            bbox.east,
            bbox.north,
        )
        bbox_geom = shapely.box(xmin, ymin, xmax, ymax)
        gdf_bbox = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
        gdf_bbox = gdf_bbox.to_crs(gdf.crs)
        gdf = dask_geopandas.sjoin(gdf, gdf_bbox, how="inner", predicate="intersects")

    if to_epsg_4326:
        gdf = gdf.to_crs(epsg=4326)

    # this does the trick of stacking the arrays, but is VERY slow
    # dask checks if the arrays are of same length and can be stacked -> SLOW
    # embedding_array = da.stack(gdf_embeddings[emb_column_name])

    # same effect but more efficient: we already KNOW that each array is same length
    # 1) make one delayed dask object per partition
    # 2) for each delayed dask object make a delayed (lazy) dask array
    # 3) concat all lazy dask arrays to one big lazy dask array
    geom_parts = gdf[
        geom_column_name
    ].to_delayed()  # delayed objects, one per partition, for both columns
    emb_parts = gdf[emb_column_name].to_delayed()

    # compute geometries per partition (you need them as numpy coords anyway).
    # This ALSO gives us the exact row count of every partition.
    geoms_by_part = dask.compute(*geom_parts)  # list of pandas Series
    lengths = [len(g) for g in geoms_by_part]  # known rows per partition
    geoms_array = np.concatenate([g.normalize().to_numpy() for g in geoms_by_part])

    arrays = [
        da.from_delayed(
            _stack_partition(part),
            shape=(length, emb_size),  # rows per partition unknown -> nan
            dtype=emb_dtype,
        )
        for part, length in zip(emb_parts, lengths)
    ]
    embedding_array = da.concatenate(arrays, axis=0)
    return geoms_array, embedding_array


def _load_parquet_item(
    path: str, bbox: BoundingBox | None, to_epsg_4326: bool = False
) -> xr.DataArray:
    # check geom column
    # check embedding column (embedding or embeddings?)
    # check if col is correct dtype (list? what float?)
    with fsspec.open(path, "rb") as file:
        parquet_schema = pq.read_schema(file)

    # search for embedding column
    col_names = [n.name for n in parquet_schema]
    possible_embedding_col_names = ["embedding", "embeddings", "emb", "embs"]
    for pos_emb_col_name in possible_embedding_col_names:
        if pos_emb_col_name in col_names:
            emb_column_name = pos_emb_col_name
            break
    else:
        raise Exception(
            f"Could not identify embedding column. Must be named one of "
            f"{','.join(possible_embedding_col_names)}"
        )

    # get embedding column info: embedding length and datatype
    emb_col_dtype = parquet_schema.field(emb_column_name).type
    if isinstance(emb_col_dtype, pyarrow.FixedSizeListType):
        emb_size = emb_col_dtype.list_size
        emb_dtype = emb_col_dtype.value_type.to_pandas_dtype()
    else:
        raise NotImplementedError(
            f"Embedding column data type is {str(emb_col_dtype)} which is unsupported."
            f"Must be FixedSizeList"
        )

    # get geometry column name
    geo_metadata_bytes = parquet_schema.metadata.get(b"geo")
    if geo_metadata_bytes:
        # Parse JSON bytes
        geo_metadata = json.loads(geo_metadata_bytes.decode("utf-8"))

        # Get the primary geometry column name
        geom_column_name = geo_metadata.get("primary_column")
    else:
        raise Exception(
            "The provided parquet parquet file is not a valid GeoParquet, as no 'geo' "
            "metadata could be found."
        )

    if to_epsg_4326:
        crs = "EPSG:4326"
    else:
        crs = pyproj.CRS.from_json_dict(geo_metadata["columns"]["geometry"]["crs"])

    geom_coords, emb_values = _prepare_geoparquet(
        path, bbox, geom_column_name, emb_column_name, emb_size, emb_dtype, to_epsg_4326
    )
    emb_cube = xr.DataArray(
        emb_values, dims=["geometry", "embedding"], coords={"geometry": geom_coords}
    ).xvec.set_geom_indexes("geometry", crs=crs)
    return emb_cube


def _load_embedding_item(
    stac_item: pystac.Item,
    asset_name: str,
    bbox: BoundingBox | None,
    temporal_extent: TemporalInterval | None,
    to_epsg_4326: bool = False,
) -> xr.DataArray:
    embedding_asset = stac_item.assets[asset_name]
    media_type = embedding_asset.media_type
    path = embedding_asset.href
    time = _get_item_time(stac_item)

    # we assume that embeddings as tif or parquet are purely spatial
    # if its it zarr, it can be spatial or spatio-temporal

    # (geo)tif: does not hold embs of multiple timesteps: temp_extent is ignored
    if media_type.startswith("image/tif"):  # todo: to_epsg_4326, bbox
        footprint = shapely.from_geojson(json.dumps(stac_item.geometry))
        emb_cube = _load_tiff(path)
        emb_cube = emb_cube.expand_dims({"geometry": [footprint], "time": [time]})

        return emb_cube

    # parquet file:
    # todo: handle temporal parquet
    if media_type.startswith("application/x-parquet") or media_type.startswith(
        "application/vnd.apache.parquet"
    ):
        emb_cube = _load_parquet_item(path, bbox, to_epsg_4326)
        emb_cube = emb_cube.expand_dims({"time": [time]})
        return emb_cube

    # zarr store
    if media_type.startswith("application/vnd.zarr"):
        emb_cube = _load_zarr(path, bbox, temporal_extent)
        return emb_cube

    # if parquet: load_parquet
    # if zarr: load_zarr
    raise TypeError(f"Loading embeddings of media-type {media_type} unsupported")


def _construct_embedding_vector_cube(
    item_arrays: list[list[xr.DataArray]],
    geom_coords: list[shapely.Geometry],
    time_coords: list[Datetime],
) -> xr.DataArray:
    single_temp_cubes = []
    for single_timestep_arrays in item_arrays:
        x = xr.concat(single_timestep_arrays, dim="geometry")
        single_temp_cubes.append(x)

    embedding_cube = xr.concat(single_temp_cubes, dim="time")
    embedding_cube = embedding_cube.assign_coords(
        {"geometry": geom_coords, "time": time_coords}
    )
    embedding_cube = embedding_cube.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
    return embedding_cube


def _load_embedding_collection_tif(
    items: pystac.ItemCollection,
    asset_name: str,
    max_workers: int = 16,
):
    # 1. Extract metadata + paths first (cheap, keeps ordering)
    loaded = []
    print("load items")
    for stac_item in items:
        path = stac_item.assets[asset_name].href
        footprint = shapely.from_geojson(json.dumps(stac_item.geometry)).normalize()
        time = _get_item_time(stac_item)
        loaded.append({"path": path, "footprint": footprint, "time": time})
    print("done loading items")

    # 2. Load all tiffs concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        emb_cubes = list(executor.map(lambda e: _load_tiff(e["path"]), loaded))

    for entry, cube in zip(loaded, emb_cubes):
        entry["cube"] = cube

    # 3. Assemble the cube (serial, fast, preserves original order)
    item_arrays: list[list[xr.DataArray]] = []
    time_coords = []
    geom_coords = []

    for entry in loaded:
        time = entry["time"]
        footprint = entry["footprint"]
        emb_cube = entry["cube"]

        if time not in time_coords:
            time_coords.append(time)
            time_index = len(time_coords) - 1
            item_arrays.append(len(geom_coords) * [xr.DataArray()])
        else:
            time_index = time_coords.index(time)

        footprint_index = _match_geom_in_list(geom_coords, footprint, tolerance=0.00001)
        if footprint_index is None:
            geom_coords.append(footprint)
            footprint_index = len(geom_coords) - 1
            for time_step in item_arrays:
                time_step.append(xr.DataArray())

        item_arrays[time_index][footprint_index] = emb_cube

    return _construct_embedding_vector_cube(item_arrays, geom_coords, time_coords)


def _crs_of(href: str) -> str:
    """Extract the CRS (as a stable string) from a geoparquet file's 'geo' metadata."""
    schema = pq.read_schema(href)
    meta = schema.metadata or {}
    if b"geo" not in meta:
        raise Exception("Non-Geoparquet files were encountered.")
    geo = json.loads(meta[b"geo"])
    col = geo["columns"][geo.get("primary_column", "geometry")]
    # crs may be absent (defaults to OGC:CRS84), a dict (PROJJSON) or a string
    return json.dumps(col.get("crs"), sort_keys=True)


def _proj_necessary(
    collection: pystac.Collection, items: pystac.ItemCollection, asset_name
) -> bool:
    PROJ_CODE = "proj:code"
    # --- Step 1: collection-level proj:code ---
    if PROJ_CODE in (collection.extra_fields or {}):
        return False
    summaries = collection.summaries.to_dict() if collection.summaries else {}
    if PROJ_CODE in summaries:
        return False

    # --- Step 2: item-level proj:code ---
    codes = []
    for item in items:
        if PROJ_CODE not in item.properties:
            codes = None  # not present on all items -> fall through to step 3
            break
        codes.append(item.properties[PROJ_CODE])

    if codes:  # present on every item (and at least one item)
        return len(set(codes)) != 1

    # --- Step 3: inspect geoparquet CRS ---
    crs_values = set()
    for item in items:
        asset = item.assets.get(asset_name)
        if asset is None:
            raise Exception("Non-Geoparquet files were encountered.")
        crs_values.add(_crs_of(asset.href))

    return len(crs_values) != 1


def _load_embedding_collection_parquet(
    collection: pystac.Collection,
    items: pystac.ItemCollection,
    asset_name: str,
    bbox: BoundingBox | None,
) -> xr.DataArray:
    def _prep_and_load_item(
        item: pystac.Item,
        asset_name: str,
        bbox: BoundingBox,
        transform_to_epsg_4326: bool,
    ) -> xr.DataArray:
        emb_dc = _load_embedding_item(item, asset_name, bbox, transform_to_epsg_4326)
        emb_dc = emb_dc.squeeze("time")
        return emb_dc

    # at this point we assume that one item only contains data from one timestep
    item_datetimes = [_get_item_time(i) for i in items]

    to_epsg_4326 = _proj_necessary(collection, items, asset_name)

    # Parallelized execution using a ThreadPoolExecutor
    MAX_THREADS = 8  # Change this to your desired number of threads
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        per_item_cubes = list(
            executor.map(
                lambda i: _prep_and_load_item(i, asset_name, bbox, to_epsg_4326), items
            )
        )

    # match and order embeddings by space and time
    item_arrays: list[list[xr.DataArray]] = []
    time_coords: list[Datetime] = []
    geom_coords: list[shapely.geometry.base.BaseGeometry] = []
    for item_time, item_cube in zip(item_datetimes, per_item_cubes):
        if item_time not in time_coords:
            time_coords.append(item_time)
            time_index = len(time_coords) - 1
            item_arrays.append(len(geom_coords) * [xr.DataArray()])
        else:
            time_index = time_coords.index(item_time)

        # dc_data = item_cube.data
        for dc_geom_coord_idx, dc_geom_coord in enumerate(
            item_cube.coords["geometry"].values
        ):
            geom_idx = _match_geom_in_list(geom_coords, dc_geom_coord, 0.00001)
            if geom_idx is None:
                geom_coords.append(dc_geom_coord)
                geom_idx = len(geom_coords) - 1
                for time_step in item_arrays:
                    time_step.append(xr.DataArray())

            emb = item_cube.isel(geometry=dc_geom_coord_idx, drop=True)
            item_arrays[time_index][geom_idx] = emb

    # combine the small individual embedding cubes to
    embedding_cube = _construct_embedding_vector_cube(
        item_arrays, geom_coords, time_coords
    )
    return embedding_cube


def _parse_temporal_extent(
    temporal_extent: TemporalInterval | None, query_params: dict
) -> None:
    if temporal_extent is None:
        return

    s = str(temporal_extent[0].to_numpy()) if temporal_extent[0] is not None else None
    e = str(temporal_extent[1].to_numpy()) if temporal_extent[1] is not None else None
    query_params["datetime"] = [s, e]


def _parse_spatial_extent(
    spatial_extent: BoundingBox | None, query_params: dict
) -> None:
    if spatial_extent is None:
        raise NotImplementedError("Spatial extent is needed")

    try:
        bbox = [
            spatial_extent.west,
            spatial_extent.south,
            spatial_extent.east,
            spatial_extent.north,
        ]
        query_params["bbox"] = bbox
    except Exception as e:
        raise Exception(f"Unable to parse the provided spatial extent: {e}")


def _get_collection_asset_media_type(
    collection: pystac.Collection, items: pystac.ItemCollection, asset_name: str
) -> str:
    if collection.item_assets:
        if asset_name not in collection.item_assets:
            raise Exception
        emb_data_format = collection.item_assets[asset_name].media_type
    else:
        # parse from all items
        # i = items[0].assets["a"].media_type
        emb_data_formats = [
            i.assets[asset_name].media_type
            for i in items
            if i.assets.get(asset_name) is not None
        ]
        all_same_media_type = all(emb_data_formats[0] == e for e in emb_data_formats)

        if not all_same_media_type:
            raise Exception(
                "The STAC embedding assets are not all of the same data format"
            )

        emb_data_format = emb_data_formats[0]

    return emb_data_format


# parts of this function have been taken from this script
# https://github.com/Open-EO/openeo-processes-dask/blob/main/openeo_processes_dask/process_implementations/cubes/load.py
def _load_embedding_collection(
    url: str,
    collection: pystac.Collection,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    asset_name: str = "embeddings",
) -> xr.DataArray:
    catalog_url, collection_id = stac_utils.search_for_parent_catalog(url)
    query_params = {"collections": [collection_id]}
    stac_client = pystac_client.Client.open(catalog_url)

    _parse_temporal_extent(temporal_extent, query_params)
    _parse_spatial_extent(spatial_extent, query_params)

    items = stac_client.search(**query_params).item_collection()

    if len(items) == 0:
        raise Exception(
            "Could not find any embeddings in the STAC collection for the provided "
            "spatial bounding box and/or timespan."
        )

    # figure out the data format that all assets have in common
    emb_data_format = _get_collection_asset_media_type(collection, items, asset_name)

    # use media-type specific loader
    if emb_data_format.startswith("image/tif"):
        embedding_cube = _load_embedding_collection_tif(items, asset_name)
        return embedding_cube

    if emb_data_format.startswith(
        "application/x-parquet"
    ) or emb_data_format.startswith("application/vnd.apache.parquet"):
        embedding_cube = _load_embedding_collection_parquet(
            collection, items, asset_name, spatial_extent
        )
        return embedding_cube

    # how to deal with collection of zarr-items?
    # how to deal with collectino with collection-asset?

    raise NotImplementedError(f"Cannot read embeddings of type {emb_data_format}")


def load_embeddings(
    url: str,
    spatial_extent: BoundingBox | None = None,
    temporal_extent: TemporalInterval | None = None,
    asset_name: str = "embeddings",
) -> xr.DataArray:
    stac_obj_dict = stac_utils.load_stac_json(url)

    # todo: validate STAC and embedding extension

    try:
        stac_obj = pystac.read_dict(stac_obj_dict)
    except pystac.STACTypeError as e:
        raise OpenEOException("Provided URL does not point to a valid STAC object")

    if isinstance(stac_obj, pystac.Item):
        return _load_embedding_item(
            stac_obj, asset_name, spatial_extent, temporal_extent, False
        )
    elif isinstance(stac_obj, pystac.Collection):
        return _load_embedding_collection(
            url, stac_obj, spatial_extent, temporal_extent, asset_name
        )
    raise NotImplementedError(
        f"Loading of a STAC object of type {stac_obj.STAC_OBJECT_TYPE} is not supported"
    )
