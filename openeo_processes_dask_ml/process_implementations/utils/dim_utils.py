import difflib
from typing import Union

import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.exceptions import (
    BandNotFoundException,
)

# only add lower-case dimensions. use .lower() when comparing dimension names
band_dim_options = ["band", "bands", "b", "channel", "channels"]
time_dim_options = ["time", "times", "t", "date", "dates"]
x_dim_options = ["x", "lon", "lng", "longitude"]
y_dim_options = ["y", "lat", "latitude"]
spatial_dim_options = [*x_dim_options, *y_dim_options]
batch_dim_options = ["batch", "batches"]
embedding_dim_options = ["embedding", "embeddings"]
geometry_dim_options = ["geom", "geometry"]


def _find_alternative_dim_name_in_datacube(
    dc: xr.DataArray | xr.Dataset, dim_name_options: list[str]
) -> str:
    """
    Identify a dimension in a datacube based on a list of possible dimension names
    :param dc: The datacube
    :param dim_name_options: Dimension names to be searched for in the datacube
    :return: The found dimension name
    """
    for dim_name in dc.dims:
        if isinstance(dim_name, str) and dim_name.lower() in dim_name_options:
            return dim_name
    raise ValueError(
        f"The datacube does not contain one of the following dimensions: "
        f"{', '.join(dim_name_options)}"
    )


def get_band_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    """
    Identifies the bands dimension in a data cube.
    :param dc: The datacube
    :return: Name of the band dimension
    :raise DimensionMissing: When no bands dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, band_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a band dimension in the datacube. "
            f"If this is a mistake, rename the band dimension to one of the following: "
            f"{', '.join(band_dim_options)}"
        )


def get_time_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    """
    Identifies the time dimension in a data cube.
    :param dc: The datacube
    :return: Name of the time dimension
    :raise DimensionMissing: When no time dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, time_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a time dimension in the datacube. "
            f"If this is a mistake, rename the time dimension to one of the following: "
            f"{', '.join(time_dim_options)}"
        )


def get_x_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    """
    Identifies the X dimension in a data cube.
    :param dc: The datacube
    :return: Name of the X dimension
    :raise DimensionMissing: When no X dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, x_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find an X dimension in the datacube. "
            f"If this is a mistake, rename the X dimension to one of the following: "
            f"{', '.join(x_dim_options)}"
        )


def get_y_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    """
    Identifies the Y dimension in a data cube.
    :param dc: The datacube
    :return: Name of the Y dimension
    :raise DimensionMissing: When no Y dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, y_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a Y dimension in the datacube. "
            f"If this is a mistake, rename the Y dimension to one of the following: "
            f"{', '.join(y_dim_options)}"
        )


def get_spatial_dim_names(dc: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    """
    Identifies the spatial dimensions in a data cube.
    :param dc: The datacube
    :return: Names of the spatial dimensions
    :raise DimensionMissing: When a spatial dimension could not be identified.
    """
    x_dim_name = get_x_dim_name(dc)
    y_dim_name = get_y_dim_name(dc)
    return x_dim_name, y_dim_name


def get_embedding_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    try:
        return _find_alternative_dim_name_in_datacube(dc, embedding_dim_options)
    except ValueError:
        raise DimensionMissing("Could not find an embedding dimension in the datacube")


def get_geometry_dim_name(dc: xr.DataArray | xr.Dataset) -> str:
    try:
        return _find_alternative_dim_name_in_datacube(dc, geometry_dim_options)
    except ValueError:
        raise DimensionMissing("Could not find a geometry dimension in the datacube")


def is_raster_datacube(dc: xr.DataArray | xr.Dataset) -> bool:
    try:
        _ = get_spatial_dim_names(dc)
        return True
    except DimensionMissing:
        return False


def is_vector_datacube(dc: xr.DataArray | xr.Dataset) -> bool:
    try:
        _ = get_geometry_dim_name(dc)
        return True
    except DimensionMissing:
        return False


def get_alternative_datacube_dim_name(dc: xr.DataArray, dim_name: str) -> str | None:
    """
    Search for an alternative dimension name in the datacube.
    E.g. searching for "bands" dimension name will return "band" dimension if in DC.
    :param dc: The datacube to be searched for dimensions
    :param dim_name: The dimension name to search for
    :return: The name of the dimension in the datacube, or None if no match was found
    """

    dc_dims = dc.dims

    if dim_name in dc_dims:
        return dim_name

    t_dim_names = time_dim_options
    if dim_name in t_dim_names:
        return next((t_dim for t_dim in t_dim_names if t_dim in dc_dims), None)

    b_dim_names = band_dim_options
    if dim_name in b_dim_names:
        return next((b_dim for b_dim in b_dim_names if b_dim in dc_dims), None)

    x_dim_names = x_dim_options
    if dim_name in x_dim_names:
        return next((x_dim for x_dim in x_dim_names if x_dim in dc_dims), None)

    y_dim_names = y_dim_options
    if dim_name in y_dim_names:
        return next((y_dim for y_dim in y_dim_names if y_dim in dc_dims), None)

    batch_dim_names = band_dim_options
    if dim_name in batch_dim_names:
        return next((b_dim for b_dim in batch_dim_names if b_dim in dc_dims), None)

    return None


def get_band_alternative_names(band_name: str) -> list[str]:
    """
    Get alternative names for a band name, e.g. "b04" -> "red"
    :param band_name: The band to find altenative names
    :return: list of possible band name alternatives
    """
    band_name_lower = band_name.lower()
    band_name_groups = [
        # sentienl 2 bands
        ["b01", "coastal", "b01_20m", "b01_60m"],
        ["b02", "b2", "blue", "b02_10m", "b02_20m", "b02_60m"],
        ["b03", "b3", "green", "b03_10m", "b03_20m", "b03_60m"],
        ["b04", "b4", "red", "b04_10m", "b04_20m", "b04_60m"],
        ["b05", "b5", "rededge1", "b05_20m", "b05_60m"],
        ["b06", "b6", "rededge2", "b06_20m", "b06_60m"],
        ["b07", "b7", "rededge3", "b07_20m", "b07_60m"],
        ["b08", "b8", "nir", "b08_10m", "b08_20m", "b08_60m"],
        ["b8a", "b08a", "nir08", "nir08", "nir08a", "nir_narrow", "b8a_20m", "b8a_60m"],
        ["b09", "b9", "nir09", "b09_60m"],
        ["b10", "cirrus"],
        ["b11", "swir16", "swir1", "swir_1", "swir-1", "b11_20m", "b11_60m"],
        ["b12", "swir22", "swir2", "swir_2", "swir-2", "b12_20m", "b12_60m"],
        ["aot"],
        ["scl"],
        ["snw"],
        # sentinel 1 bands
        ["hh"],
        ["hv"],
        ["vh"],
        ["vv"],
        # other
        ["ndvi"],
    ]

    for band_name_group in band_name_groups:
        if band_name_lower in band_name_group:
            return band_name_group

    raise BandNotFoundException(
        f"Could not find band name alternatives for band {band_name}"
    )


def get_dc_band_names(dc_band_names: list[str], band_names: list[str]) -> list[str]:
    """
    Get a list of band coordinates from the datacube that correspond to the list of band
    names submitted to the function, also considering alternative band names
    :param dc_band_names: band coordinates from the data cube
    :param band_names: the list of band names
    :return: list of band names in data cube
    """

    bands_in_dc = []

    for b_name in band_names:
        if b_name in dc_band_names:
            bands_in_dc.append(b_name)
            continue

        alt_names = get_band_alternative_names(b_name)
        for alt_name in alt_names:
            lower_dc_names = [n.lower() for n in dc_band_names]
            if alt_name in lower_dc_names:
                match = dc_band_names[lower_dc_names.index(alt_name)]
                bands_in_dc.append(match)
                break

    return bands_in_dc


def compare_input_and_output_datacube_dims(
    input_dims: list[str], output_dims: list[str]
) -> tuple[list[int], list[int]]:
    """
    Compares two lists and finds the indices of added and removed elements.

    Args:
        input_dims: List of dimensions in input datacube
        output_dims: list of dimensions in output datacube

    Returns:
        A tuple containing two lists:
        - A list of indices of elements removed from list1.
        - A list of indices of elements added to list2.
    """
    matcher = difflib.SequenceMatcher(None, input_dims, output_dims)

    removed_indices = []
    added_indices = []

    # get_opcodes() returns a list of tuples describing the differences.
    # Each tuple is (tag, i1, i2, j1, j2)
    # tag can be 'equal', 'delete', 'insert', or 'replace'.
    # i1:i2 is the slice from list1.
    # j1:j2 is the slice from list2.
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            # This block was in list1 but not in list2.
            # Collect all indices from this block.
            removed_indices.extend(range(i1, i2))

        elif tag == "insert":
            # This block is in list2 but was not in list1.
            added_indices.extend(range(j1, j2))

        elif tag == "replace":
            # A 'replace' is essentially a 'delete' followed by an 'insert'.
            removed_indices.extend(range(i1, i2))
            added_indices.extend(range(j1, j2))

    return removed_indices, added_indices
