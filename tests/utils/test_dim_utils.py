import dask.array as da
import pytest
import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.exceptions import (
    BandNotFoundException,
)
from openeo_processes_dask_ml.process_implementations.utils import dim_utils


@pytest.mark.parametrize(
    "dim_name", ["band", "bands", "b", "Band", "B", "channel", "Channel", "foobar"]
)
def test_get_band_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random((2, 2, 2, 2)), dims=["time", dim_name, "y", "x"])
    if dim_name != "foobar":
        assert dim_utils.get_band_dim_name(dc) == dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_band_dim_name(dc)


def test_spatial_dim_options():
    d = dim_utils.spatial_dim_options
    assert "x" in d
    assert "y" in d
    assert "lat" in d
    assert "lon" in d


@pytest.mark.parametrize("dim_name", ["time", "times", "Time", "foobar"])
def test_get_time_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random((2, 2, 2, 2)), dims=[dim_name, "band", "y", "x"])
    if dim_name != "foobar":
        assert dim_utils.get_time_dim_name(dc) == dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_time_dim_name(dc)


@pytest.mark.parametrize(
    "dim_name", ["x", "lng", "X", "longitude", "Longitude", "foobar"]
)
def test_get_x_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random((2, 2)), dims=["y", dim_name])
    if dim_name != "foobar":
        assert dim_utils.get_x_dim_name(dc) == dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_x_dim_name(dc)


@pytest.mark.parametrize(
    "dim_name", ["y", "lat", "Y", "Latitude", "latitude", "foobar"]
)
def test_get_y_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random((2, 2)), dims=[dim_name, "x"])
    if dim_name != "foobar":
        assert dim_utils.get_y_dim_name(dc) == dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_y_dim_name(dc)


@pytest.mark.parametrize("dim_names", [["x", "y"], ["lon", "lat"], ["foo", "bar"]])
def test_get_spatial_dim_names(dim_names: list[str]):
    dc = xr.DataArray(da.random.random((2, 2)), dims=dim_names)
    if dim_names[0] != "foo":
        assert dim_utils.get_spatial_dim_names(dc) == tuple(dim_names)
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_spatial_dim_names(dc)


@pytest.mark.parametrize("dim_name", ["geom", "geometry", "GEOMETRY", "geom", "foo"])
def test_get_geometry_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random((2, 2)), dims=[dim_name, "bar"])
    if dim_name != "foo":
        assert dim_utils.get_geometry_dim_name(dc) == dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_y_dim_name(dc)


@pytest.mark.parametrize("dim_name", ["x", "times", "y", "lat", "bands"])
def test_get_alternative_datacube_dim_name(dim_name: str):
    dc = xr.DataArray(da.random.random(2), dims=[dim_name])
    d = dim_utils.get_alternative_datacube_dim_name(dc, dim_name)
    assert dim_name == d

    d = dim_utils.get_alternative_datacube_dim_name(dc, "batch")
    assert d is None


@pytest.mark.parametrize("band_name", ("b04", "B04", "foo"))
def test_get_band_alternative_names(band_name: str):
    if band_name != "foo":
        alt_names = dim_utils.get_band_alternative_names(band_name)
        assert band_name.lower() in alt_names
        assert "red" in alt_names
    else:
        with pytest.raises(BandNotFoundException):
            dim_utils.get_band_alternative_names(band_name)


@pytest.mark.parametrize(
    "dc_bands, mlm_bands, dc_bands_selected",
    (
        (["red", "nir", "green"], ["red", "nir"], ["red", "nir"]),
        (["b04", "b08", "b09"], ["B04", "B08"], ["b04", "b08"]),
        (["B04", "B08", "foo"], ["red", "nir"], ["B04", "B08"]),
    ),
)
def test_get_dc_band_names(
    dc_bands: list[str], mlm_bands: list[str], dc_bands_selected: list[str]
):
    selected_bands = dim_utils.get_dc_band_names(dc_bands, mlm_bands)
    assert selected_bands == dc_bands_selected


@pytest.mark.parametrize(
    "in_dims, out_dims, truth_removed, truth_added",
    (
        (
            ["batch", "bands", "x", "y", "time"],
            ["batch", "embedding", "x", "y", "time"],
            [1],
            [1],
        ),
        (["y", "x"], ["y", "x"], [], []),
        (["x", "y"], ["x", "y", "time"], [], [2]),
        (["x", "y", "time"], ["x", "time"], [1], []),
        (["x", "y"], ["time", "x", "y"], [], [0]),
    ),
)
def test_compare_input_and_output_datacube_dims(
    in_dims: list[str],
    out_dims: list[str],
    truth_removed: list[int],
    truth_added: list[int],
):
    removed, added = dim_utils.compare_input_and_output_datacube_dims(in_dims, out_dims)
    assert truth_removed == removed
    assert truth_added == added
