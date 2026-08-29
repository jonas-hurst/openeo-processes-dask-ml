from datetime import datetime

import numpy as np
import pyproj
import pystac
import pytest
import shapely
import xarray as xr
from dask import array as da
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval

from openeo_processes_dask_ml.process_implementations.load_embeddings import (
    _construct_embedding_vector_cube,
    _crs_of,
    _get_item_time,
    _load_embedding_collection,
    _load_embedding_collection_parquet,
    _load_embedding_collection_tif,
    _load_embedding_item,
    _load_parquet_item,
    _load_tiff,
    _match_geom_in_list,
    _prepare_geoparquet,
    load_embeddings,
)


def _make_item(
    item_id: str, date: datetime, polygon_coords, asset_path: str, mediatype: str
) -> pystac.Item:
    """Build a minimal pystac.Item with the given {asset_name: media_type} assets."""
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": polygon_coords,
        },
        bbox=[0, 0, 1, 1],
        datetime=date,
        properties={},
    )
    item.add_asset(
        "embedding",
        pystac.Asset(href=asset_path, media_type=mediatype),
    )
    return item


def _make_item_collection(items: list[pystac.Item]) -> pystac.ItemCollection:
    return pystac.ItemCollection(items)


def test_get_item_time():
    dt = datetime.fromisoformat("2026-07-15T02:54:14-12:00")
    start_dt = datetime.fromisoformat("2015-07-15T02:54:14-12:00")
    end_dt = datetime.fromisoformat("2015-07-15T02:54:14-12:00")

    i = pystac.Item("asdf", None, None, dt, {})
    t = _get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == dt.isoformat()

    i = pystac.Item(
        "asdf", None, None, dt, {}, start_datetime=start_dt, end_datetime=end_dt
    )
    t = _get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == dt.isoformat()

    i = pystac.Item(
        "asdf", None, None, None, {}, start_datetime=start_dt, end_datetime=end_dt
    )
    t = _get_item_time(i)
    assert isinstance(t, datetime)
    assert t.isoformat() == start_dt.isoformat()


def test_match_geom_in_list_point():
    pnt1 = shapely.Point([5.555, 1.111])
    pnt2 = shapely.Point([5.555, 1.111])
    pnt3 = shapely.Point([10.3, 3.3])
    pnt4 = shapely.Point([10.3, 3.300001])

    geom_list = []

    i = _match_geom_in_list(geom_list, pnt1, tolerance=0.00001)
    assert i is None
    geom_list.append(pnt1)

    i = _match_geom_in_list(geom_list, pnt2, tolerance=0.00001)
    assert i == 0

    i = _match_geom_in_list(geom_list, pnt3, tolerance=0.00001)
    assert i is None
    geom_list.append(pnt3)

    i = _match_geom_in_list(geom_list, pnt4, tolerance=0.00001)
    assert i == 1


def test_match_geom_in_list_polygon():
    # Helper to create a small square polygon offset by a specific coordinate
    def make_square(x, y):
        # Creates a 1x1 square with the bottom-left corner at (x, y)
        return shapely.Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])

    # Recreate the exact same spatial layout/relationships using Polygons
    poly1 = make_square(5.555, 1.111)
    poly2 = make_square(5.555, 1.111)
    poly3 = make_square(10.3, 3.3)
    poly4 = make_square(10.3, 3.300001)

    geom_list = []

    # 1. First polygon shouldn't match anything in an empty list
    i = _match_geom_in_list(geom_list, poly1, tolerance=0.00001)
    assert i is None
    geom_list.append(poly1)

    # 2. Second polygon is identical to the first, so it should match index 0
    i = _match_geom_in_list(geom_list, poly2, tolerance=0.00001)
    assert i == 0

    # 3. Third polygon is in a new location, shouldn't match index 0
    i = _match_geom_in_list(geom_list, poly3, tolerance=0.00001)
    assert i is None
    geom_list.append(poly3)

    # 4. Fourth polygon is offset within the 0.00001 tolerance, so it should match index 1
    i = _match_geom_in_list(geom_list, poly4, tolerance=0.00001)
    assert i == 1


def test_load_tiff_1x1():
    path = "tests/data/embedding_1x1.tif"
    e_dc = _load_tiff(path)

    assert "band" not in e_dc.dims
    assert "embedding" in e_dc.dims
    assert "x" not in e_dc.dims
    assert "y" not in e_dc.dims
    assert isinstance(e_dc.data, da.Array)

    e_dc.compute()


def test_prepare_geoparquet_without_bbox_without_transform():
    path = "tests/data/embeddings.parquet"
    geom_array, da_array = _prepare_geoparquet(
        path, None, "geometry", "embedding", 4, np.float32, False
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 4
    assert da_array.shape == (4, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = poly_utm.contains(geom_array)
    outside = ~poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_without_bbox_with_transform():
    path = "tests/data/embeddings.parquet"
    geom_array, da_array = _prepare_geoparquet(
        path, None, "geometry", "embedding", 4, np.float32, True
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 4
    assert da_array.shape == (4, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = ~poly_utm.contains(geom_array)
    outside = poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_with_bbox_without_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    geom_array, da_array = _prepare_geoparquet(
        path, bbox, "geometry", "embedding", 4, np.float32, False
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 2
    assert da_array.shape == (2, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = poly_utm.contains(geom_array)
    outside = ~poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_prepare_geoparquet_with_bbox_with_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    geom_array, da_array = _prepare_geoparquet(
        path, bbox, "geometry", "embedding", 4, np.float32, True
    )
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    assert len(geom_array) == 2
    assert da_array.shape == (2, 4)
    assert isinstance(da_array, da.Array)
    assert np.all([isinstance(x, shapely.Point) for x in geom_array.ravel()])
    assert da_array.dtype == np.float32

    inside = ~poly_utm.contains(geom_array)
    outside = poly_wgs84.contains(geom_array)
    assert inside.all()
    assert outside.all()


def test_load_parquet_item_without_bbox_without_transform():
    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = _load_parquet_item(path, None, False)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (4, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    inside = poly_utm.contains(geom_coords)
    outside = ~poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))


def test_load_parquet_item_without_bbox_with_transform():
    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = _load_parquet_item(path, None, True)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (4, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    outside = ~poly_utm.contains(geom_coords)
    inside = poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))


def test_load_parquet_item_with_bbox_without_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = _load_parquet_item(path, bbox, False)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (2, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    inside = poly_utm.contains(geom_coords)
    outside = ~poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))


def test_load_parquet_item_with_bbox_with_transform():
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    path = "tests/data/embeddings.parquet"
    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    e_dc = _load_parquet_item(path, bbox, True)

    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert e_dc.shape == (2, 4)
    assert e_dc.dtype == np.float32
    assert isinstance(e_dc.data, da.Array)
    geom_coords = e_dc.coords["geometry"].values
    assert np.all([isinstance(x, shapely.Point) for x in geom_coords])

    outside = ~poly_utm.contains(geom_coords)
    inside = poly_wgs84.contains(geom_coords)
    assert inside.all()
    assert outside.all()
    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))


def test_load_embedding_item_tiff():
    item = pystac.Item.from_file("tests/data/item_tif.json")
    asset_name = "embeddings"

    e_dc = _load_embedding_item(item, asset_name, None, None)
    assert e_dc.shape == (1, 1, 768)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)


@pytest.mark.parametrize("reproject_to_4326", (True, False))
def test_load_embedding_item_parquet_no_bbox(reproject_to_4326: bool):
    item = pystac.Item.from_file("tests/data/item_pq.json")
    asset_name = "embeddings"

    e_dc = _load_embedding_item(item, asset_name, None, None, reproject_to_4326)

    assert e_dc.shape == (1, 4, 4)
    assert "time" in e_dc.dims
    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)

    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    geom_coords = e_dc.coords["geometry"].values

    if reproject_to_4326:
        assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))
        outside = ~poly_utm.contains(geom_coords)
        inside = poly_wgs84.contains(geom_coords)
    else:
        assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))
        inside = poly_utm.contains(geom_coords)
        outside = ~poly_wgs84.contains(geom_coords)

    assert inside.all()
    assert outside.all()


@pytest.mark.parametrize("reproject_to_4326", (True, False))
def test_load_embedding_item_parquet_with_bbox(reproject_to_4326: bool):
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )

    item = pystac.Item.from_file("tests/data/item_pq.json")
    asset_name = "embeddings"

    e_dc = _load_embedding_item(item, asset_name, bbox, None, reproject_to_4326)

    assert e_dc.shape == (1, 2, 4)
    assert "time" in e_dc.dims
    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)

    poly_utm: shapely.Polygon = shapely.box(404000, 5756000, 406000, 5758000)
    poly_wgs84 = shapely.box(0, 0, 180, 90)

    geom_coords = e_dc.coords["geometry"].values

    if reproject_to_4326:
        assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))
        outside = ~poly_utm.contains(geom_coords)
        inside = poly_wgs84.contains(geom_coords)
    else:
        assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:32632"))
        inside = poly_utm.contains(geom_coords)
        outside = ~poly_wgs84.contains(geom_coords)

    assert inside.all()
    assert outside.all()


def test_construct_embedding_vector_cube():
    x = xr.DataArray(da.random.random(4))
    i = [[x, x], [x, x], [x, xr.DataArray()]]
    geoms = [shapely.Point(0, 0), shapely.Point(1, 1)]
    times = [
        datetime.fromisoformat("2026-01-01"),
        datetime.fromisoformat("2026-01-02"),
        datetime.fromisoformat("2026-01-03"),
    ]

    e_dc = _construct_embedding_vector_cube(i, geoms, times)

    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert e_dc.shape == (3, 2, 4)
    assert isinstance(e_dc.data, da.Array)
    assert np.isnan(
        e_dc.isel(time=2, geometry=1).compute()
    ).all()  # check nans are preserved
    assert not np.isnan(
        e_dc.isel(time=1, geometry=1).compute()
    ).all()  # check other is not nan


def test_crs_of():
    path = "tests/data/embeddings.parquet"
    s = _crs_of(path)
    assert s is not None
    assert isinstance(s, str)

    path = "tests/data/non-geoparquet.parquet"
    with pytest.raises(Exception):
        _crs_of(path)


def test_load_collection_tif():
    poly_coords = [
        [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        [[[0, 1], [0, 2], [1, 2], [1, 2], [0, 1]]],
        [[[1, 0], [1, 1], [2, 1], [2, 0], [1, 0]]],
        [[[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]],
    ]
    dates = [datetime(2020, 6, 1), datetime(2020, 7, 1)]

    items = []
    i = 0
    for poly in poly_coords:
        for date in dates:
            items.append(
                _make_item(
                    str(i), date, poly, "tests/data/embedding_1x1.tif", "image/tif"
                )
            )
            i += 1
    item_collection = _make_item_collection(items)

    e_dc = _load_embedding_collection_tif(item_collection, "embedding")

    assert e_dc.shape == (2, 4, 768)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims

    assert all([isinstance(i, shapely.Polygon) for i in e_dc.coords["geometry"].values])
    assert all([isinstance(i, np.datetime64) for i in e_dc.coords["time"].values])
    assert isinstance(e_dc.data, da.Array)

    assert e_dc.geometry.crs.equals(pyproj.CRS("EPSG:4326"))


@pytest.mark.vcr()
def test_load_embedding_collection_tif():
    # this should work regardless of a local STAC running
    # the VCR module records and replays the request
    url = "http://localhost:8082/collections/terramind_embeddings"
    bbox = BoundingBox(
        west=8.38, south=48.02, east=8.38 + 2 * 0.04, north=48.02 + 2 * 0.04
    )
    time = TemporalInterval(["2025-01-01", "2025-01-03"])

    coll = pystac.Collection.from_file(url)
    asset_name = "embeddings"

    e_dc = _load_embedding_collection(url, coll, bbox, time, asset_name)

    assert e_dc.shape == (3, 9, 768)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)


@pytest.mark.vcr()
def test_load_embeddings_tif_collection():
    # this should work regardless of a local STAC running
    # the VCR module recorded and now replays the request
    url = "http://localhost:8082/collections/terramind_embeddings"
    bbox = BoundingBox(
        west=8.38, south=48.02, east=8.38 + 2 * 0.04, north=48.02 + 2 * 0.04
    )
    time = TemporalInterval(["2025-01-01", "2025-01-02"])
    asset_name = "embeddings"

    e_dc = load_embeddings(url, bbox, time, asset_name)

    assert e_dc.shape == (2, 9, 768)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)


@pytest.mark.vcr()
def test_load_embeddings_tif_singleitem():
    # this should work regardless of a local STAC running
    # the VCR module recorded and now replays the request
    url = (
        "http://localhost:8082/collections/terramind_embeddings/items/"
        "S2A_MSIL2A_20170102T175732_N0500_R141_T13TEF_20230926T044006__"
        "0-256_10724-10980_embedding_74"
    )
    asset_name = "embeddings"

    e_dc = load_embeddings(url, asset_name=asset_name)
    assert e_dc.shape == (1, 1, 768)
    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert "time" in e_dc.dims
    assert isinstance(e_dc.data, da.Array)


def test_load_embedding_collection_parquet():
    coll = pystac.Collection(
        "asdf",
        "asdf",
        pystac.Extent(
            pystac.SpatialExtent([0, 0, 1, 1]),
            pystac.TemporalExtent([datetime(2025, 1, 1)]),
        ),
        extra_fields={"proj:code": "EPSG:32632"},
    )
    items = []
    dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
    poly = [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
    i = 0
    for date in dates:
        items.append(
            _make_item(
                str(i),
                date,
                poly,
                "tests/data/embeddings.parquet",
                "application/vnd.apache.parquet",
            )
        )
        i += 1
    item_collection = _make_item_collection(items)
    e_dc = _load_embedding_collection_parquet(coll, item_collection, "embedding", None)
    assert e_dc.shape == (2, 4, 4)
    assert "embedding" in e_dc.dims
    assert "time" in e_dc.dims
    assert "geometry" in e_dc.dims


@pytest.mark.vcr()
def test_load_embedding_collection_parquetcollection():
    url = "http://localhost:8082/collections/terramind_embeddings"
    coll = pystac.Collection.from_file(url)
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )
    temp = TemporalInterval(["2025-01-01", "2025-01-03"])

    e_dc = _load_embedding_collection(url, coll, bbox, temp, "embeddings")
    assert e_dc.shape == (3, 2, 4)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims


@pytest.mark.vcr()
def test_load_embeddings_parquet_collection():
    url = "http://localhost:8082/collections/terramind_embeddings"
    bbox = BoundingBox(
        east=7.626471282171344,
        south=51.95677890608484,
        west=7.628845144763302,
        north=51.95733618982899,
        crs="EPSG:4326",
    )
    temp = TemporalInterval(["2025-01-01", "2025-01-03"])

    e_dc = load_embeddings(url, bbox, temp, "embeddings")
    assert e_dc.shape == (3, 2, 4)
    assert "geometry" in e_dc.dims
    assert "time" in e_dc.dims
    assert "embedding" in e_dc.dims


@pytest.mark.vcr()
def test_load_embeddings_parquet_singleitem():
    url = "http://localhost:8082/collections/terramind_embeddings/items/emb_0"
    e_dc = load_embeddings(url, asset_name="embeddings")

    assert e_dc.shape == (1, 4, 4)
    assert "geometry" in e_dc.dims
    assert "embedding" in e_dc.dims
    assert "time" in e_dc.dims
