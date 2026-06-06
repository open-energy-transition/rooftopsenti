import geopandas as gpd
import numpy as np
from shapely.geometry import box

from rooftopsenti import geo


def test_area_m2_roughly_correct():
    # ~0.001° x 0.001° box at 52°N: 111320*0.001 * 111320*cos(52)*0.001 ≈ 7630 m²
    g = gpd.GeoDataFrame(geometry=[box(6.0, 52.0, 6.001, 52.001)], crs="EPSG:4326")
    area = geo.area_m2(g).iloc[0]
    assert 6500 < area < 8500


def test_filter_min_area():
    small = box(6.0, 52.0, 6.0001, 52.0001)  # ~76 m²
    large = box(6.0, 52.0, 6.001, 52.001)  # ~7600 m²
    g = gpd.GeoDataFrame({"name": ["small", "large"]}, geometry=[small, large], crs="EPSG:4326")
    out = geo.filter_min_area(g, 1000)
    assert list(out["name"]) == ["large"]
    assert "area_m2" in out.columns


def test_utm_epsg():
    assert geo.utm_epsg_for(5.5, 52.0) == 32631  # Netherlands
    assert geo.utm_epsg_for(67.0, 30.0) == 32642  # Pakistan
    assert geo.utm_epsg_for(106.0, -6.0) == 32748  # southern hemisphere


def test_mgrs_tiles_for_geometry_venlo():
    # Venlo at 6.1°E sits just inside UTM zone 32, near 100 km cell seams
    tiles = geo.mgrs_tiles_for_geometry(box(6.10, 51.30, 6.25, 51.45))
    assert "32ULB" in tiles  # cell containing the bbox centre
    assert all(len(t) == 5 for t in tiles)
    assert len(tiles) <= 6


def test_mgrs_tile_polygon_roundtrip():
    import mgrs

    m = mgrs.MGRS()
    for tile in ("32ULB", "31UFT", "42RVR"):  # NL (two zones) + Pakistan
        poly = geo.mgrs_tile_polygon(tile)
        # the cell centre must map back to the same tile name
        c = poly.centroid
        assert m.toMGRS(c.y, c.x, MGRSPrecision=0) == tile
        # ~100x100 km
        area = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(geo.EQUAL_AREA).area.iloc[0]
        assert 0.9e10 < area < 1.1e10


def test_bbox_grid_covers_bounds():
    cells = geo.bbox_grid((6.0, 51.0, 7.3, 52.2), 0.5)
    assert len(cells) == 3 * 3
    xs = [c[0] for c in cells] + [c[2] for c in cells]
    ys = [c[1] for c in cells] + [c[3] for c in cells]
    assert min(xs) == 6.0 and max(xs) == 7.3
    assert min(ys) == 51.0 and max(ys) == 52.2


def test_spatial_block_id_distinct_blocks():
    # two points 30 km apart in x -> different 25 km blocks; 1 km apart -> same
    ids = geo.spatial_block_id([0.0, 30_000.0, 500.0], [0.0, 0.0, 0.0], block_size_km=25)
    assert ids[0] != ids[1]
    assert ids[0] == ids[2]
    assert ids.dtype == np.int64
