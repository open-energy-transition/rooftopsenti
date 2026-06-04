import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.windows
from shapely.geometry import box

from rooftopsenti.config import Config
from rooftopsenti.stages.chips import (
    _assign_splits,
    _jittered_window,
    _rasterize_labels,
    _read_chip,
    _solar_free_buildings,
)

UTM = "EPSG:32632"  # zone covering the Venlo smoke area


@pytest.fixture
def synthetic_raster(tmp_path):
    """A 512x512 10-band uint16 raster at 10 m resolution in UTM."""
    path = tmp_path / "comp.tif"
    transform = rasterio.transform.from_origin(700000, 5700000, 10, 10)
    data = np.random.default_rng(0).integers(1, 5000, (10, 512, 512), dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", width=512, height=512, count=10,
        dtype="uint16", crs=UTM, transform=transform, nodata=0,
    ) as dst:
        dst.write(data)
    return path


def _cfg() -> Config:
    return Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.0, 51.0, 7.0, 52.0]},
            "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
            "split": {"block_size_km": 1, "seed": 1},
        }
    )


def test_jittered_window_and_read(synthetic_raster):
    rng = np.random.default_rng(0)
    with rasterio.open(synthetic_raster) as src:
        x, y = src.xy(256, 256)
        w = _jittered_window(src, x, y, 128, rng)
        assert (w.width, w.height) == (128, 128)
        chip = _read_chip(src, w)
        assert chip is not None and chip.shape == (10, 128, 128)

        # window fully outside data -> all fill -> rejected
        far = rasterio.windows.Window(-1000, -1000, 128, 128)
        assert _read_chip(src, far) is None


def test_rasterize_labels(synthetic_raster):
    with rasterio.open(synthetic_raster) as src:
        # a label polygon covering exactly a 100 m x 100 m square (10x10 px)
        x0, y0 = 700100, 5699900
        labels = gpd.GeoDataFrame(geometry=[box(x0, y0 - 100, x0 + 100, y0)], crs=UTM)
        window = rasterio.windows.Window(0, 0, 64, 64)
        mask = _rasterize_labels(labels, window, src)
        assert mask.shape == (64, 64)
        assert mask.sum() == 100  # 10x10 pixels

        empty = _rasterize_labels(labels, rasterio.windows.Window(400, 400, 64, 64), src)
        assert empty.sum() == 0


def test_solar_free_buildings():
    buildings = gpd.GeoDataFrame(
        geometry=[
            box(6.000, 51.000, 6.001, 51.001),  # overlaps solar
            box(6.100, 51.100, 6.101, 51.101),  # far away
        ],
        crs="EPSG:4326",
    )
    solar = gpd.GeoDataFrame(geometry=[box(6.0005, 51.0005, 6.0008, 51.0008)], crs="EPSG:4326")
    free = _solar_free_buildings(buildings, solar)
    assert len(free) == 1
    assert free.geometry.iloc[0].bounds[0] == 6.100


def test_assign_splits_disjoint_blocks():
    rng = np.random.default_rng(0)
    # 200 chips spread over a ~30 km extent -> many 1 km blocks
    xs = rng.uniform(6.0, 6.4, 200)
    ys = rng.uniform(51.0, 51.3, 200)
    index = gpd.GeoDataFrame(
        geometry=[box(x, y, x + 0.002, y + 0.002) for x, y in zip(xs, ys, strict=True)],
        crs="EPSG:4326",
    )
    out = _assign_splits(index, _cfg())
    assert set(out["split"]) <= {"train", "val", "test"}
    assert (out["split"] == "train").sum() > 0
    assert (out["split"] == "val").sum() > 0
    # no block appears in two splits
    assert (out.groupby("block")["split"].nunique() == 1).all()
