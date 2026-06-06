import numpy as np
import pystac
import pytest
import rasterio

from rooftopsenti.stac_catalog import composite_assets, register_composite

UTM = "EPSG:32632"


@pytest.fixture
def tiny_cog(tmp_path):
    path = tmp_path / "composites" / "32ULB" / "composite_0.tif"
    path.parent.mkdir(parents=True)
    transform = rasterio.transform.from_origin(700000, 5700000, 10, 10)
    data = np.ones((4, 32, 32), dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", width=32, height=32, count=4,
        dtype="uint16", crs=UTM, transform=transform, nodata=0,
    ) as dst:
        dst.write(data)
    return path


def _register(catalog_path, cog):
    return register_composite(
        catalog_path, "testregion", "32ULB", 0, cog,
        ("2023-04-01", "2023-09-30"), ["B02", "B03", "B04", "B08"],
    )


def test_register_and_lookup(tmp_path, tiny_cog):
    catalog_path = tmp_path / "composites" / "catalog.json"
    item = _register(catalog_path, tiny_cog)
    assert item.id == "32ULB_range0"
    assert pystac.Catalog.from_file(str(catalog_path)).get_item("32ULB_range0")
    assert composite_assets(catalog_path) == {("32ULB", 0): tiny_cog.resolve()}


def test_corrupt_catalog_is_rebuilt(tmp_path, tiny_cog):
    """A crash mid-write can leave binary garbage — registration must recover."""
    catalog_path = tmp_path / "composites" / "catalog.json"
    catalog_path.write_bytes(b"\xdb\x28\xe4\x77\xab\xcf")  # not UTF-8, not JSON
    _register(catalog_path, tiny_cog)
    assert composite_assets(catalog_path) == {("32ULB", 0): tiny_cog.resolve()}


def test_register_is_idempotent(tmp_path, tiny_cog):
    catalog_path = tmp_path / "composites" / "catalog.json"
    _register(catalog_path, tiny_cog)
    _register(catalog_path, tiny_cog)  # re-register (e.g. fresh-skip path)
    items = pystac.Catalog.from_file(str(catalog_path)).get_items(recursive=True)
    assert [i.id for i in items] == ["32ULB_range0"]


def test_no_tmp_files_left_behind(tmp_path, tiny_cog):
    catalog_path = tmp_path / "composites" / "catalog.json"
    _register(catalog_path, tiny_cog)
    assert not list(tmp_path.rglob("*.tmp"))  # atomic writes renamed away
