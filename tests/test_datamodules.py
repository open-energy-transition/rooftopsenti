import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from shapely.geometry import box

from rooftopsenti.config import Config
from rooftopsenti.datamodules import ChipDataset, load_chip_index
from rooftopsenti.io_artifacts import ArtifactStore
from rooftopsenti.stages import pack_chips

UTM = "EPSG:32631"


def _cfg(tmp_path, region="nl", train_regions=()):
    return Config.model_validate(
        {
            "region": region,
            "aoi": {"source": "bbox", "bbox": [4.0, 51.0, 5.0, 52.0]},
            "imagery": {"date_ranges": [["2023-01-01", "2023-12-31"]]},
            "model": {"train_regions": list(train_regions)},
            "data_dir": str(tmp_path),
        }
    )


def _write_region_chips(tmp_path, region, n, n_bands, splits):
    chips_dir = tmp_path / region / "chips"
    (chips_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    transform = rasterio.transform.from_origin(500000, 5700000, 10, 10)
    for i in range(n):
        img = chips_dir / "images" / f"chip_{i}.tif"
        with rasterio.open(
            img, "w", driver="GTiff", width=8, height=8, count=n_bands,
            dtype="uint16", crs=UTM, transform=transform,
        ) as dst:
            dst.write(np.ones((n_bands, 8, 8), dtype=np.uint16))
        rows.append(
            {"image": str(img), "mask": str(img), "split": splits[i % len(splits)],
             "geometry": box(4.0 + i, 51.0, 4.1 + i, 51.1)}
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_parquet(chips_dir / "index.parquet")


def test_multi_region_merge_and_primary_test_split(tmp_path):
    _write_region_chips(tmp_path, "nl", 4, 3, ["train", "val", "test", "train"])
    _write_region_chips(tmp_path, "de", 2, 3, ["train", "train"])
    cfg = _cfg(tmp_path, "nl", ["de"])
    index = load_chip_index(cfg, ArtifactStore(cfg))
    assert len(index) == 6
    assert set(index["region"]) == {"nl", "de"}
    # extra-region chips never land in the primary test pool
    test_rows = index[(index["split"] == "test")]
    assert set(test_rows["region"]) == {"nl"}


def test_missing_extra_region_raises(tmp_path):
    _write_region_chips(tmp_path, "nl", 2, 3, ["train", "val"])
    cfg = _cfg(tmp_path, "nl", ["de"])
    with pytest.raises(FileNotFoundError, match="configs/de.yaml"):
        load_chip_index(cfg, ArtifactStore(cfg))


def test_channel_mismatch_raises(tmp_path):
    _write_region_chips(tmp_path, "nl", 2, 3, ["train", "val"])
    _write_region_chips(tmp_path, "de", 2, 5, ["train", "train"])
    cfg = _cfg(tmp_path, "nl", ["de"])
    with pytest.raises(ValueError, match="channels"):
        load_chip_index(cfg, ArtifactStore(cfg))


def test_packed_h5_matches_tiff_reads(tmp_path):
    _write_region_chips(tmp_path, "nl", 3, 4, ["train", "val", "train"])
    cfg = _cfg(tmp_path, "nl")
    store = ArtifactStore(cfg)
    pack_chips.run(cfg, store)

    index = load_chip_index(cfg, store)
    assert index["h5_path"].notna().all()
    ds_h5 = ChipDataset(index, augment=False)
    ds_tif = ChipDataset(index.drop(columns=["h5_path", "h5_row"]), augment=False)
    for i in range(len(ds_h5)):
        assert torch.equal(ds_h5[i]["image"], ds_tif[i]["image"])
        assert torch.equal(ds_h5[i]["mask"], ds_tif[i]["mask"])


def test_stale_h5_pack_falls_back_to_tiffs(tmp_path):
    _write_region_chips(tmp_path, "nl", 2, 3, ["train", "val"])
    cfg = _cfg(tmp_path, "nl")
    store = ArtifactStore(cfg)
    pack_chips.run(cfg, store)

    # rewriting the index (e.g. a chips re-run) must invalidate the pack
    _write_region_chips(tmp_path, "nl", 2, 3, ["train", "val"])
    index = load_chip_index(cfg, store)
    assert index["h5_path"].isna().all()
    sample = ChipDataset(index, augment=False)[0]
    assert sample["image"].shape == (3, 8, 8)


def test_multi_region_pack_rows_survive_concat(tmp_path):
    # only one of two regions is packed; rows must keep per-region h5 rows
    _write_region_chips(tmp_path, "nl", 3, 3, ["train", "val", "test"])
    _write_region_chips(tmp_path, "de", 2, 3, ["train", "train"])
    cfg = _cfg(tmp_path, "nl", ["de"])
    store = ArtifactStore(cfg)
    pack_chips.run(cfg, store)

    index = load_chip_index(cfg, store)
    nl = index[index["region"] == "nl"]
    de = index[index["region"] == "de"]
    assert nl["h5_path"].notna().all()
    assert list(nl["h5_row"]) == [0, 1, 2]
    assert de["h5_path"].isna().all()


def test_train_regions_change_run_id(tmp_path):
    base = _cfg(tmp_path, "nl")
    multi = _cfg(tmp_path, "nl", ["de", "pk"])
    assert base.run_id() != multi.run_id()
