import geopandas as gpd
from shapely.geometry import box

from rooftopsenti.config import Config
from rooftopsenti.io_artifacts import ArtifactStore, read_gdf, write_gdf


def _cfg(tmp_path) -> Config:
    return Config.model_validate(
        {
            "region": "testregion",
            "aoi": {"source": "bbox", "bbox": [6.0, 51.0, 7.0, 52.0]},
            "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
            "data_dir": str(tmp_path / "data"),
        }
    )


def test_paths_are_region_scoped(tmp_path):
    store = ArtifactStore(_cfg(tmp_path))
    assert "testregion" in str(store.aoi_boundary)
    assert store.composite_tif("31UFT").name == "composite_0.tif"


def test_gdf_roundtrip(tmp_path):
    store = ArtifactStore(_cfg(tmp_path))
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    write_gdf(gdf, store.aoi_boundary)
    back = read_gdf(store.aoi_boundary)
    assert back.crs.to_epsg() == 4326
    assert back.iloc[0]["a"] == 1


def test_freshness_lifecycle(tmp_path):
    cfg = _cfg(tmp_path)
    store = ArtifactStore(cfg)
    artifact = store.osm_labels
    cfg_slice = {"solar_area_min_m2": 1000, "bbox": (6.0, 51.0, 7.0, 52.0)}

    assert not store.is_fresh(artifact, cfg_slice)

    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    write_gdf(gdf, artifact)
    store.write_meta(artifact, cfg_slice)
    assert store.is_fresh(artifact, cfg_slice)

    # config change invalidates
    assert not store.is_fresh(artifact, {**cfg_slice, "solar_area_min_m2": 2000})


def test_freshness_tracks_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    store = ArtifactStore(cfg)
    upstream = store.aoi_boundary
    artifact = store.osm_labels
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    write_gdf(gdf, upstream)
    write_gdf(gdf, artifact)
    store.write_meta(artifact, {}, inputs=[upstream])
    assert store.is_fresh(artifact, {}, inputs=[upstream])

    # touching the upstream artifact invalidates the downstream one
    write_gdf(gdf, upstream)
    assert not store.is_fresh(artifact, {}, inputs=[upstream])
