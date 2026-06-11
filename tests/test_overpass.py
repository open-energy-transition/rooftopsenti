import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from rooftopsenti.config import Config
from rooftopsenti.overpass import elements_to_polygons, solar_query
from rooftopsenti.stages.osm_labels import build_labels

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def elements():
    return json.loads((FIXTURES / "overpass_solar.json").read_text())["elements"]


def _cfg(rule="intersect_or_roof_tag") -> Config:
    return Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.0, 51.0, 7.0, 52.0]},
            "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
            "osm": {"rooftop_rule": rule, "solar_area_min_m2": 1000},
        }
    )


def test_elements_to_polygons(elements):
    gdf = elements_to_polygons(elements)
    # closed way + relation survive; open way, node, and duplicate way dropped
    assert len(gdf) == 2
    assert set(gdf["osm_type"]) == {"way", "relation"}
    assert gdf.crs.to_epsg() == 4326

    way = gdf[gdf["osm_type"] == "way"].iloc[0]
    assert way["location_tag"] == "roof"

    rel = gdf[gdf["osm_type"] == "relation"].iloc[0]
    # inner ring subtracted -> area smaller than the full outer ring
    outer_only = box(6.20, 51.45, 6.202, 51.452)
    assert rel.geometry.area < outer_only.area


def test_solar_query_bbox_order():
    ql = solar_query((6.0, 51.0, 7.0, 52.0), 180)
    assert "(51.0,6.0,52.0,7.0)" in ql  # Overpass wants S,W,N,E
    assert '["generator:source"="solar"]' in ql
    assert "out body geom;" in ql


def test_build_labels_rules(elements):
    solar = elements_to_polygons(elements)
    # building overlapping the relation polygon only
    buildings = gpd.GeoDataFrame(
        geometry=[box(6.199, 51.449, 6.203, 51.453)], crs="EPSG:4326"
    )

    labels = build_labels(solar, buildings, _cfg("intersect_or_roof_tag"))
    assert len(labels) == 2  # way via roof tag, relation via building intersect
    assert labels["area_m2"].min() >= 1000

    roof_only = build_labels(solar, buildings, _cfg("roof_tag_only"))
    assert len(roof_only) == 1
    assert roof_only.iloc[0]["osm_type"] == "way"

    intersect_only = build_labels(solar, buildings, _cfg("intersect_only"))
    assert len(intersect_only) == 1
    assert intersect_only.iloc[0]["osm_type"] == "relation"


def test_build_labels_empty_inputs():
    empty = elements_to_polygons([])
    labels = build_labels(empty, empty, _cfg())
    assert labels.empty


def _solar(geoms, location_tags=None):
    n = len(geoms)
    return gpd.GeoDataFrame(
        {
            "osm_type": ["way"] * n,
            "osm_id": list(range(n)),
            "tags": ["{}"] * n,
            "location_tag": location_tags or [None] * n,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )


# two ~700 m² panels (each < 1000 m² threshold), not touching, both on one building
_SMALL_A = box(6.0002, 51.0002, 6.0005, 51.0005)
_SMALL_B = box(6.0014, 51.0014, 6.0017, 51.0017)
_BUILDING = gpd.GeoDataFrame(geometry=[box(6.0, 51.0, 6.002, 51.002)], crs="EPSG:4326")


def test_subdivided_array_grouped_per_building():
    # individually sub-threshold, but their per-building union clears 1000 m²
    labels = build_labels(_solar([_SMALL_A, _SMALL_B]), _BUILDING, _cfg())
    assert len(labels) == 1
    assert labels.iloc[0]["area_m2"] >= 1000


def test_subdivided_array_without_building_stays_subthreshold():
    # no building + the panels don't touch -> no union -> each stays < threshold
    no_bldg = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    assert build_labels(_solar([_SMALL_A, _SMALL_B]), no_bldg, _cfg()).empty


def test_multiple_generators_collapse_to_one_label():
    # two large (~1944 m²) generators on the same roof -> one label, not two
    big = _solar([box(6.0002, 51.0002, 6.0007, 51.0007),
                  box(6.0012, 51.0012, 6.0017, 51.0017)])
    assert len(build_labels(big, _BUILDING, _cfg())) == 1


def test_solar_on_distinct_buildings_stay_separate():
    buildings = gpd.GeoDataFrame(
        geometry=[box(6.0, 51.0, 6.001, 51.001), box(6.0015, 51.0015, 6.0025, 51.0025)],
        crs="EPSG:4326",
    )
    solar = _solar([box(6.0003, 51.0003, 6.0008, 51.0008),
                    box(6.0017, 51.0017, 6.0022, 51.0022)])
    assert len(build_labels(solar, buildings, _cfg())) == 2
