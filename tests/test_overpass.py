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
