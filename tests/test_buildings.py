from rooftopsenti.config import BuildingsConfig, Config
from rooftopsenti.overture import theme_path


def test_buildings_config_defaults():
    cfg = BuildingsConfig()
    assert cfg.building_area_min_m2 == 1000.0
    # the GBA/huggingface backend is gone — buildings come only from Overture
    assert not hasattr(cfg, "source")


def test_config_buildings_section():
    cfg = Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.1, 51.3, 6.25, 51.45]},
            "imagery": {"date_ranges": [["2023-06-01", "2023-08-31"]]},
            "buildings": {"building_area_min_m2": 500},
        }
    )
    assert cfg.buildings.building_area_min_m2 == 500


def test_buildings_theme_path():
    assert theme_path("2026-05-20.0", "buildings", "building") == (
        "s3://overturemaps-us-west-2/release/2026-05-20.0/theme=buildings/type=building/*"
    )
