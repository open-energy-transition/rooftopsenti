from pathlib import Path

import pytest
from pydantic import ValidationError

from rooftopsenti.config import Config, load_config

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("name", ["netherlands", "smoke_nl_tile", "pakistan", "vietnam"])
def test_shipped_configs_load(name):
    cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
    assert cfg.region == name.replace("smoke_nl_tile", "smoke_nl_tile")
    assert cfg.in_channels == len(cfg.imagery.bands) == 10


def test_bbox_source_requires_bbox():
    with pytest.raises(ValidationError, match="bbox"):
        Config.model_validate(
            {
                "region": "x",
                "aoi": {"source": "bbox"},
                "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
            }
        )


def test_invalid_bbox_order():
    with pytest.raises(ValidationError, match="W<E"):
        Config.model_validate(
            {
                "region": "x",
                "aoi": {"source": "bbox", "bbox": [7.0, 51.0, 6.0, 52.0]},
                "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
            }
        )


def test_invalid_date_range():
    with pytest.raises(ValidationError, match="precede"):
        Config.model_validate(
            {
                "region": "x",
                "aoi": {"source": "bbox", "bbox": [6.0, 51.0, 7.0, 52.0]},
                "imagery": {"date_ranges": [["2023-02-01", "2023-01-01"]]},
            }
        )


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Config.model_validate(
            {
                "region": "x",
                "aoi": {"source": "bbox", "bbox": [6.0, 51.0, 7.0, 52.0]},
                "imagery": {"date_ranges": [["2023-01-01", "2023-02-01"]]},
                "split": {"ratios": {"train": 0.5, "val": 0.1, "test": 0.1}},
            }
        )


def test_run_id_stable_and_sensitive(tmp_path):
    cfg1 = load_config(CONFIGS_DIR / "netherlands.yaml")
    cfg2 = load_config(CONFIGS_DIR / "netherlands.yaml")
    assert cfg1.run_id() == cfg2.run_id()

    changed = cfg1.model_copy(deep=True)
    changed.model.lr = 9e-9
    assert changed.run_id() != cfg1.run_id()

    # postprocess thresholds do NOT affect the trained-model run id
    pp = cfg1.model_copy(deep=True)
    pp.postprocess.prob_threshold = 0.9
    assert pp.run_id() == cfg1.run_id()
