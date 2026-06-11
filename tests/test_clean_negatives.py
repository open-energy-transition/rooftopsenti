import pandas as pd

from rooftopsenti.config import ChipsConfig, CleanNegativesConfig, Config
from rooftopsenti.datamodules import drop_cleaned_out


def _cfg(**clean):
    return Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.1, 51.3, 6.25, 51.45]},
            "imagery": {"date_ranges": [["2023-06-01", "2023-08-31"]]},
            "clean_negatives": clean,
        }
    )


def test_config_defaults():
    assert ChipsConfig().solar_free_buffer_m == 50.0
    cn = CleanNegativesConfig()
    assert cn.prob_threshold == 0.5
    assert cn.max_solar_fraction == 0.05


def test_clean_negatives_not_in_run_id():
    # cleaning is a manual post-hoc step on the chip index, so it must not change
    # the auto run id (which identifies the trained model's data/hyperparams)
    assert _cfg(max_solar_fraction=0.05).run_id() == _cfg(max_solar_fraction=0.5).run_id()


def test_solar_free_buffer_changes_run_id():
    # buffer affects which chips are generated -> must invalidate the run
    a = Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.1, 51.3, 6.25, 51.45]},
            "imagery": {"date_ranges": [["2023-06-01", "2023-08-31"]]},
            "chips": {"solar_free_buffer_m": 50.0},
        }
    )
    b = a.model_copy(update={"chips": ChipsConfig(solar_free_buffer_m=100.0)})
    assert a.run_id() != b.run_id()


def test_drop_cleaned_out_filters_flagged_rows():
    df = pd.DataFrame(
        {"name": ["a", "b", "c"], "kind": ["pos", "neg", "neg"],
         "cleaned_out": [False, True, False]}
    )
    out = drop_cleaned_out(df)
    assert list(out["name"]) == ["a", "c"]


def test_drop_cleaned_out_missing_column_is_noop():
    df = pd.DataFrame({"name": ["a", "b"], "kind": ["pos", "neg"]})
    out = drop_cleaned_out(df)
    assert list(out["name"]) == ["a", "b"]


def test_drop_cleaned_out_handles_nan():
    # concat of cleaned + un-cleaned region indices leaves NaN in the column
    df = pd.DataFrame({"name": ["a", "b"], "cleaned_out": [True, None]})
    out = drop_cleaned_out(df)
    assert list(out["name"]) == ["b"]
