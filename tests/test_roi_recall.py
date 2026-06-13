"""Tests for the recall-first ROI changes: thresholds, dedupe, screen head."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import box

from rooftopsenti.config import BuildingsConfig, Config, PostprocessConfig, ScreenConfig
from rooftopsenti.embeddings import LinearHead, embed
from rooftopsenti.stages.buildings import dedupe_sources


def test_roi_threshold_defaults_to_building_min():
    cfg = BuildingsConfig(building_area_min_m2=1000.0)
    assert cfg.effective_roi_area_min_m2 == 1000.0
    assert cfg.fetch_area_min_m2 == 1000.0


def test_roi_threshold_lowers_fetch_min():
    cfg = BuildingsConfig(building_area_min_m2=1000.0, roi_area_min_m2=200.0)
    assert cfg.effective_roi_area_min_m2 == 200.0
    # the artifact must keep everything ROIs *or* negatives need
    assert cfg.fetch_area_min_m2 == 200.0


def test_postprocess_default_is_recall_first():
    # detections are human-validated downstream, so the default favours recall
    assert PostprocessConfig().prob_threshold < 0.5


def test_screen_config_defaults():
    assert ScreenConfig().stride_frac == 1.0
    assert Config.model_validate(
        {
            "region": "t",
            "aoi": {"source": "bbox", "bbox": [6.1, 51.3, 6.25, 51.45]},
            "imagery": {"date_ranges": [["2023-06-01", "2023-08-31"]]},
        }
    ).screen.prob_threshold == 0.5


def test_buildings_config_new_fields_default_safe():
    cfg = BuildingsConfig()
    assert cfg.sources == ["overture"]
    assert cfg.roi_buffer_m == 0.0
    assert cfg.use_screen_candidates is False
    assert "{iso3}" in cfg.vida_url_template
    # the legacy single-source attribute must stay gone
    assert not hasattr(cfg, "source")


def _grid(n: int, source: str, offset: float = 0.0) -> gpd.GeoDataFrame:
    geoms = [box(i + offset, 0, i + offset + 0.5, 0.5) for i in range(n)]
    return gpd.GeoDataFrame(
        {"id": [f"{source}_{i}" for i in range(n)], "source": source},
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_dedupe_keeps_priority_source_and_drops_overlaps():
    primary = _grid(3, "overture")  # boxes at x=0,1,2
    # secondary: one exact overlap (x=0) + two fresh (x=10, 11)
    secondary = gpd.GeoDataFrame(
        {"id": ["v0", "v1", "v2"], "source": "vida"},
        geometry=[box(0.1, 0.1, 0.4, 0.4), box(10, 0, 10.5, 0.5), box(11, 0, 11.5, 0.5)],
        crs="EPSG:4326",
    )
    out = dedupe_sources([primary, secondary])
    assert len(out) == 5  # 3 primary + 2 fresh
    assert (out["source"] == "overture").sum() == 3
    assert (out["source"] == "vida").sum() == 2


def test_dedupe_handles_empty_frames():
    empty = gpd.GeoDataFrame({"id": [], "source": []}, geometry=[], crs="EPSG:4326")
    out = dedupe_sources([empty, _grid(2, "vida")])
    assert len(out) == 2
    assert dedupe_sources([empty, empty]).empty


def test_embed_pools_deepest_feature_map():
    # a fake encoder returning a list of feature maps, like smp encoders do
    def encoder(x):
        n = x.shape[0]
        return [torch.ones(n, 4, 8, 8), torch.arange(n * 16 * 2 * 2).float().reshape(n, 16, 2, 2)]

    out = embed(encoder, torch.zeros(3, 10, 16, 16))
    assert out.shape == (3, 16)  # pooled deepest map (16 channels)


def test_linear_head_standardizes_and_scores():
    head = LinearHead(5)
    head.mean.copy_(torch.arange(5).float())
    head.std.copy_(torch.full((5,), 2.0))
    out = head(torch.arange(5).float().unsqueeze(0))  # equals mean -> standardized to 0
    # standardized input is all zeros, so logit == bias
    assert torch.allclose(out, head.linear.bias)
    assert out.shape == (1,)


def test_linear_head_roundtrips_buffers():
    head = LinearHead(3)
    head.mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    head.std.copy_(torch.tensor([4.0, 5.0, 6.0]))
    reloaded = LinearHead(3)
    reloaded.load_state_dict(head.state_dict())
    assert torch.allclose(reloaded.mean, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(reloaded.std, torch.tensor([4.0, 5.0, 6.0]))


def test_embed_array_label_shapes_are_consistent():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    labels = (rng.random(20) > 0.5).astype(int)
    assert emb.shape[0] == labels.shape[0]
