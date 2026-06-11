"""Region configuration: pydantic schema, YAML loader, stable run id."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

DateRange = tuple[str, str]
BBox = tuple[float, float, float, float]  # W, S, E, N


class AOIConfig(BaseModel):
    source: Literal["geoboundaries", "overpass_admin", "bbox"] = "geoboundaries"
    iso3: str | None = None
    admin_level: int = 0
    admin_name: str | None = None
    bbox: BBox | None = None

    @model_validator(mode="after")
    def _check_source_args(self) -> AOIConfig:
        if self.source == "bbox" and self.bbox is None:
            raise ValueError("aoi.source=bbox requires aoi.bbox")
        if self.source == "geoboundaries" and self.iso3 is None:
            raise ValueError("aoi.source=geoboundaries requires aoi.iso3")
        if self.bbox is not None:
            w, s, e, n = self.bbox
            if not (w < e and s < n):
                raise ValueError(f"aoi.bbox must be (W, S, E, N) with W<E and S<N, got {self.bbox}")
        return self


class OvertureConfig(BaseModel):
    release: str = "2026-05-20.0"


class OSMConfig(BaseModel):
    source: Literal["overture", "overpass"] = "overture"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    solar_area_min_m2: float = 1000.0
    rooftop_rule: Literal["intersect_or_roof_tag", "roof_tag_only", "intersect_only"] = (
        "intersect_or_roof_tag"
    )
    chunk_deg: float = 0.5
    timeout_s: int = 300


# Bands available in the CDSE Sentinel-2 quarterly cloudless mosaics (10 m, int16)
CDSE_MOSAIC_BANDS = ("B02", "B03", "B04", "B08")

# Bands in the Earth Genome Sentinel-2 temporal mosaics (~19 m Web Mercator
# px ≈ 12 m ground at 52°N, uint16, public HTTPS on Source Cooperative)
EARTHGENOME_MOSAIC_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
)


class ImageryConfig(BaseModel):
    stac_source: Literal[
        "earth_search", "planetary_computer", "cdse_mosaics", "earthgenome"
    ] = "earthgenome"
    stac_url: str | None = None  # defaults per stac_source when omitted
    collection: str = "sentinel-2-l2a"
    date_ranges: list[DateRange]
    bands: list[str] = Field(
        default=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    )
    composite_method: Literal["median", "medoid", "mean"] = "median"
    cloud_mask: Literal["scl", "omnicloudmask"] = "scl"
    max_cloud_pct: float = 60.0  # skip scenes cloudier than this in the STAC search
    max_scenes: int = 15  # clearest scenes kept per tile/range (bandwidth is the bottleneck)
    target_resolution_m: float = 10.0
    tiles: list[str] | None = None  # optional MGRS whitelist

    @model_validator(mode="after")
    def _default_stac_url(self) -> ImageryConfig:
        if self.stac_url is None:
            self.stac_url = {
                "earth_search": "https://earth-search.aws.element84.com/v1",
                "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
                "cdse_mosaics": "https://stac.dataspace.copernicus.eu/v1",
                "earthgenome": "https://stac.earthgenome.org",
            }[self.stac_source]
        mosaic_sources = {
            "cdse_mosaics": ("sentinel-2-global-mosaics", CDSE_MOSAIC_BANDS),
            "earthgenome": ("sentinel2-temporal-mosaics", EARTHGENOME_MOSAIC_BANDS),
        }
        if self.stac_source in mosaic_sources:
            default_collection, available = mosaic_sources[self.stac_source]
            if "collection" not in self.model_fields_set:
                self.collection = default_collection
            unavailable = [b for b in self.bands if b not in available]
            if unavailable:
                raise ValueError(
                    f"{self.stac_source} only provides bands {list(available)}; "
                    f"remove {unavailable} from imagery.bands"
                )
            if self.cloud_mask != "scl":
                raise ValueError(
                    f"{self.stac_source} is pre-composited (cloud-free); cloud_mask "
                    "has no effect — leave it at the default 'scl'"
                )
        return self

    @field_validator("date_ranges")
    @classmethod
    def _check_ranges(cls, v: list[DateRange]) -> list[DateRange]:
        if not v:
            raise ValueError("imagery.date_ranges must contain at least one [start, end] pair")
        for start, end in v:
            if start >= end:
                raise ValueError(f"date range start must precede end: {start} >= {end}")
        return v


class BuildingsConfig(BaseModel):
    # Building footprints come from Overture Maps (bbox-queryable GeoParquet on
    # S3 — see the `overture` section for the release pin).
    building_area_min_m2: float = 1000.0


class ModelConfig(BaseModel):
    arch: Literal["unet"] = "unet"
    encoder: str = "resnet18"
    # Additional regions whose chips are merged into train/val (their pipelines
    # must have produced chips with the SAME bands). Test metrics stay on the
    # primary region only, so they remain interpretable.
    train_regions: list[str] = Field(default_factory=list)
    pretrained: str | None = "ssl4eo_s2_moco"
    num_classes: int = 2
    loss: Literal["focal_dice", "ce", "focal"] = "focal_dice"
    lr: float = 1e-4
    batch_size: int = 32
    patch_size: int = 256
    max_epochs: int = 60
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ChipsConfig(BaseModel):
    pos_per_label: int = 4
    neg_ratio: int = 5
    # hard negatives must have no OSM solar (any size) within this distance
    solar_free_buffer_m: float = 50.0


class CleanNegativesConfig(BaseModel):
    """Iterative hard-negative cleaning (positive-unlabeled mitigation).

    A "solar-free" hard negative whose chip a trained baseline model predicts as
    solar is most likely a building whose solar is simply unmapped in OSM. Such
    negatives are excluded from the re-training set instead of being taught as
    true negatives (which would push the model to suppress real detections).
    """

    prob_threshold: float = 0.5       # per-pixel solar probability counted as "hot"
    max_solar_fraction: float = 0.05  # drop a negative if more than this fraction of its pixels are hot


class PostprocessConfig(BaseModel):
    prob_threshold: float = 0.5
    building_coverage_min: float = 0.10
    min_solar_area_m2: float = 1000.0


class SplitRatios(BaseModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitRatios:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split.ratios must sum to 1.0, got {total}")
        return self


class SplitConfig(BaseModel):
    method: Literal["spatial_blocks"] = "spatial_blocks"
    block_size_km: float = 25.0
    ratios: SplitRatios = SplitRatios()
    seed: int = 42


class Config(BaseModel):
    region: str
    aoi: AOIConfig
    overture: OvertureConfig = OvertureConfig()
    osm: OSMConfig = OSMConfig()
    imagery: ImageryConfig
    buildings: BuildingsConfig = BuildingsConfig()
    model: ModelConfig = ModelConfig()
    chips: ChipsConfig = ChipsConfig()
    clean_negatives: CleanNegativesConfig = CleanNegativesConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    split: SplitConfig = SplitConfig()
    data_dir: Path = Path("data")

    @property
    def in_channels(self) -> int:
        """Model input channels, derived from the configured band list."""
        return len(self.imagery.bands)

    @property
    def region_dir(self) -> Path:
        return self.data_dir / self.region

    def run_id(self) -> str:
        """Stable short hash of fields that affect the trained model."""
        relevant = {
            "region": self.region,
            "imagery": self.imagery.model_dump(mode="json"),
            "overture": self.overture.model_dump(mode="json"),
            "osm": self.osm.model_dump(mode="json"),
            "model": self.model.model_dump(mode="json"),
            "chips": self.chips.model_dump(mode="json"),
            "split": self.split.model_dump(mode="json"),
        }
        digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()
        return digest[:10]


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
