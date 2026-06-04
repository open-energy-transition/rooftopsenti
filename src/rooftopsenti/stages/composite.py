"""Stage c) Cloud-free Sentinel-2 composites per MGRS tile via STAC.

Primary backend (``cloud_mask: scl``): query Element84 Earth Search with
pystac-client, lazily load with odc-stac, mask clouds with the L2A Scene
Classification Layer, and reduce with a temporal median. Runs anywhere, no GPU.

Optional backend (``cloud_mask: omnicloudmask``): delegate to S2Mosaic, which
uses the OmniCloudMask deep-learning mask (better, GPU-friendly) but queries
Microsoft Planetary Computer instead of Earth Search.

Every composite is written as a COG and registered in a local STAC catalog.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import odc.stac
import pystac_client
import xarray as xr
from loguru import logger

from ..config import Config
from ..io_artifacts import ArtifactStore
from ..stac_catalog import register_composite
from . import aoi

# Earth Search v1 asset names for Sentinel-2 L2A bands
EARTH_SEARCH_ASSETS = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}

# SCL classes to KEEP: 4 vegetation, 5 not-vegetated, 6 water, 7 unclassified.
# Masked: nodata/saturated/dark (0-2), cloud shadow (3), clouds (8-10), snow (11).
SCL_VALID = (4, 5, 6, 7)

NODATA = 0


def _search_items(tile: str, date_range: tuple[str, str], cfg: Config):
    catalog = pystac_client.Client.open(cfg.imagery.stac_url)
    search = catalog.search(
        collections=[cfg.imagery.collection],
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"grid:code": {"eq": f"MGRS-{tile}"}},
    )
    items = list(search.items())
    logger.info("{}: {} STAC items in {}..{}", tile, len(items), *date_range)
    return items


def _scl_median_composite(
    items, bands: list[str], aoi_geom, cfg: Config
) -> xr.DataArray | None:
    """Lazy-load items with odc-stac, SCL-mask, temporal median -> (band, y, x)."""
    assets = [EARTH_SEARCH_ASSETS[b] for b in bands] + [EARTH_SEARCH_ASSETS["SCL"]]
    ds = odc.stac.load(
        items,
        bands=assets,
        geopolygon=aoi_geom,
        resolution=cfg.imagery.target_resolution_m,
        groupby="solar_day",
        chunks={"x": 1024, "y": 1024},
        dtype="uint16",
        resampling="bilinear",
    )
    scl = ds[EARTH_SEARCH_ASSETS["SCL"]]
    valid = xr.zeros_like(scl, dtype=bool)
    for cls in SCL_VALID:
        valid = valid | (scl == cls)

    # reduce per band (keeps peak memory per dask task at one band's time stack)
    reduced = []
    for b in bands:
        da = ds[EARTH_SEARCH_ASSETS[b]]
        da = da.where(valid & (da > 0))  # NaN out clouds and nodata
        if cfg.imagery.composite_method == "mean":
            reduced.append(da.mean(dim="time", skipna=True))
        else:  # median (default); medoid not supported on this backend
            reduced.append(da.median(dim="time", skipna=True))
    comp = xr.concat(reduced, dim="band")
    comp = comp.fillna(NODATA).round().astype("uint16")
    comp = comp.assign_coords(band=bands)
    comp.rio.write_crs(str(ds.odc.crs), inplace=True)
    comp.rio.write_nodata(NODATA, inplace=True)
    return comp


def _write_cog(comp: xr.DataArray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.rio.to_raster(
        out_path,
        driver="COG",
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
    )


def _composite_s2mosaic(
    tile: str, date_range: tuple[str, str], cfg: Config, out_path: Path
) -> None:
    """OmniCloudMask backend via S2Mosaic (queries Planetary Computer)."""
    from datetime import date

    from s2mosaic import mosaic

    start = date.fromisoformat(date_range[0])
    end = date.fromisoformat(date_range[1])
    duration_days = (end - start).days
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = mosaic(
        grid_id=tile,
        start_year=start.year,
        start_month=start.month,
        start_day=start.day,
        duration_days=duration_days,
        output_dir=out_path.parent,
        mosaic_method="median" if cfg.imagery.composite_method == "median" else "mean",
        required_bands=cfg.imagery.bands,
        overwrite=True,
    )
    Path(result).rename(out_path)


def run(cfg: Config, store: ArtifactStore, only_tiles: list[str] | None = None) -> None:
    import os

    import rioxarray  # noqa: F401  (registers .rio accessor)

    # Allow parallel tile workers to share cores/RAM (see README scaling notes)
    n_threads = os.environ.get("ROOFTOPSENTI_DASK_THREADS")
    if n_threads:
        import dask

        dask.config.set(scheduler="threads", num_workers=int(n_threads))

    cfg_slice = cfg.imagery.model_dump(mode="json")
    boundary = aoi.load_boundary(store)
    tiles = aoi.load_tiles(store)
    if only_tiles:
        tiles = [t for t in tiles if t in set(only_tiles)]

    for tile in tiles:
        for range_idx, date_range in enumerate(cfg.imagery.date_ranges):
            out_path = store.composite_tif(tile, range_idx)
            if store.is_fresh(out_path, cfg_slice):
                logger.info("{} range {}: fresh — skipping", tile, range_idx)
                continue

            if cfg.imagery.cloud_mask == "omnicloudmask":
                _composite_s2mosaic(tile, tuple(date_range), cfg, out_path)
            else:
                items = _search_items(tile, tuple(date_range), cfg)
                if not items:
                    logger.warning("{} range {}: no items found — skipping", tile, range_idx)
                    continue
                comp = _scl_median_composite(items, cfg.imagery.bands, boundary, cfg)
                logger.info("{} range {}: computing median composite ...", tile, range_idx)
                comp = comp.compute()
                if int((np.asarray(comp.values) != NODATA).sum()) == 0:
                    logger.warning("{} range {}: composite empty — skipping", tile, range_idx)
                    continue
                _write_cog(comp, out_path)

            register_composite(
                store.stac_catalog,
                cfg.region,
                tile,
                range_idx,
                out_path,
                tuple(date_range),
                cfg.imagery.bands,
            )
            store.write_meta(out_path, cfg_slice)
            logger.info("{} range {}: composite written -> {}", tile, range_idx, out_path)
