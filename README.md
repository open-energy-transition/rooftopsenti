# rooftopsenti

Detect **large rooftop solar PV** installations from Sentinel-2 imagery at country/state/province
scale, and flag large buildings that have visible solar but **no solar mapping in OSM**.

## How it works

1. **AOI** — resolve a region (country/state/province) to a boundary via
   [geoBoundaries](https://www.geoboundaries.org/) and derive the Sentinel-2 MGRS tile worklist.
2. **Labels** — extract large rooftop solar polygons (`power=generator` +
   `generator:source=solar`, area ≥ 1000 m², on a building) from OpenStreetMap via the
   Overpass API. These are the training positives.
3. **Imagery** — per-MGRS-tile **cloud-free Sentinel-2 composites**, written as COGs and
   catalogued in a local STAC collection. Three `imagery.stac_source` backends:
   - `cdse_mosaics` *(default in the shipped configs)* — pre-composited
     [CDSE quarterly cloudless mosaics](https://dataspace.copernicus.eu/news/2023-11-28-quarterly-cloudless-sentinel-2-mosaics-available-data-collections-and-copernicus)
     (10 m, B02/B03/B04/B08). No per-scene downloads — ~95% less transfer; ideal on slow links.
   - `planetary_computer` / `earth_search` — full 10-band L2A scene compositing
     (SCL cloud mask + temporal median); bandwidth-heavy but gives SWIR/red-edge bands.
4. **Buildings** — fetch [GlobalBuildingAtlas LoD1](https://github.com/zhu-xlab/GlobalBuildingAtlas)
   footprints and keep only **large buildings** (≥ 1000 m²). These define the inference ROIs.
5. **Model** — U-Net semantic segmentation ([TorchGeo](https://torchgeo.org/) +
   SSL4EO Sentinel-2 pretrained encoder), trained on chips around OSM labels with hard
   negatives sampled from solar-free large buildings.
6. **Inference** — run only on composite windows containing a large building, aggregate
   predicted solar pixels per building footprint, and compare with OSM to output a
   **`missing_in_osm`** candidate list (GeoParquet/GeoJSON + HTML map).

## Quickstart

```bash
pixi install
pixi run smoke               # end-to-end on a tiny AOI (Venlo, NL)
pixi run test
```

The default imagery backend (`cdse_mosaics`) reads from `s3://eodata` and needs **free**
Copernicus Data Space credentials: [register](https://dataspace.copernicus.eu), create an
S3 key pair at <https://eodata-s3keysmanager.dataspace.copernicus.eu>, then:

```bash
export CDSE_S3_ACCESS_KEY=...
export CDSE_S3_SECRET_KEY=...
```

Full pipeline for a region:

```bash
pixi run rooftopsenti run-all -c configs/netherlands.yaml
```

Or stage by stage:

```bash
rooftopsenti aoi        -c configs/netherlands.yaml
rooftopsenti labels     -c configs/netherlands.yaml
rooftopsenti composite  -c configs/netherlands.yaml --resume
rooftopsenti gba        -c configs/netherlands.yaml
rooftopsenti chips      -c configs/netherlands.yaml
rooftopsenti train      -c configs/netherlands.yaml
rooftopsenti infer      -c configs/netherlands.yaml --resume
rooftopsenti postprocess -c configs/netherlands.yaml
rooftopsenti report     -c configs/netherlands.yaml
```

On a machine with an NVIDIA driver use the CUDA env: `pixi run -e cuda rooftopsenti train ...`

## Regions

Each region is a YAML in `configs/`. The pilot is the **Netherlands** (dense, well-mapped OSM
rooftop solar). Transfer targets like **Pakistan** and **Vietnam** run inference-only with a
model trained on well-mapped regions — detections there are candidates requiring manual review,
since OSM cannot be treated as ground truth.

## Data sources & licenses

- Sentinel-2 L2A & quarterly cloudless mosaics: Copernicus, free
- OpenStreetMap: © OSM contributors, ODbL
- GlobalBuildingAtlas (TUM): polygons ODbL; heights/LoD1 CC BY-NC 4.0
