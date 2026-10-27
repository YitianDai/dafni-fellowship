# DAFNI packaging

How the repository maps onto the DAFNI platform.

| DAFNI object | Contents | Source in this repo |
|---|---|---|
| Dataset `GB storm network cases` | `Arwen2021/`, `Eunice2022/` (bus, branch, gen, load_Pd, gen_Pmax, `GB_ETYS_<storm>.py`) | `data/network/` |
| Dataset `ERA5 storm fields` | `era5_arwen_2021.nc`, `era5_eunice_2022.nc`, `gb_land_mask_49x65.npy` | `data/era5/` |
| Model `gb-storm-wind-damage` | Monte Carlo wind damage | `dafni/wind_damage/` |
| Model `gb-storm-cascade-dccfm` | Hourly DC cascading failure | `dafni/storm_cascade/` |
| Model `gb-storm-cascade-figures` | Map, frames, GIF, viewer, stages | `dafni/figures/` |
| Workflow `Storm single case` | wind_damage -> storm_cascade (SAMPLE) -> figures | built on DAFNI |
| Workflow `Storm ensemble` | wind_damage -> storm_cascade (SAMPLES_FIRST..LAST) -> figures | built on DAFNI |
| Visualisation | `figs/<storm>_cascade_viewer.html`, GIF and PNGs from the figures model outputs | DAFNI visualisation on the workflow outputs |

## Conventions used by the containers

All three images share one entry point, `models/dafni_entry.py <model>`:

* parameters arrive as environment variables (`STORM`, `N_SAMPLES`, `SEED`, `SAMPLE`, `SAMPLES_FIRST`, `SAMPLES_LAST`, `FIGURE_SAMPLE`);
* dataslots are mounted under `/data/inputs/network`, `/data/inputs/era5` and `/data/inputs/results` (the latter being the upstream model's outputs);
* everything written to `/data/outputs/<storm>/` is collected by DAFNI.

The entry point assembles the repository layout under `/data/work` (symlinks to the slots, `results` pointing at `/data/outputs`) and calls the model script with `--root /data/work`, so the scripts are unchanged between a local run and a DAFNI run.

## Local test of the DAFNI entry points (no Docker)

```
mkdir -p test/inputs && cp -r data/network data/era5 test/inputs/
DAFNI_DATA=test STORM=Eunice2022 N_SAMPLES=50 SEED=42 python models/dafni_entry.py wind_damage
mkdir -p test/inputs/results && cp -r test/outputs/Eunice2022 test/inputs/results/
DAFNI_DATA=test STORM=Eunice2022 SAMPLE=3 python models/dafni_entry.py storm_cascade
DAFNI_DATA=test STORM=Eunice2022 FIGURE_SAMPLE=3 python models/dafni_entry.py figures
```
(on PowerShell set the variables with `$env:DAFNI_DATA="test"` and so on before each command.)

## Building the images without Docker

`.github/workflows/dafni-images.yml` (repository root) builds the three images on GitHub Actions on every push to `main` touching `demonstrator/` (or by hand through Actions, Run workflow). Each run attaches `wind_damage-image.tar.gz`, `storm_cascade-image.tar.gz` and `figures-image.tar.gz`; download them from the run page.

## End-to-end container test

The same Actions workflow then runs the three images as containers in the DAFNI way (parameters as environment variables, inputs mounted at `/data/inputs`, outputs collected from `/data/outputs`) on Storm Eunice, wind damage seed 42 and cascade sample 297, and `dafni/test/compare_outputs.py` checks that the container outputs reproduce the reference results committed in `results/Eunice2022/` (same sampled failures, same hourly cascade table, figures present). A green "End-to-end container test" job means the images behave exactly as the local run; the container outputs are attached to the run as `container-test-outputs`.

## Uploading to DAFNI

1. Data: upload `data/network` as one dataset and `data/era5` as another (title, description and licence filled in on the platform).
2. Models: for each model upload the image `.tar.gz` together with `dafni/<model>/model-definition.yaml`.
3. Workflows: chain the models, binding each model's `inputs/network`, `inputs/era5` slots to the datasets and `inputs/results` to the previous step's output; set `STORM` the same on every step.
4. Visualisation: create a visualisation on the workflow outputs and open `figs/<storm>_cascade_viewer.html`.
