# Windstorm resilience of the GB transmission network: DAFNI demonstrator

Hour-by-hour simulation of windstorm damage and cascading failure on the Great Britain transmission network,
built for the DAFNI Fellowship workshop. Two storms are set up: Storm Arwen (26 Nov 2021 12:00 to 27 Nov 09:00 UTC)
and Storm Eunice (18 Feb 2022 00:00 to 19 Feb 00:00 UTC).

The chain is: ERA5 hourly gusts -> line exposure and fragility (Monte Carlo line failures per hour) ->
DC cascading failure model on the hourly network state -> energy not served, maps and animation.

## Layout

```
data/network/<storm>/    GB ETYS network case: bus.csv, branch.csv, gen.csv, load_Pd.csv, gen_Pmax.csv,
                         GB_ETYS_<storm>.py (PYPOWER case module, load_case(h) for hour h); README.md gives provenance
data/era5/               ERA5 hourly gust, 10 m wind and MSLP for each storm; GB land mask on the ERA5 grid
models/dccfm/            DC cascading failure model (cascade_simulator.py, rundcopf_mp.py, dcopf_solver_mp.py)
models/wind_damage/      hourly line failure sampling from ERA5 gusts and the fragility curve
models/storm_cascade/    hour-by-hour cascade driver (carried network state, DCCFM each damaged hour)
models/dafni_entry.py    shared entry point used inside the DAFNI containers
models/figures/          plot_cascade.py (composite frames, GIF, viewer, stage diagram, ensemble maps); plot_network.py (network map)
workflows/run_case.py    runs the chain locally (single sample or ensemble)
dafni/                   Dockerfiles, DAFNI model definitions and packaging notes (see dafni/README.md)
tools/pypsa_gb_export/   builds data/network/ from PyPSA-GB (needs the PyPSA-GB environment; not run on DAFNI)
tools/hazard/            ERA5 download and the hazard-only windstorm animation
results/<storm>/         wind_damage.csv, cascade_sample<k>.csv (+ _lostload.csv), figs/:
                           <storm>_era5_viewer.html / _era5.gif   hazard only (ERA5 gust animation)
                           <storm>_network.png                    network only (as modelled)
                           <storm>_cascade_viewer.html / _cascade.gif   hour-by-hour damage, cascade and demand loss
                           <storm>_cascade_stages.png             stage diagram of the largest cascade hour
```

## Run

```
pip install -r requirements.txt            # Python 3.11
python workflows/run_case.py Arwen2021 --sample 330          # replay one wind-damage sample, draw figures
python workflows/run_case.py Arwen2021 --samples 0 99 --figure-sample 330   # ensemble of 100 samples
```
Or step by step: `models/wind_damage/wind_damage.py Arwen2021`, `models/storm_cascade/storm_cascade.py Arwen2021 --sample 330`,
`models/figures/plot_cascade.py Arwen2021 --sample 330`. Each script resolves paths from the repository root
(override with `--root`). A storm replay takes about half a minute; one cascade hour is a few DC OPF solves.

## Method in brief

* Network: NESO ETYS 2022 Appendix B topology, ratings and impedances via PyPSA-GB; DUKES and REPD generation fleet;
  ESPENI demand split to GSPs; ERA5-driven wind and solar availability. PyPSA-GB scenarios `Historical_2021_ETYS_Arwen`
  and `Historical_2022_ETYS_Eunice` (config in `tools/pypsa_gb_export/`). Interconnectors and storage are listed but
  switched off; the two internal HVDC links are equivalent AC branches. Details in `data/network/README.md`.
* Hazard: ERA5 hourly 10 m gust (`i10fg`), wind direction and MSLP on a 0.25 degree grid, used without bias correction.
* Line exposure: each of the 932 located overhead circuits is a straight route between its substations, cut at ERA5
  cell edges; per cell and hour the effective gust is `v * max(|sin(attack angle)|, 0.5)` (IEC 60826 load form),
  and the line takes the maximum over its cells.
* Fragility: hourly failure probability `Phi((ln v_eff - 3.75) / 0.14)` (lognormal, median 42.5 m/s; Dai and Preece,
  Sustainable Energy, Grids and Networks, 2025), applied per line without length scaling. 1000 Monte Carlo samples; a failed
  line stays out for the rest of the window.
* Cascade: hours with new wind failures are simulated; hours without carry the previous state unchanged. Each simulated
  hour takes that hour's demand and generator availability, applies the carried state (lost lines, blacked-out buses,
  tripped generators, disconnected demand as a per-bus fraction), solves a pre-fault DC OPF (HiGHS) and runs the DC
  cascading failure model (per-island redispatch with 0.99 headroom, overload tripping at 101 % of rating, OFGT,
  load shedding at 5000 GBP/MWh). Losses are permanent within the window. Generator linear costs carry a tie-break
  of 1e-6 GBP/MWh per generator index so the dispatch LP has a unique optimum and results are reproducible across
  machines (PyPSA-GB gives only 11 distinct marginal costs across ~3400 units).

## Data sources

ERA5 (Copernicus Climate Data Store, Hersbach et al. 2020); NESO Electricity Ten Year Statement 2022 Appendix B;
PyPSA-GB (Lyden et al. 2024, Energy Strategy Reviews); DUKES; REPD; ESPENI (Wilson and Staffell). Coastline from the
`global-land-mask` package.

## Author

Yitian Dai, University of Manchester. DAFNI Fellowship 2026 to 2027.
