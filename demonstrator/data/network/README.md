# GB ETYS storm-window inputs for DCCFM (PYPOWER)

Two storm windows (Arwen 22 h, Eunice 25 h) on the full GB transmission network, built with PyPSA-GB and exported
for the DC cascading failure model. Generated 2026-10-22 from the PyPSA-GB networks in `networks/`
(`Historical_<year>_ETYS_<storm>.nc`, copies of PyPSA-GB `resources/network/<scenario>_..._interconnectors.nc`),
so this folder is self-contained.

| Folder | Window (hourly, 25 snapshots) | Fleet and demand year |
|---|---|---|
| `Arwen2021/`  | 2021-11-26 12:00 to 2021-11-27 09:00 (22 snapshots) | 2021 |
| `Eunice2022/` | 2022-02-18 00:00 to 2022-02-19 00:00 | 2022 |

## Scripts (this folder)
- `run_dcopf.py` : MAIN FILE. Imports `<storm>/GB_ETYS_<storm>.py`, builds each hour's PYPOWER case
  (hourly PD and PMAX swapped in) and checks it with the cascade model's own `load2disp` + `rundcopf_mp`.
  `python run_dcopf.py Arwen2021 [--hours 0 12 24] [--include generator storage]`
  Run in the cascade-model Python 3.11 environment; expects `DCCFM - PYPOWER` next to this `data` folder.
- `export_dccfm_inputs.py` : regenerates the CSV tables from `networks/` (pypsa-gb-stable environment), then calls
  `make_case_py.py`. It does not rebuild the line classification in branch.csv: re-running it resets `kind` to "line".
- `make_case_py.py` : writes `<storm>/GB_ETYS_<storm>.py` from the CSV tables.

## Files in each folder
- `bus.csv` : 2,023 buses, `bus_id` 1..N, PyPSA name, type (3 slack, 2 PV, 1 PQ), kV, lon, lat (WGS84).
- `branch.csv` : 3,021 branches. `kind`: overhead (761), mixed (223), cable (448), zero_length (47),
  connection (95), transformer (1,445), hvdc_equiv (2). Overhead and cable lengths (`ohl_km`, `cable_km`), owner and
  data source per line come from ETYS 2022 Appendix B (B-2-1a to d) and GB_network.xlsx (one-off classification,
  not rebuilt by the export script). "connection" rows are PyPSA-GB generator connection edges (Extra_WF_edges,
  Extra_BMUs_edges), not physical circuits. `geometry_known` (overhead and mixed lines only) is False for 52 short
  circuits (median 1.45 km OHL) whose two ends share one site location in PyPSA-GB, so their route and bearing are
  unknown; they stay in the network but are excluded from wind damage. `r_pu`, `x_pu`, `b_pu` on 100 MVA (ETYS values as stored by PyPSA-GB), `rateA_MW`, `length_km` (ETYS circuit route length).
- `gen.csv` : 3,488 / 3,509 units; `kind` = generator (included), storage or interconnector_import (`included = 0`).
- `load_Pd.csv` : hour x bus_id, MW (PyPSA loads summed per bus).
- `gen_Pmax.csv` : hour x gen_id, `Pmax` in MW (= p_nom x p_max_pu(t)); Pmin is 0 for every unit.
- `GB_ETYS_<storm>.py` : self-contained case module in the style of `IEEE39_CapacityMod.py`:
  `from GB_ETYS_Arwen2021 import GB_ETYS_Arwen2021 as load_case; ppc = load_case(h)` for hour index h (0 to 24).
  Tables are np.array literals (bus, gen, branch, gencost) plus `PD_HOURLY` and `PMAX_HOURLY`; `include` selects
  unit kinds (default generators only). Extra keys: genfuel, gentype, gen_names,
  bus_names, branch_names, branch_kind, snapshot. About 1.9 MB, 3 s to import.
- `validation_dcopf.csv` : output of `run_dcopf.py`, base-case DC OPF per hour.

## Provenance
- Topology, reactances, ratings: NESO ETYS 2022 Appendix B via PyPSA-GB (the nearest vintage; 2021 appendix not held).
- Thermal fleet: DUKES; renewables, hydro, marine: REPD sites. Nuclear availability: ESPENI fleet factor.
- Wind and solar Pmax: atlite on ERA5 cutouts (Vestas V112 3 MW onshore, NREL 5 MW offshore, CSi PV).
- Demand: ESPENI national half-hourly, split to GSPs by FES regional shares.
- Marginal costs: PyPSA-GB per-carrier values, used as given (no tie-break increments: they made HiGHS dual simplex
  fail with status Unknown on some hours with scipy 1.16; HiGHS handles the degenerate optimum itself).

## Mapping choices
- 7 external HVDC buses and the Caithness-Moray DC midpoint dropped (no AC connection).
- Western Link (FLIB41-HUCS4-, 2,250 MW, 422 km) and Caithness-Moray (SPIT2K-BLHI4R, 800 MW, 160 km)
  as equivalent AC branches, x = 0.3 ohm/km on 400 kV (kind = hvdc_equiv). This reactance and the route lengths
  are our assumption (PyPSA-GB models them as links with no impedance). Flow follows DC power flow, not control.
- Interconnectors excluded (GB was a net exporter in both storms); 9 import units listed at the GB landing bus
  with included = 0. External buses, HVDC links to them and EU export loads are removed. Greenlink (commissioned
  2025) is in the PyPSA-GB list for both years and should be dropped if interconnectors are ever switched on.
- Storage (76 to 81 batteries 2.3 GW, 4 pumped hydro 2.8 GW) excluded; listed with included = 0 for a later
  fast-reserve case (cost 1000 GBP/MWh when switched on). No energy limit is modelled.
- PyPSA-GB load-shedding slack generators removed; shedding is handled by the cascade model.
- Generators with zero Pmax in an hour are kept (constant gen table across hours).
- PyPSA-GB substitutes defaults where ETYS gives zero or missing impedance: 90 lines carry the line default
  (r 0.002, x 0.02, b 0.5 p.u.), 25 offshore wind farm links the cable default (r 0.005, x 0.015, b 1.5 p.u.).
- 62 lines have x < 1e-4 p.u. (bus couplers in ETYS), minimum 1e-6 p.u.

## Validation (validation_dcopf.csv)
All 50 hours converge (about 2 s each), all load served, no shedding. Peak demand 45.6 GW (Arwen 17:00)
and 43.7 GW (Eunice 12:00). Wind output 15 to 19 GW throughout Arwen, rising from 7 to 17 GW in Eunice.
About 15 to 27 branches sit at 100 % loading in each least-cost base case (no N-1 margin), so expect
immediate overloads on any contingency near those circuits.
