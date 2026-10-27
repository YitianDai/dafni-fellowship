r"""
Export PyPSA-GB ETYS storm windows to PYPOWER / DCCFM inputs.

For each storm it reads the final PyPSA-GB network, copied to
  networks/<scenario>.nc   (= PyPSA-GB resources/network/<scenario>_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc)
restricts to the 25-hour window and writes into <out>/<storm>/:
  bus.csv, branch.csv, gen.csv, load_Pd.csv, gen_Pmax.csv, and GB_ETYS_<storm>.py (via make_case_py.py)

No OPF is run: the cascade model solves its own base case from PD, PMIN, PMAX and cost.
Mapping choices are documented in README_network_inputs.md.
NOTE: the delivered branch.csv also carries the overhead / mixed / cable classification (kind, ohl_km, cable_km, owner,
source) taken from ETYS 2022 Appendix B and GB_network.xlsx; this script does not rebuild it. Re-running it overwrites
branch.csv with kind = "line" for all lines.

Usage (Windows, in the pypsa-gb-stable environment):
  python export_dccfm_inputs.py
  python export_dccfm_inputs.py --networks networks --out .
"""
import os, sys, argparse, warnings, logging
import numpy as np, pandas as pd, pypsa, networkx as nx
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

BASE_MVA   = 100.0
OHM_PER_KM = 0.3                    # equivalent reactance for internal HVDC, 400 kV base
HVDC = [  # name, bus0, bus1, rating MW, route length km
    ("Western Link",    "FLIB41", "HUCS4-", 2250.0, 422.0),
    ("Caithness-Moray", "SPIT2K", "BLHI4R",  800.0, 160.0),
]
STORMS = {
    "Arwen2021":  ("Historical_2021_ETYS_Arwen",  "2021-11-26 12:00", "2021-11-27 09:00"),
    "Eunice2022": ("Historical_2022_ETYS_Eunice", "2022-02-18 00:00", "2022-02-19 00:00"),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def export(tag, scenario, start, end, networks, out):
    nc = os.path.join(networks, f"{scenario}.nc")
    n = pypsa.Network(nc)
    snaps = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]

    # ---- buses: drop external HVDC terminals and the Caithness-Moray DC midpoint (no AC connection)
    ext = [b for b in n.buses.index if b.startswith("HVDC_External") or b == "Central DC Substation"]
    buses = n.buses.drop(ext)

    # ---- branches: lines, transformers, HVDC equivalents.
    # PyPSA-GB stores ETYS "R, X, B (% on 100 MVA)" / 100 in PyPSA's r, x, b fields, i.e. already p.u. on 100 MVA
    # (process_ETYS_data.py). PyPSA's derived r_pu/x_pu/b_pu assume ohms (lines) or p.u. on s_nom (transformers)
    # and are therefore NOT used here.
    br = [dict(name=i, kind="line", bus0=r.bus0, bus1=r.bus1, r_pu=r.r, x_pu=r.x, b_pu=r.b,
               rateA=r.s_nom * r.s_max_pu, ratio=0.0, angle=0.0, kV=r.v_nom, length_km=r.length_km)  # ETYS circuit route length
          for i, r in n.lines.iterrows()]
    br += [dict(name=i, kind="transformer", bus0=r.bus0, bus1=r.bus1, r_pu=r.r, x_pu=r.x, b_pu=r.b,
                rateA=r.s_nom * r.s_max_pu, ratio=r.tap_ratio, angle=r.phase_shift, kV=n.buses.v_nom[r.bus0], length_km=0.0)
           for i, r in n.transformers.iterrows()]
    for nm, b0, b1, p, L in HVDC:
        br.append(dict(name=f"HVDC_{nm} ({b0}-{b1})", kind="hvdc_equiv", bus0=b0, bus1=b1, r_pu=0.0, b_pu=0.0,
                       x_pu=OHM_PER_KM * L / (400 ** 2 / BASE_MVA), rateA=p, ratio=0.0, angle=0.0, kV=400.0, length_km=L))
    br = pd.DataFrame(br)
    G = nx.Graph(); G.add_nodes_from(buses.index); G.add_edges_from(zip(br.bus0, br.bus1))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    assert len(comps) == 1, f"{tag}: network not connected, components {[len(c) for c in comps]}"
    bus_id = pd.Series(np.arange(1, len(buses) + 1), index=buses.index)

    # ---- loads per bus (GB buses only)
    L = n.loads_t.p_set.reindex(columns=n.loads.index).fillna(0).loc[snaps]
    PD = pd.DataFrame(0.0, index=snaps, columns=buses.index)
    for ld, bus in n.loads.bus.items():
        if bus in PD: PD[bus] += L[ld]

    # ---- generators (load-shedding slacks removed); storage and interconnectors listed with included = 0
    g = n.generators[n.generators.carrier != "load_shedding"]
    pmax_t = n.get_switchable_as_dense("Generator", "p_max_pu").loc[snaps, g.index] * g.p_nom
    rows, Pmax = [], {}
    for i, r in g.iterrows():
        if r.carrier == "EU_import":      # foreign-side unit: list at the GB landing bus, off by default
            lk = n.links[(n.links.bus1 == r.bus) | (n.links.bus0 == r.bus)]
            gb = [x for x in pd.concat([lk.bus0, lk.bus1]) if x in buses.index][0]
            rows.append(dict(name=i, bus=gb, carrier="interconnector", kind="interconnector_import",
                             p_nom=r.p_nom, marginal_cost=0.0, included=0))
            Pmax[i] = pd.Series(r.p_nom, index=snaps)
        else:
            rows.append(dict(name=i, bus=r.bus, carrier=r.carrier, kind="generator",
                             p_nom=r.p_nom, marginal_cost=r.marginal_cost, included=1))
            Pmax[i] = pmax_t[i]
    for i, r in n.storage_units.iterrows():
        rows.append(dict(name=i, bus=r.bus, carrier=r.carrier, kind="storage", p_nom=r.p_nom, marginal_cost=0.0, included=0))
        Pmax[i] = pd.Series(r.p_nom, index=snaps)
    gens = pd.DataFrame(rows)
    gens["gen_id"] = np.arange(1, len(gens) + 1)
    gens["bus_id"] = bus_id[gens.bus].values
    gens["c1"] = gens.marginal_cost                                # linear cost (PyPSA-GB marginal cost, no tie-break)
    Pmax = pd.DataFrame(Pmax)[gens.name]; Pmax.columns = gens.gen_id.values

    slack = gens[gens.carrier.isin(["CCGT", "nuclear"])].groupby("bus").p_nom.sum().idxmax()
    btype = pd.Series(1, index=buses.index); btype[gens[gens.included == 1].bus.unique()] = 2; btype[slack] = 3

    # ---- write
    d = os.path.join(out, tag); os.makedirs(d, exist_ok=True)
    pd.DataFrame({"bus_id": bus_id.values, "pypsa_name": buses.index, "bus_type": btype.values,
                  "baseKV": buses.v_nom.values, "lon": buses.lon.values, "lat": buses.lat.values}).to_csv(os.path.join(d, "bus.csv"), index=False)
    pd.DataFrame({"branch_idx": br.index, "pypsa_name": br.name, "kind": br.kind,
                  "from_bus": bus_id[br.bus0].values, "to_bus": bus_id[br.bus1].values, "from_name": br.bus0, "to_name": br.bus1,
                  "r_pu": br.r_pu, "x_pu": br.x_pu, "b_pu": br.b_pu, "rateA_MW": br.rateA, "ratio": br.ratio, "angle_deg": br.angle,
                  "kV": br.kV, "length_km": br.length_km}).to_csv(os.path.join(d, "branch.csv"), index=False)
    gens[["gen_id", "name", "bus_id", "bus", "carrier", "kind", "included", "p_nom", "marginal_cost", "c1"]] \
        .rename(columns={"bus": "bus_name"}).to_csv(os.path.join(d, "gen.csv"), index=False)
    PDo = PD.copy(); PDo.columns = bus_id[PD.columns].values; PDo.index.name = "snapshot"
    PDo.round(3).to_csv(os.path.join(d, "load_Pd.csv"))
    Pmax.index.name = "snapshot"; Pmax.round(3).to_csv(os.path.join(d, "gen_Pmax.csv"))
    sys.path.insert(0, HERE); from make_case_py import write_py; write_py(d)
    print(f"{tag}: buses {len(buses)}, branches {len(br)}, gens {len(gens)} (included {gens.included.sum()}), "
          f"hours {len(snaps)}, PD {PD.sum(axis=1).min()/1e3:.1f} to {PD.sum(axis=1).max()/1e3:.1f} GW")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", default=os.path.join(HERE, "networks"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--storms", nargs="*", default=list(STORMS))
    a = ap.parse_args()
    for tag in a.storms:
        export(tag, *STORMS[tag], a.networks, a.out)
