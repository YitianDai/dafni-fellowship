"""Write <storm>/GB_ETYS_<storm>.py: a self-contained PYPOWER case module in the style of IEEE39_CapacityMod.py,
with the 25 hourly load (PD) and generator Pmax tables embedded.  Usage: python make_case_py.py [Arwen2021 Eunice2022]"""
import os, sys, numpy as np, pandas as pd

def fmt_row(vals, g=8):
    return "[" + ", ".join(f"{v:.{g}g}" if isinstance(v, float) else str(v) for v in vals) + "],"

def write_py(folder):
    tag = os.path.basename(os.path.normpath(folder)); fn = f"GB_ETYS_{tag}"
    B = pd.read_csv(os.path.join(folder, "bus.csv")); R = pd.read_csv(os.path.join(folder, "branch.csv"))
    G = pd.read_csv(os.path.join(folder, "gen.csv"))
    PD = pd.read_csv(os.path.join(folder, "load_Pd.csv"), index_col=0); PD.columns = PD.columns.astype(int)
    PM = pd.read_csv(os.path.join(folder, "gen_Pmax.csv"), index_col=0); PM.columns = PM.columns.astype(int)
    PD = PD[B.bus_id]; PM = PM[G.gen_id]
    L = []
    w = L.append
    w('"""')
    w(f"{fn} -- GB transmission network, storm window {PD.index[0]} to {PD.index[-1]} ({len(PD)} hourly snapshots)")
    w("")
    w("Built from PyPSA-GB: ETYS 2022 Appendix B topology (buses, lines, transformers, ratings), DUKES/REPD fleet,")
    w("ERA5 wind and solar availability, ESPENI demand split to GSPs.  See README_network_inputs.md.")
    w("")
    w("Differences from a stock case:")
    w(f"    - {len(B)} buses (7 external HVDC terminals and the Caithness-Moray DC midpoint dropped)")
    w(f"    - {len(R)} branches: {int((R.kind=='line').sum())} lines, {int((R.kind=='transformer').sum())} transformers,")
    w("      2 internal HVDC links as equivalent AC branches (Western Link, Caithness-Moray)")
    w(f"    - {len(G)} units: {int(G.included.sum())} generators on by default; storage and interconnectors listed, off by default")
    w("    - Pmin = 0 for every unit; Pmax = p_nom x availability(t), Pg = 0 (dispatch left to the DC OPF)")
    w("    - gencost linear: c1 = PyPSA-GB marginal cost (GBP/MWh), no tie-break; no load shedding (use load2disp)")
    w("")
    w("Usage:")
    w(f"    from {fn} import {fn} as load_case")
    w("    ppc = load_case()                         # hour 0")
    w("    ppc = load_case(h=12)                     # hour index 0..24, see ppc['snapshot']")
    w("    ppc = load_case(h=12, include=('generator', 'storage'))   # storage on as fast reserve")
    w('"""')
    w("")
    w("import numpy as np")
    w("")
    w(f"SNAPSHOTS = {[str(s) for s in PD.index]!r}")
    w("")
    w("STORAGE_COST = 1000.0  # GBP/MWh applied to storage when included (well above any generator); no energy limit")
    w("")
    w(f"def {fn}(h=0, include=('generator',)):")
    w('    """Return the PYPOWER case for hour index h of the storm window (MATPOWER-compatible dict)."""')
    w("    ppc = {'version': '2', 'baseMVA': 100.0, 'snapshot': SNAPSHOTS[h]}")
    w("")
    w("    ppc['bus'] = np.array([")
    w("        # bus  type  Pd(hour 0)  Qd  Gs  Bs  area  Vm   Va  baseKV  zone  Vmax  Vmin")
    for b, t, kv, pd0 in zip(B.bus_id, B.bus_type, B.baseKV, PD.iloc[0].values):
        w(f"        [{b}, {t}, {pd0:.3f}, 0.0, 0, 0, 1, 1.0, 0.0, {kv:g}, 1, 1.05, 0.95],")
    w("    ], dtype=float)")
    w("")
    w("    ppc['gen'] = np.array([")
    w("        # bus  Pg  Qg  Qmax  Qmin  Vg  mBase  status  Pmax(hour 0)  Pmin  Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf")
    for b, pmx in zip(G.bus_id, PM.iloc[0].values):
        w(f"        [{b}, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1, {pmx:.3f}, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],")
    w("    ], dtype=float)")
    w("")
    w("    ppc['branch'] = np.array([")
    w("        # fbus  tbus  r  x  b  rateA  rateB  rateC  ratio  angle  status  angmin  angmax")
    for r in R.itertuples():
        w(f"        [{r.from_bus}, {r.to_bus}, {r.r_pu:.10g}, {r.x_pu:.10g}, {r.b_pu:.10g}, {r.rateA_MW:g}, {r.rateA_MW:g}, {r.rateA_MW:g}, {r.ratio:g}, {r.angle_deg:g}, 1, -360, 360],")
    w("    ], dtype=float)")
    w("")
    w("    ppc['gencost'] = np.array([")
    w("        # model startup shutdown n  c2  c1  c0")
    for c1 in G.c1:
        w(f"        [2, 0, 0, 3, 0.0, {c1:.6f}, 0.0],")
    w("    ], dtype=float)")
    w("")
    w("    # ---- unit metadata (parallel to ppc['gen']) ----")
    w(f"    ppc['genfuel'] = {G.carrier.tolist()!r}")
    w(f"    ppc['gentype'] = {G.kind.tolist()!r}")
    w(f"    ppc['gen_names'] = {G.name.tolist()!r}")
    w(f"    ppc['bus_names'] = {B.pypsa_name.tolist()!r}")
    w(f"    ppc['branch_names'] = {R.pypsa_name.tolist()!r}")
    w(f"    ppc['branch_kind'] = {R.kind.tolist()!r}")
    w("")
    w("    # ---- hourly tables: rows = SNAPSHOTS, columns = ppc['bus'] / ppc['gen'] rows ----")
    w("    PD_HOURLY = np.array([")
    for i, row in enumerate(PD.values):
        w("        " + fmt_row([float(v) for v in np.round(row, 3)], 7) + f"  # {PD.index[i]}")
    w("    ], dtype=float)")
    w("    PMAX_HOURLY = np.array([")
    for i, row in enumerate(PM.values):
        w("        " + fmt_row([float(v) for v in np.round(row, 3)], 7) + f"  # {PM.index[i]}")
    w("    ], dtype=float)")
    w("")
    w("    # ---- assemble hour h ----")
    w("    ppc['bus'][:, 2] = PD_HOURLY[h]")
    w("    ppc['gen'][:, 8] = PMAX_HOURLY[h]")
    w("    keep = np.isin(np.array(ppc['gentype']), include)")
    w("    ppc['gen'] = ppc['gen'][keep]; ppc['gencost'] = ppc['gencost'][keep]")
    w("    for k in ('genfuel', 'gentype', 'gen_names'):")
    w("        ppc[k] = [v for v, m in zip(ppc[k], keep) if m]")
    w("    st = np.array(ppc['gentype']) == 'storage'")
    w("    ppc['gencost'][st, 5] = STORAGE_COST")
    w("    return ppc")
    w("")
    p = os.path.join(folder, fn + ".py"); open(p, "w").write("\n".join(L))
    print(f"{tag}: wrote {fn}.py ({os.path.getsize(p)/1e6:.1f} MB)")
    return p

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for tag in sys.argv[1:] or ["Arwen2021", "Eunice2022"]:
        write_py(os.path.join(here, tag))
