"""
Hour-by-hour storm cascade: hourly network case + one sample of hourly wind damage -> DCCFM cascade.

For a storm (Arwen2021 / Eunice2022) and one wind-damage sample from <storm>/wind_damage.csv:
  hours before the first wind failure, and later hours with no new wind failure, are not simulated:
    the network state at the end of the previous hour carries on unchanged
  in an hour with new wind failures:
    1. load_case(h): that hour's demand and generator Pmax
    2. apply the carried state: lost branches out, dead buses isolated (type 4), OFGT-tripped generators off,
       disconnected demand removed as a per-bus fraction of the hour's demand
    3. load2disp + fresh pre-fault DC OPF (rundcopf_mp), one reference bus per island
       (pre-fault shedding is not expected; if it occurs it is flagged and counted for that hour only)
    4. CascadeSimulator.run(faulty_lines = this hour's new wind failures), headroom 0.99 (class default)
    5. network_after becomes the carried state for the following hours
  energy not served each hour = disconnected fraction x that hour's bus demand (+ any flagged pre-fault shed)

Output: results/<storm>/cascade_sample<k>.csv, one row per hour (simulated or carried); cascade_iter_branches lists the
        branches tripped per cascade iteration; results/<storm>/cascade_sample<k>_lostload.csv gives demand lost per bus and hour (MW).
        cascade_iter_branches lists the
        branches tripped in each cascade iteration (iterations separated by |), dead_bus_ids the buses lost that hour.
Usage:  python storm_cascade.py Arwen2021                 random sample, chosen sample printed
        python storm_cascade.py Arwen2021 --sample 123    a given sample
        python storm_cascade.py Arwen2021 --sample 5 --rating-factor 1.2   all branch ratings +20 %
Inputs: data/network/<storm>/ and results/<storm>/wind_damage.csv under the repo root (override with --root REPO).
"""
import os, sys, copy, time, argparse, importlib
import numpy as np, pandas as pd, networkx as nx

ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (models/storm_cascade/..)
if "--root" in sys.argv: ROOT = sys.argv[sys.argv.index("--root") + 1]                    # data root (data/, results/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dccfm"))   # DCCFM code sits beside this script
if not hasattr(np, "in1d"): np.in1d = np.isin
from cascade_simulator import CascadeSimulator, load2disp
from rundcopf_mp import rundcopf_mp
from pypower.api import ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD
from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, GEN_STATUS
from pypower.idx_brch import F_BUS, T_BUS, BR_STATUS

STORMS = ["Arwen2021", "Eunice2022"]
PPOPT = ppoption(VERBOSE=0, OUT_ALL=0, PF_DC=1)
TIEBREAK = 1e-6   # GBP/MWh per generator index added to the linear cost so the dispatch LP has a unique optimum
                  # (only 11 distinct marginal costs across ~3400 units otherwise; which optimum HiGHS returns then
                  # depends on platform rounding, and at a line near its limit that changes the cascade)


def get_case(storm):
    sys.path.insert(0, os.path.join(ROOT, "data", "network", storm))
    mod = importlib.import_module(f"GB_ETYS_{storm}")
    return getattr(mod, f"GB_ETYS_{storm}"), mod.SNAPSHOTS


def add_reference_buses(ppc):
    """One reference bus per island of active buses (islands without one get their first bus as slack)."""
    bus, br = ppc["bus"], ppc["branch"]
    active = bus[:, BUS_TYPE] != 4
    row = {int(b): i for i, b in enumerate(bus[:, BUS_I])}
    G = nx.Graph(); G.add_nodes_from(np.where(active)[0])
    for f, t, s in br[:, [F_BUS, T_BUS, BR_STATUS]]:
        i, j = row[int(f)], row[int(t)]
        if s == 1 and active[i] and active[j]: G.add_edge(i, j)
    n_isl = 0
    for comp in nx.connected_components(G):
        comp = sorted(comp); n_isl += 1
        if not np.any(bus[comp, BUS_TYPE] == 3):
            bus[comp[0], BUS_TYPE] = 3
    return n_isl


def apply_state(ppc, st):
    """Carried state onto an hourly case (in place)."""
    ppc["branch"][list(st["branches_out"]), BR_STATUS] = 0
    ids = ppc["bus"][:, BUS_I].astype(int)
    dead = np.isin(ids, list(st["dead_buses"]))
    ppc["bus"][dead, BUS_TYPE] = 4; ppc["bus"][dead, PD] = 0; ppc["bus"][dead, QD] = 0
    frac = np.array([st["load_lost_frac"].get(int(b), 0.0) for b in ids])
    ppc["bus"][:, PD] *= (1 - frac)
    off = np.zeros(len(ppc["gen"]), bool); off[list(st["gens_off"])] = True
    off |= np.isin(ppc["gen"][:, GEN_BUS].astype(int), list(st["dead_buses"]))
    ppc["gen"][off, PMAX] = 0; ppc["gen"][off, PMIN] = 0; ppc["gen"][off, GEN_STATUS] = 0
    return ppc


def update_state(st, base, after, ng, branch_wind, h):
    """Read network_after (rows aligned with base: ng real gens, then load2disp rows) into the carried state."""
    out = set(np.where(after["branch"][:, BR_STATUS] == 0)[0].tolist())
    new_out = out - st["branches_out"]
    casc = new_out - set(branch_wind)
    st["branches_out"] |= out
    dead = set(after["bus"][after["bus"][:, BUS_TYPE] == 4, BUS_I].astype(int).tolist())
    new_dead = dead - st["dead_buses"]; st["dead_buses"] |= dead
    g = after["gen"][:ng]
    off = set(np.where((g[:, GEN_STATUS] == 0) | (g[:, GEN_BUS] == 0))[0].tolist())
    new_off = off - st["gens_off"]; st["gens_off"] |= off
    lb = base["gen"][ng:]; la = after["gen"][ng:]
    for k in range(len(lb)):                       # disconnected demand as a fraction of the bus demand
        bus_id = int(lb[k, GEN_BUS]); pd0 = -lb[k, PMIN]
        if pd0 <= 0: continue
        served = max(-la[k, PG], 0.0) if la[k, GEN_BUS] != 0 else 0.0
        f_new = min(max(1 - served / pd0, 0.0), 1.0)
        f_old = st["load_lost_frac"].get(bus_id, 0.0)
        st["load_lost_frac"][bus_id] = 1 - (1 - f_old) * (1 - f_new)
    for b in st["dead_buses"]:
        st["load_lost_frac"][b] = 1.0
    st["cascade_log"] += [(h, int(i)) for i in sorted(casc)]
    return sorted(casc), sorted(new_dead), sorted(new_off)


def run(storm, sample, verbose=False, rating_factor=1.0, tiebreak=True):
    load_case, snaps = get_case(storm)
    out_dir = os.path.join(ROOT, "results", storm); os.makedirs(out_dir, exist_ok=True)
    wd = pd.read_csv(os.path.join(out_dir, "wind_damage.csv"))
    n_samples = int(wd["sample"].max()) + 1
    if sample is None:
        seed = int.from_bytes(os.urandom(4), "little"); sample = int(np.random.default_rng(seed).integers(n_samples))
        print(f"random sample {sample} of {n_samples} (seed {seed}); re-run with --sample {sample}")
    W = wd[wd["sample"] == sample]
    wind_by_h = W.groupby("h").branch_idx.apply(list).to_dict()
    print(f"{storm}, sample {sample}: {len(W)} wind failures in hours {sorted(wind_by_h)}"
          + (f"; branch ratings x {rating_factor}" if rating_factor != 1.0 else ""))

    st = dict(branches_out=set(), dead_buses=set(), gens_off=set(), load_lost_frac={}, cascade_log=[])
    rows = []; lost_rows = []; t_all = time.time()
    for h, snap in enumerate(snaps):
        full = load_case(h)                                   # that hour's demand and Pmax, intact network
        full["branch"][:, 5:8] *= rating_factor                # RATE_A/B/C scaling (test option; data files unchanged)
        if tiebreak:                                          # unique LP optimum: +1e-6 GBP/MWh per generator index
            full["gencost"][:, 5] += TIEBREAK * np.arange(1, full["gencost"].shape[0] + 1)
        pd_full = full["bus"][:, PD].copy(); pd_tot = pd_full.sum()
        ids = full["bus"][:, BUS_I].astype(int)
        wind_new = [b for b in wind_by_h.get(h, []) if b not in st["branches_out"]]
        row = dict(h=h, snapshot=snap, simulated=False, wind_failures=len(wind_new), wind_branches=";".join(map(str, wind_new)),
                   cascade_trips=0, cascade_branches="", cascade_iter_branches="", new_dead_buses=0, dead_bus_ids="",
                   new_gens_off=0, islands=np.nan, cascade_iterations=0, prefault_shed_MW=0.0, sec=0.0)
        if wind_new:
            t0 = time.time()
            ppc = apply_state(copy.deepcopy(full), st)
            n_isl = add_reference_buses(ppc)
            ng = len(ppc["gen"]); ppc_d = load2disp(ppc)
            base = rundcopf_mp(copy.deepcopy(ppc_d), PPOPT)
            if not base["success"]:
                print(f"  h={h}: pre-fault OPF failed, hour skipped"); row["simulated"] = "prefault OPF failed"
            else:
                demand = -base["gen"][ng:, PMIN].sum(); served0 = -base["gen"][ng:, PG].sum()
                row["prefault_shed_MW"] = round(demand - served0, 3)
                if demand - served0 > 1.0:
                    print(f"  WARNING h={h}: pre-fault OPF sheds {demand - served0:.1f} MW (counted for this hour only)")
                # pass only the case arrays to the simulator: pypower's internal objects in the OPF result
                # (order, om, raw, ...) are deep-copied for every island and double the cascade time
                case = {k: base[k] for k in ("version", "baseMVA", "bus", "branch", "gen", "gencost", "genfuel", "gentype")}
                sim = CascadeSimulator(case, verbose=verbose)
                res = sim.run(faulty_lines=np.array(wind_new, dtype=int))
                casc, nd, no = update_state(st, base, res.network_after, ng, wind_new, h)
                row.update(simulated=True, cascade_trips=len(casc), cascade_branches=";".join(map(str, casc)),
                           cascade_iter_branches="|".join(";".join(str(int(b)) for b in it) for it in res.ovld_lines_log),
                           new_dead_buses=len(nd), dead_bus_ids=";".join(map(str, nd)), new_gens_off=len(no), islands=res.n_active_islands[-1] if res.n_active_islands else n_isl,
                           cascade_iterations=len(res.ovld_lines_log), sec=round(time.time() - t0, 1))
        frac = np.array([st["load_lost_frac"].get(int(b), 0.0) for b in ids])
        ens = float((frac * pd_full).sum()) + (row["prefault_shed_MW"] if row["simulated"] is True and row["prefault_shed_MW"] > 1.0 else 0.0)
        row.update(demand_MW=round(pd_tot, 1), disconnected_MW=round(ens, 1), disconnected_pct=round(100 * ens / pd_tot, 3),
                   branches_out_total=len(st["branches_out"]), dead_buses_total=len(st["dead_buses"]), gens_off_total=len(st["gens_off"]))
        rows.append(row)
        for b, f in st["load_lost_frac"].items():                 # per-bus demand lost this hour (MW), all hours
            mw = f * float(pd_full[ids == b][0]) if f > 0 else 0.0
            if mw > 0.01: lost_rows.append(dict(h=h, snapshot=snap, bus_id=int(b), lost_frac=round(f, 4), lost_MW=round(mw, 2)))
        if row["simulated"] is True:
            print(f"  h={h:2d} {snap}: wind {len(wind_new)}, cascade {row['cascade_trips']} ({row['cascade_iterations']} iter), "
                  f"dead buses +{row['new_dead_buses']}, gens off +{row['new_gens_off']}, disconnected {ens:.0f} MW "
                  f"({row['disconnected_pct']:.2f}%), {row['sec']}s")
    out = pd.DataFrame(rows)
    tag = f"cascade_sample{sample}" + (f"_rating{rating_factor:g}" if rating_factor != 1.0 else "")
    out.to_csv(os.path.join(out_dir, tag + ".csv"), index=False)
    pd.DataFrame(lost_rows, columns=["h", "snapshot", "bus_id", "lost_frac", "lost_MW"]).to_csv(os.path.join(out_dir, tag + "_lostload.csv"), index=False)
    ens_mwh = out.disconnected_MW.sum()
    print(f"  total: {int(out.wind_failures.sum())} wind failures, {int(out.cascade_trips.sum())} cascade trips, "
          f"{len(st['dead_buses'])} dead buses, {len(st['gens_off'])} generators off; energy not served {ens_mwh:.0f} MWh "
          f"over {len(out)} h; peak disconnected {out.disconnected_MW.max():.0f} MW; {time.time()-t_all:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("storms", nargs="*", default=STORMS, help=f"storm folder(s) from {STORMS}; default: all")
    ap.add_argument("--sample", type=int, default=None, help="wind-damage sample; default: random")
    ap.add_argument("--samples", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None, help="run samples FIRST..LAST inclusive")
    ap.add_argument("--verbose", action="store_true", help="print CascadeSimulator iterations")
    ap.add_argument("--rating-factor", type=float, default=1.0, help="scale all branch ratings (e.g. 1.2 = +20%%)")
    ap.add_argument("--no-tiebreak", action="store_true", help="do not add the 1e-6 x index cost tie-break")
    ap.add_argument("--root", default=None, help="repo root (default: two levels above this script)")
    a = ap.parse_args()
    bad = [x for x in a.storms if x not in STORMS]
    if bad: ap.error(f"unknown storm {bad}; choose from {STORMS}")
    for s in a.storms:
        if a.samples:
            for k in range(a.samples[0], a.samples[1] + 1):
                run(s, k, a.verbose, a.rating_factor, not a.no_tiebreak)
        else:
            run(s, a.sample, a.verbose, a.rating_factor, not a.no_tiebreak)
