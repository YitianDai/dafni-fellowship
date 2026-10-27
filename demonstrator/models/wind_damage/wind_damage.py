"""
Hourly wind-induced line damage for a storm window (no power flow, no cascade).

Inputs (repo layout, override with --root):
  data/network/<storm>/branch.csv, bus.csv   network topology; exposed = kind 'overhead' or 'mixed', any voltage,
                                             with geometry_known (both ends located apart; 52 short circuits are not)
  data/era5/era5_<storm_lower>.nc            ERA5 hourly i10fg (gust), u10, v10 on a 0.25 deg grid
Model:
  each exposed line is a straight route between its end buses; it crosses one or more ERA5 cells
  per cell and hour: gust v, wind direction from (u10, v10), attack angle Omega to the line bearing
  v_eff = v * max(|sin Omega|, S_MIN)                 (form from IEC 60826: conductor load ~ (v sin Omega)^2)
  line hourly v_eff = max over its cells
  P_line(h) = Phi((ln v_eff - MU) / SIGMA)            (Dai and Preece 2025 SEGAN lognormal, whole-line, no length scaling)
  Monte Carlo: each surviving line fails with P_line(h) each hour; a failed line stays failed for the rest of the window
Output in results/<storm>/:
  wind_damage.csv   one row per sampled line failure: sample, hour index, snapshot, branch_idx, line name, kV, owner,
                    OHL km, effective gust v_eff (m/s) and failure probability P_line in that hour
  (a per-line summary is printed to the console only)
Usage:  python wind_damage.py [Arwen2021] [Eunice2022] [--samples 1000] [--seed 42] [--root REPO]   (no storm given: both)
"""
import os, argparse
import numpy as np, pandas as pd, xarray as xr
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (models/wind_damage/..)
MU, SIGMA = 3.75, 0.14          # SEGAN fragility (median exp(3.75) = 42.5 m/s gust)
S_MIN = 0.5                     # floor on |sin Omega| (tower and insulator load in along-line wind)
STEP_KM = 0.5                   # sampling step along the route to find the cells crossed
STORMS = {"Arwen2021":  ("era5_arwen_2021.nc",  "2021-11-26 12:00", "2021-11-27 09:00"),
          "Eunice2022": ("era5_eunice_2022.nc", "2022-02-18 00:00", "2022-02-19 00:00")}


def line_geometry(branch, bus, lat_grid, lon_grid):
    """Exposed lines with bearing (deg from north, 0-180) and the ERA5 cells (lat, lon index) they cross."""
    e = branch[branch.kind.isin(["overhead", "mixed"]) & (branch.geometry_known.astype(str) == "True")].copy()
    b = bus.set_index("bus_id")
    la0, lo0 = b.lat[e.from_bus].values, b.lon[e.from_bus].values
    la1, lo1 = b.lat[e.to_bus].values, b.lon[e.to_bus].values
    kx = 111.2 * np.cos(np.radians((la0 + la1) / 2)); dx = (lo1 - lo0) * kx; dy = (la1 - la0) * 111.2
    e["straight_km"] = np.hypot(dx, dy)
    e["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 180
    e["ux"], e["uy"] = np.sin(np.radians(e.bearing_deg)), np.cos(np.radians(e.bearing_deg))   # unit vector (east, north)
    cells = []
    for i, (a0, o0, a1, o1, L) in enumerate(zip(la0, lo0, la1, lo1, e.straight_km)):
        f = np.linspace(0, 1, max(int(np.ceil(L / STEP_KM)) + 1, 2))
        ia = np.abs(lat_grid[None, :] - (a0 + f * (a1 - a0))[:, None]).argmin(1)
        io = np.abs(lon_grid[None, :] - (o0 + f * (o1 - o0))[:, None]).argmin(1)
        cells.append(sorted(set(zip(ia, io))))
    e["cells"] = cells; e["n_cells"] = [len(c) for c in cells]
    return e.reset_index(drop=True)


def hourly_veff(e, ds, times):
    """v_eff (hours x lines): max over each line's cells of gust * max(|sin Omega|, S_MIN)."""
    g = ds["i10fg"].sel(valid_time=times).values; u = ds["u10"].sel(valid_time=times).values; v = ds["v10"].sel(valid_time=times).values
    ws = np.hypot(u, v); wx, wy = u / np.where(ws > 0, ws, 1), v / np.where(ws > 0, ws, 1)
    out = np.zeros((len(times), len(e)))
    for j, r in e.iterrows():
        ia = np.array([c[0] for c in r.cells]); io = np.array([c[1] for c in r.cells])
        s = np.abs(r.ux * wy[:, ia, io] - r.uy * wx[:, ia, io])             # |sin Omega| = |line x wind|
        out[:, j] = (g[:, ia, io] * np.maximum(s, S_MIN)).max(axis=1)
    return out


def main(storm, n_samples, seed):
    era5, start, end = STORMS[storm]; d = os.path.join(ROOT, "data", "network", storm); out = os.path.join(ROOT, "results", storm); os.makedirs(out, exist_ok=True)
    branch = pd.read_csv(os.path.join(d, "branch.csv")); bus = pd.read_csv(os.path.join(d, "bus.csv"))
    ds = xr.open_dataset(os.path.join(ROOT, "data", "era5", era5))
    times = pd.date_range(start, end, freq="h")
    e = line_geometry(branch, bus, ds.latitude.values, ds.longitude.values)
    veff = hourly_veff(e, ds, times)
    P = norm.cdf((np.log(np.maximum(veff, 1e-6)) - MU) / SIGMA)

    cols = e.branch_idx.values

    rng = np.random.default_rng(seed)
    alive = np.ones((n_samples, len(e)), bool); rows = []
    fails_per_hour = np.zeros((n_samples, len(times)), int)
    for h in range(len(times)):
        new = alive & (rng.random((n_samples, len(e))) < P[h])
        alive &= ~new; fails_per_hour[:, h] = new.sum(1)
        s_idx, l_idx = np.nonzero(new)
        rows.append(pd.DataFrame({"sample": s_idx, "h": h, "snapshot": times[h], "branch_idx": cols[l_idx],
                                  "pypsa_name": e.pypsa_name.values[l_idx], "kV": e.kV.values[l_idx],
                                  "owner": e.owner.values[l_idx], "ohl_km": e.ohl_km.values[l_idx],
                                  "v_eff": veff[h, l_idx].round(2), "P_line": P[h, l_idx].round(5)}))
    S = pd.concat(rows, ignore_index=True).sort_values(["sample", "h", "branch_idx"])
    S.to_csv(os.path.join(out, "wind_damage.csv"), index=False)

    p_once = 1 - np.prod(1 - P, axis=0)
    first = S.groupby("branch_idx").h.mean()
    lines = e[["branch_idx", "pypsa_name", "kind", "kV", "owner", "ohl_km", "bearing_deg", "n_cells"]].copy()
    lines["veff_max"] = veff.max(0); lines["P_hour_max"] = P.max(0); lines["P_at_least_once"] = p_once
    lines["peak_hour"] = times[P.argmax(0)]; lines["mean_failure_h"] = lines.branch_idx.map(first)
    lines["sampled_failure_share"] = lines.branch_idx.map(S.groupby("branch_idx").size() / n_samples).fillna(0)

    tot = fails_per_hour.sum(1)
    print(f"{storm}: {len(e)} exposed lines, "
          f"cells per line 1 to {e.n_cells.max()}, {len(times)} hours, {n_samples} samples, seed {seed}")
    print(f"  highest effective gust on any line: {veff.max():.1f} m/s; "
          f"lines with P_hour > 1%: {(P.max(0) > 0.01).sum()}, > 10%: {(P.max(0) > 0.1).sum()}")
    print(f"  lines failed per sample over the window: mean {tot.mean():.2f}, median {np.median(tot):.0f}, "
          f"95th pct {np.percentile(tot, 95):.0f}, max {tot.max()}; samples with no failure {100*(tot==0).mean():.0f}%")
    print("  expected new failures per hour:", np.round(fails_per_hour.mean(0), 2).tolist())
    top = lines.sort_values("P_at_least_once", ascending=False).head(8)
    print(top[["pypsa_name", "kV", "owner", "ohl_km", "veff_max", "P_at_least_once", "peak_hour"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("storms", nargs="*", default=list(STORMS),
                    help=f"storm folder(s) from {list(STORMS)}; default: all")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None, help="repo root (default: two levels above this script)")
    a = ap.parse_args()
    if a.root: ROOT = a.root
    bad = [x for x in a.storms if x not in STORMS]
    if bad: ap.error(f"unknown storm {bad}; choose from {list(STORMS)}")
    for storm in a.storms:
        main(storm, a.samples, a.seed)
