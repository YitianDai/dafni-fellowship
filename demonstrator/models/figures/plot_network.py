"""
Network-only map for a storm case: the GB transmission network as modelled (no hazard, no damage).
    python plot_network.py Arwen2021 [--root REPO]   ->  results/<storm>/figs/<storm>_network.png
Teal: overhead and mixed circuits exposed to wind. Grey: cables, HVDC equivalents and the 52 unlocated short circuits.
Rings: generation capacity at a bus (included units). Purple dots: peak demand at a bus over the window.
"""
import os, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from global_land_mask import globe
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TITLE = {"Arwen2021": "Storm Arwen case: 2021 network and fleet", "Eunice2022": "Storm Eunice case: 2022 network and fleet"}
EXTENT = (-7.2, 2.1, 49.8, 59.2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("storm"); ap.add_argument("--root", default=ROOT); a = ap.parse_args()
    d = os.path.join(a.root, "data", "network", a.storm); figs = os.path.join(a.root, "results", a.storm, "figs"); os.makedirs(figs, exist_ok=True)
    bus = pd.read_csv(f"{d}/bus.csv").set_index("bus_id"); br = pd.read_csv(f"{d}/branch.csv"); gen = pd.read_csv(f"{d}/gen.csv"); PD = pd.read_csv(f"{d}/load_Pd.csv", index_col=0)
    segs = lambda idx: [[(bus.lon[f], bus.lat[f]), (bus.lon[t], bus.lat[t])] for f, t in zip(br.loc[idx, "from_bus"], br.loc[idx, "to_bus"])]
    located = br.geometry_known.astype(str) == "True"
    exposed = br[br.kind.isin(["overhead", "mixed"]) & located].index
    other = br[br.kind.isin(["cable", "hvdc_equiv"]) | (br.kind.isin(["overhead", "mixed"]) & ~located)].index
    flat = np.arange(49.5, 61.0, 0.01); flon = np.arange(-11, 5, 0.01); FLON, FLAT = np.meshgrid(flon, flat); land = globe.is_land(FLAT, FLON).astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 8.6)); ax.set_facecolor("#fbfbfb")
    ax.contourf(flon, flat, land, levels=[0.5, 1.5], colors=["#f1f3f4"], zorder=0); ax.contour(flon, flat, land, levels=[0.5], colors="#6f7b85", linewidths=0.7, zorder=1)
    ax.add_collection(LineCollection(segs(other), colors="#9aa5ad", linewidths=0.7, zorder=2)); ax.add_collection(LineCollection(segs(exposed), colors="#1e6e76", linewidths=0.8, zorder=3))
    gb = gen[gen.included == 1].groupby("bus_id").p_nom.sum(); gb = gb[gb > 100]
    ax.scatter(bus.lon[gb.index], bus.lat[gb.index], s=4 + gb.values / 60, facecolor="none", edgecolor="#8c510a", linewidth=0.7, zorder=4)
    ld = PD.max(); ld.index = ld.index.astype(int); ld = ld[ld > 50]
    ax.scatter(bus.lon[ld.index], bus.lat[ld.index], s=2 + ld.values / 25, c="#6a3d9a", alpha=0.45, linewidth=0, zorder=4)
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1 / np.cos(np.radians(54.5))); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
    n_ex, km = len(exposed), br.loc[exposed, "ohl_km"].sum(); inc = gen[gen.included == 1]
    ax.set_title(f"{TITLE[a.storm]}\n{len(bus)} buses, {int((br.kind != 'transformer').sum())} circuits ({n_ex} overhead circuits exposed to wind, {km:,.0f} km), "
                 f"{int((br.kind == 'transformer').sum())} transformers\n{len(inc)} generators, {inc.p_nom.sum()/1e3:.0f} GW installed; peak demand in the window {PD.sum(axis=1).max()/1e3:.1f} GW", fontsize=7.5, loc="left")
    ax.legend(handles=[Line2D([], [], color="#1e6e76", lw=1.2, label="overhead circuit (wind-exposed)"), Line2D([], [], color="#9aa5ad", lw=1.0, label="cable, HVDC equivalent or unlocated circuit"),
                       Line2D([], [], marker="o", color="w", markeredgecolor="#8c510a", markerfacecolor="none", markersize=7, label="generation at bus (ring area = MW)"),
                       Line2D([], [], marker="o", color="w", markerfacecolor="#6a3d9a", alpha=0.5, markersize=7, label="peak demand at bus (area = MW)")], loc="lower right", fontsize=6.5, framealpha=0.9, edgecolor="none")
    fig.tight_layout(); out = os.path.join(figs, f"{a.storm}_network.png"); fig.savefig(out, dpi=170); print("wrote", out)
