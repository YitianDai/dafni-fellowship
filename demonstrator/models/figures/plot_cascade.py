"""
Figures for one storm cascade run, and for the ensemble when several samples exist.

Inputs (repo layout, override with --root):
  data/network/<storm>/      bus.csv, branch.csv, load_Pd.csv
  data/era5/                 era5_<storm>.nc (i10fg, u10, v10, msl), gb_land_mask_49x65.npy
  results/<storm>/           wind_damage.csv, cascade_sample<k>.csv, cascade_sample<k>_lostload.csv
Outputs in results/<storm>/figs/:
  <storm>_cascade_frames/frame_<hh>_<time>.png   one composite frame per hour (map + four time panels)
  <storm>_cascade.gif              the frames as an animation
  <storm>_cascade_viewer.html      the frames with a slider (self-contained)
  <storm>_cascade_stages.png       stage diagram of the largest cascade hour
  <storm>_ensemble_map.png, <storm>_ensemble_ens.png   when >= 5 cascade samples exist
Usage:  python plot_cascade.py Arwen2021 --sample 330 [--root REPO] [--no-frames]
"""
import os, io, sys, glob, base64, argparse, logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)   # silence the "Ignoring fixed x limits" notice from the map aspect
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from global_land_mask import globe

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERA5 = {"Arwen2021": "era5_arwen_2021.nc", "Eunice2022": "era5_eunice_2022.nc"}
TITLE = {"Arwen2021": "Storm Arwen", "Eunice2022": "Storm Eunice"}
# gust classes and colours as in the windstorm animation, lightened to 45 %
GUST_LEVELS = [0, 10, 15, 20, 25, 30, 35, 40, 45, 99]
GUST_COLS_FULL = ["#f4f2ee", "#faf3d8", "#fff1b8", "#fed47a", "#fdae3c", "#f07e12", "#d64f05", "#a62a02", "#5e1000"]
lighten = lambda c, f: tuple(1 - (1 - x) * f for x in to_rgb(c))
GUST_COLS = [lighten(c, 0.45) for c in GUST_COLS_FULL]
NET, C_WIND, C_CASC, C_LOST, ISO, GUST, SERVED = "#1e6e76", "#d62728", "#ff7f0e", "#6a3d9a", "#9aa5ad", "#f07e12", "#bfd4d6"
EXTENT = (-7.2, 2.1, 49.8, 59.2)


# ------------------------------------------------------------------ data
class Case:
    def __init__(self, storm, sample, root):
        self.storm, self.sample = storm, sample
        d = os.path.join(root, "data", "network", storm); r = os.path.join(root, "results", storm)
        self.bus = pd.read_csv(f"{d}/bus.csv").set_index("bus_id"); self.br = pd.read_csv(f"{d}/branch.csv").set_index("branch_idx")
        self.PD = pd.read_csv(f"{d}/load_Pd.csv", index_col=0); self.PD.columns = self.PD.columns.astype(int)
        self.cas = pd.read_csv(f"{r}/cascade_sample{sample}.csv")
        lf = f"{r}/cascade_sample{sample}_lostload.csv"; self.lost = pd.read_csv(lf) if os.path.exists(lf) else None
        wd = pd.read_csv(f"{r}/wind_damage.csv"); self.wd = wd[wd["sample"] == sample]
        self.ds = xr.open_dataset(os.path.join(root, "data", "era5", ERA5[storm]))
        self.mask = np.load(os.path.join(root, "data", "era5", "gb_land_mask_49x65.npy"))
        self.times = pd.to_datetime(self.cas.snapshot); t = self.times.values
        self.gust = self.ds["i10fg"].sel(valid_time=t).values; self.msl = self.ds["msl"].sel(valid_time=t).values / 100
        self.gust_gb = np.nanmax(np.where(self.mask[None], self.gust, np.nan).reshape(len(t), -1), 1)
        self.lon, self.lat = self.ds.longitude.values, self.ds.latitude.values
        self.net = self.br[self.br.kind.isin(["overhead", "mixed", "cable", "hvdc_equiv"])].index
        flat = np.arange(49.5, 61.0, 0.01); flon = np.arange(-11, 5, 0.01); FLON, FLAT = np.meshgrid(flon, flat)
        self.coast = (flon, flat, globe.is_land(FLAT, FLON).astype(float))
        self.figs = os.path.join(r, "figs"); os.makedirs(self.figs, exist_ok=True)

    def segs(self, idx):
        b = self.bus; return [[(b.lon[f], b.lat[f]), (b.lon[t], b.lat[t])] for f, t in zip(self.br.loc[idx, "from_bus"], self.br.loc[idx, "to_bus"])]

    def state(self, h):
        """wind-failed and cascade-tripped branches and buses blacked out, cumulative to hour h."""
        wind = sorted(set(self.wd[self.wd.h <= h].branch_idx)); casc, dead = [], []
        for _, r in self.cas[self.cas.h <= h].iterrows(): casc += split(r.cascade_branches); dead += split(r.dead_bus_ids)
        return wind, sorted(set(casc)), sorted(set(dead))

    def lost_mw(self, h):
        if self.lost is not None:
            x = self.lost[self.lost.h == h]; return dict(zip(x.bus_id.astype(int), x.lost_MW))
        return {b: float(self.PD.iloc[h][b]) for b in self.state(h)[2]}


def split(s):
    return [int(x) for x in str(s).split(";")] if isinstance(s, str) and s else []


# ------------------------------------------------------------------ map
def draw_map(ax, C, h, wind, casc, lost, show_cbar=True):
    lon, lat, dl = C.lon, C.lat, 0.25
    el = np.append(lon - dl / 2, lon[-1] + dl / 2); ea = np.append(lat + dl / 2, lat[-1] - dl / 2)
    ax.pcolormesh(el, ea, C.gust[h], cmap=ListedColormap(GUST_COLS), norm=BoundaryNorm(GUST_LEVELS, len(GUST_COLS)), zorder=0)
    cs = ax.contour(lon, lat, gaussian_filter(C.msl[h], 1), levels=np.arange(940, 1040, 4), colors=ISO, linewidths=0.6, zorder=2); ax.clabel(cs, fmt="%d", fontsize=5.5, colors=ISO)
    ax.contour(*C.coast, levels=[0.5], colors="#6f7b85", linewidths=0.7, zorder=2)
    ax.add_collection(LineCollection(C.segs(C.net), colors=NET, linewidths=0.7, zorder=3))
    if len(casc): ax.add_collection(LineCollection(C.segs(casc), colors=C_CASC, linewidths=2.0, zorder=5))
    if len(wind): ax.add_collection(LineCollection(C.segs(wind), colors=C_WIND, linewidths=2.6, zorder=6))
    lb = [b for b, m in lost.items() if m > 0.5]
    if lb:
        mw = np.array([lost[b] for b in lb]); ax.scatter(C.bus.lon[lb], C.bus.lat[lb], s=10 + mw * 0.6, c=C_LOST, alpha=0.85, zorder=7, edgecolor="white", linewidth=0.6)
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1 / np.cos(np.radians(54.5)), adjustable="datalim"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
    ax.legend(handles=[Line2D([], [], color=C_WIND, lw=2.6, label="wind-damaged line"), Line2D([], [], color=C_CASC, lw=2, label="cascade-tripped line"),
                       Line2D([], [], marker="o", color="w", markerfacecolor=C_LOST, markersize=7, label="demand lost (area = MW)")],
              loc="lower right", fontsize=6.5, framealpha=0.9, edgecolor="none")
    if show_cbar:
        cax = ax.inset_axes([0.04, 0.05, 0.03, 0.34]); cb = plt.colorbar(plt.cm.ScalarMappable(cmap=ListedColormap(GUST_COLS), norm=BoundaryNorm(GUST_LEVELS, len(GUST_COLS))), cax=cax, ticks=GUST_LEVELS[1:-1])
        cb.ax.tick_params(labelsize=6, length=2); cb.set_label("10 m gust (m/s)", fontsize=6.5)


# ------------------------------------------------------------------ composite frame
def frame(C, h):
    cas, hrs, times = C.cas, C.cas.h.values, C.times; k = h + 1
    wind, casc, dead = C.state(h); lost = C.lost_mw(h)
    fig = plt.figure(figsize=(12.2, 7.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[0.78, 1.2], height_ratios=[1, 1, 1, 0.6], hspace=0.6, wspace=0.12, left=0.015, right=0.985, top=0.885, bottom=0.075)
    ax = fig.add_subplot(gs[:, 0]); draw_map(ax, C, h, wind, casc, lost)
    fig.text(0.015, 0.945, f"{TITLE[C.storm]}   {times[h]:%a %d %b %Y %H:%M} UTC", fontsize=14, weight="bold")
    fig.text(0.015, 0.918, f"hour {h+1} of {len(cas)}   |   lines out {int(cas.branches_out_total[h])}     buses blacked out {int(cas.dead_buses_total[h])}     "
                           f"generators off {int(cas.gens_off_total[h])}     demand lost {cas.disconnected_MW[h]:.0f} MW ({cas.disconnected_pct[h]:.1f} %)", fontsize=8.5, color="#333333")
    def style(a, title):
        a.set_title(title, fontsize=8.5, loc="left", pad=3); a.tick_params(labelsize=7); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.axvline(h, color="#999999", lw=0.8, ls="--"); a.set_xlim(-0.5, len(cas) - 0.5)
    a1 = fig.add_subplot(gs[0, 1]); a1.plot(hrs[:k], C.gust_gb[:k], color=GUST, lw=1.6); a1.fill_between(hrs[:k], C.gust_gb[:k], color=GUST, alpha=0.18); a1.plot(h, C.gust_gb[h], "o", color=GUST)
    style(a1, "Peak gust over GB (m/s)"); a1.set_ylim(0, max(45, C.gust_gb.max() * 1.1))
    a2 = fig.add_subplot(gs[1, 1], sharex=a1); a2.bar(hrs[:k], cas.wind_failures[:k], color=C_WIND, label="wind-damaged"); a2.bar(hrs[:k], cas.cascade_trips[:k], bottom=cas.wind_failures[:k], color=C_CASC, label="cascade-tripped")
    style(a2, "Lines lost in the hour (dotted: cumulative, right axis)"); a2.set_ylim(0, max(5, (cas.wind_failures + cas.cascade_trips).max() * 1.15)); a2.legend(fontsize=6.5, frameon=False, loc="upper left")
    a2b = a2.twinx(); a2b.plot(hrs[:k], cas.branches_out_total[:k], color=NET, lw=1, ls=":"); a2b.set_ylim(0, cas.branches_out_total.max() * 1.15 + 1); a2b.tick_params(labelsize=7, colors=NET); a2b.spines["top"].set_visible(False); a2b.set_xlim(-0.5, len(cas) - 0.5)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1); a3.fill_between(hrs[:k], cas.disconnected_MW[:k], color=C_LOST, alpha=0.22); a3.plot(hrs[:k], cas.disconnected_MW[:k], color=C_LOST, lw=1.8); a3.plot(h, cas.disconnected_MW[h], "o", color=C_LOST)
    style(a3, f"Demand disconnected (MW); energy not served so far {cas.disconnected_MW[:k].sum()/1e3:.1f} GWh"); a3.set_ylim(0, cas.disconnected_MW.max() * 1.25 + 1)
    a4 = fig.add_subplot(gs[3, 1], sharex=a1); served = (cas.demand_MW - cas.disconnected_MW) / 1e3
    a4.bar(hrs[:k], served[:k], color=SERVED, width=0.8, label="served"); a4.bar(hrs[:k], cas.disconnected_MW[:k] / 1e3, bottom=served[:k], color=C_LOST, width=0.8, label="lost")
    style(a4, "GB demand (GW): served (pale teal) and lost (purple, on top)"); a4.set_ylim(0, 50); a4.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    for a in (a1, a2, a3): plt.setp(a.get_xticklabels(), visible=False)
    a4.set_xticks(hrs[::3]); a4.set_xticklabels([f"{t:%H:%M}" for t in times[::3]], fontsize=7)
    pb = fig.add_axes([0.015, 0.018, 0.97, 0.012]); pb.set_xlim(0, len(cas) - 1); pb.set_ylim(0, 1); pb.axis("off"); pb.barh(0.5, len(cas) - 1, height=1, color="#eeeeee"); pb.barh(0.5, h, height=1, color=GUST)
    fig.text(0.015, 0.035, f"{times[0]:%d %b %H:%M}", fontsize=7); fig.text(0.985, 0.035, f"{times.iloc[-1]:%d %b %H:%M}", fontsize=7, ha="right")
    return fig


def fig_frames(C, dpi=95):
    from PIL import Image
    frames, labels = [], []
    for h in range(len(C.cas)):
        r = C.cas.iloc[h]; fig = frame(C, h)
        fdir = os.path.join(C.figs, f"{C.storm}_cascade_frames"); os.makedirs(fdir, exist_ok=True)
        fig.savefig(os.path.join(fdir, f"frame_{h:02d}_{C.times[h]:%Y%m%d_%H%M}.png"), dpi=110)
        buf = io.BytesIO(); fig.savefig(buf, dpi=dpi); plt.close(fig); frames.append(buf.getvalue())
        labels.append(f"{C.times[h]:%d %b %H:%M}" + (f"   wind +{int(r.wind_failures)}, cascade +{int(r.cascade_trips)}, demand lost {r.disconnected_MW:.0f} MW" if r.wind_failures or r.cascade_trips else ""))
    imgs = [Image.open(io.BytesIO(f)).convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    imgs[0].save(os.path.join(C.figs, f"{C.storm}_cascade.gif"), save_all=True, append_images=imgs[1:], duration=[1200 if "wind" in l else 600 for l in labels], loop=0)
    b64 = [base64.b64encode(f).decode() for f in frames]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{TITLE[C.storm]} cascade</title>
<style>body{{font-family:system-ui,sans-serif;margin:14px;background:#fff}} img{{max-width:100%;height:auto;border:1px solid #e5e5e5}} .bar{{display:flex;gap:12px;align-items:center;margin:8px 0}} #lab{{min-width:300px;font-size:14px}}</style></head><body>
<div class="bar"><input type="range" id="s" min="0" max="{len(frames)-1}" value="0" style="width:480px"><button onclick="play()">play / pause</button><span id="lab"></span></div>
<img id="im"><script>const F={b64!r};const L={labels!r};const im=document.getElementById('im'),lab=document.getElementById('lab'),s=document.getElementById('s');
function show(i){{im.src='data:image/png;base64,'+F[i];lab.textContent=L[i];s.value=i}} s.oninput=()=>show(+s.value); show(0);
let t=null;function play(){{if(t){{clearInterval(t);t=null;return}} let i=+s.value;t=setInterval(()=>{{i=(i+1)%F.length;show(i);}},800)}}
document.addEventListener('keydown',e=>{{if(e.key==='ArrowRight')show(Math.min(+s.value+1,F.length-1));if(e.key==='ArrowLeft')show(Math.max(+s.value-1,0))}});</script></body></html>"""
    open(os.path.join(C.figs, f"{C.storm}_cascade_viewer.html"), "w").write(html)
    return len(frames)


# ------------------------------------------------------------------ timeline, stages, ensemble
def fig_stages(C):
    cas = C.cas[C.cas.cascade_trips > 0]
    if cas.empty: return
    r = cas.sort_values("cascade_trips", ascending=False).iloc[0]; h = int(r.h)
    its = [split(x) for x in str(r.cascade_iter_branches).split("|")] if isinstance(r.cascade_iter_branches, str) else []
    wind = sorted(set(C.wd[C.wd.h == h].branch_idx)); dead = split(r.dead_bus_ids)
    levels = [("wind failures", wind, C_WIND, "white")] + [(f"iteration {i+1}: overload trips", it, C_CASC, "black") for i, it in enumerate(its)] + [("buses blacked out", dead, "#222222", "white")]
    nmax = max(len(l[1]) for l in levels); dx = 3.4; br, bus = C.br, C.bus
    mid = lambda b: ((bus.lon[br.from_bus[b]] + bus.lon[br.to_bus[b]]) / 2, (bus.lat[br.from_bus[b]] + bus.lat[br.to_bus[b]]) / 2)
    fig, ax = plt.subplots(figsize=(2.2 + dx * len(levels), 1.6 + 0.3 * nmax)); pos = {}
    for k, (name, items, col, tc) in enumerate(levels):
        for j, it in enumerate(items):
            y = (nmax - 1) / 2 - (j - (len(items) - 1) / 2); pos[(k, it)] = (k * dx, y)
            lab = f"{br.pypsa_name[it]}  {br.kV[it]:.0f} kV" if k < len(levels) - 1 else f"{bus.pypsa_name[it]}  {bus.baseKV[it]:.0f} kV"
            ax.text(k * dx, y, lab, fontsize=6.5, ha="center", va="center", color=tc, zorder=3, bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none", alpha=0.9))
        ax.text(k * dx, nmax - 0.2, f"{name} ({len(items)})", fontsize=8, ha="center", weight="bold")
    for k in range(1, len(levels)):
        prev = levels[k - 1][1]
        for it in levels[k][1]:
            p = mid(it) if k < len(levels) - 1 else (bus.lon[it], bus.lat[it]); q = min(prev, key=lambda b: (mid(b)[0] - p[0]) ** 2 + (mid(b)[1] - p[1]) ** 2)
            (x0, y0), (x1, y1) = pos[(k - 1, q)], pos[(k, it)]; ax.plot([x0 + 1.15, x1 - 1.15], [y0, y1], color="#b0b0b0", lw=0.6, zorder=1)
    ax.set_xlim(-1.7, dx * (len(levels) - 1) + 1.7); ax.set_ylim(-1, nmax + 0.6); ax.axis("off")
    ax.set_title(f"{TITLE[C.storm]} {C.times[h]:%d %b %H:%M}: cascade stages, {r.disconnected_MW:.0f} MW disconnected.\nGrey links join each element to the geographically nearest element of the previous stage (proximity, not a causal trace).", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(C.figs, f"{C.storm}_cascade_stages.png"), dpi=160); plt.close(fig)


def fig_ensemble(C, root):
    r = os.path.join(root, "results", C.storm); files = [f for f in sorted(glob.glob(f"{r}/cascade_sample*.csv")) if "lostload" not in f and "rating" not in f]
    if len(files) < 5: print("ensemble: fewer than 5 cascade samples, skipped"); return
    wdall = pd.read_csv(f"{r}/wind_damage.csv"); wind_n, casc_n, lost_n, ens = {}, {}, {}, []
    for f in files:
        k = int(os.path.basename(f)[len("cascade_sample"):-4]); c = pd.read_csv(f); ens.append(c.disconnected_MW.sum() / 1e3)
        for b in set(wdall[wdall["sample"] == k].branch_idx): wind_n[b] = wind_n.get(b, 0) + 1
        for s in c.cascade_branches.dropna():
            for b in set(split(s)): casc_n[b] = casc_n.get(b, 0) + 1
        lf = f.replace(".csv", "_lostload.csv")
        if os.path.exists(lf):
            for b in set(pd.read_csv(lf).bus_id): lost_n[b] = lost_n.get(b, 0) + 1
    n = len(files); fig, axs = plt.subplots(1, 3, figsize=(13, 6.4))
    for ax, (title, cnt, cmap) in zip(axs[:2], [("wind damage: share of samples in which the line fails", wind_n, "Reds"), ("cascade trips: share of samples in which the line trips", casc_n, "Oranges")]):
        ax.contour(*C.coast, levels=[0.5], colors="#6f7b85", linewidths=0.7, zorder=2); ax.add_collection(LineCollection(C.segs(C.net), colors="#c9d6d8", linewidths=0.6, zorder=1))
        idx = sorted(cnt, key=cnt.get); fr = np.array([cnt[b] / n for b in idx])
        lc = LineCollection(C.segs(idx), cmap=cmap, norm=plt.Normalize(0, max(fr.max(), 0.05)), linewidths=1 + 3 * fr, zorder=5); lc.set_array(fr); ax.add_collection(lc); fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.01)
        ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1 / np.cos(np.radians(54.5))); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5, loc="left")
    ax = axs[2]; ax.contour(*C.coast, levels=[0.5], colors="#6f7b85", linewidths=0.7, zorder=2); ax.add_collection(LineCollection(C.segs(C.net), colors="#c9d6d8", linewidths=0.6, zorder=1))
    idx = list(lost_n); fr = np.array([lost_n[b] / n for b in idx])
    if len(idx): sc = ax.scatter(C.bus.lon[idx], C.bus.lat[idx], s=6 + 60 * fr, c=fr, cmap="Purples", vmin=0, vmax=max(fr.max(), 0.05), zorder=6, edgecolor="k", linewidth=0.3); fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
    ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:]); ax.set_aspect(1 / np.cos(np.radians(54.5))); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("demand loss: share of samples in which the bus loses demand", fontsize=8.5, loc="left")
    fig.suptitle(f"{TITLE[C.storm]}: ensemble over {n} wind-damage samples; energy not served mean {np.mean(ens):.1f} GWh, median {np.median(ens):.1f}, 95th pct {np.percentile(ens, 95):.1f}, max {np.max(ens):.1f}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(C.figs, f"{C.storm}_ensemble_map.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3)); ax.hist(ens, bins=15, color="#666666"); ax.set_xlabel("energy not served over the window (GWh)"); ax.set_ylabel("samples"); ax.set_title(f"{TITLE[C.storm]}: {n} samples", fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(C.figs, f"{C.storm}_ensemble_ens.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("storm", choices=list(ERA5)); ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--root", default=ROOT); ap.add_argument("--no-frames", action="store_true"); a = ap.parse_args()
    C = Case(a.storm, a.sample, a.root)
    fig_stages(C); print("stages")
    if not a.no_frames: print("frames:", fig_frames(C))
    fig_ensemble(C, a.root); print("done ->", C.figs)
