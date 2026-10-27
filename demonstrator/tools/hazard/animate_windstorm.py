"""
Hourly windstorm snapshots over Great Britain, and a GIF of them.

Input: either the Excel export (sheet "hourly_long": time, latitude, longitude, gust, u10, v10, msl)
or the ERA5 NetCDF directly. Point it at the bias-corrected Excel file later and nothing else changes.

    Set the parameters in the "run parameters" block below, then:  python animate_windstorm.py

Outputs, in figs/ (<name> = <storm>, or <storm>_network with the transmission network overlaid):
    frames/<name>/frame_###.png    one frame per hour
    <name>_windstorm.gif           the animation
    <name>_viewer.html             the frames with a time slider

Requires: pandas xarray netCDF4 matplotlib pillow cartopy global-land-mask (pyarrow optional, for caching)
"""
from __future__ import annotations
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from global_land_mask import globe
from PIL import Image

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, FIG_DIR = ROOT / "data", ROOT / "figs"
EXTENT = (-8.0, 2.5, 49.5, 61.0)                     # map view: lon_min, lon_max, lat_min, lat_max (GB window)
DATA_EXTENT = (-11.0, 5.0, 49.0, 61.0)               # data kept for drawing: wider than the view so the corners fill
BOUNDS = [0, 10, 15, 20, 25, 30, 35, 40, 45, 60]     # m/s class edges; two pale classes below 15 keep the map continuous
COLOURS = ["#f4f2ee", "#faf3d8",                     # 0 to 10, 10 to 15: calm, near neutral
           "#fff1b8", "#fed47a", "#fdae3c", "#f07e12", "#d64f05", "#a62a02", "#5e1000"]   # 15 m/s upward: pale yellow to dark red
SEA_COLOUR, LAND_COLOUR = "#d9dde0", "#fbfbfa"       # base map under the gust cells in the network variant
ARROW_STRIDE = 3                                     # one direction arrow per 3 x 3 cells
ACTIVE_MS = 25                                       # storm is "active" while the land peak gust >= this (m/s)
MARGIN_H = 3                                         # hours kept either side of the active period
ACTIVE_RUN = 3                                       # the threshold must hold for this many consecutive hours
INK, INK_SOFT, GRID = "#1E1A26", "#4E4757", "#D9D2E0"
PC = ccrs.PlateCarree()

# ---------------------------------------------------------------- run parameters (edit these, then run the script)
INPUT = None            # path to era5_<storm>.xlsx or .nc, e.g. "data/era5_arwen_2021.xlsx"; None = every storm in data/
FPS = 1.5               # GIF frames per second (one frame = one hour)
START, END = None, None # explicit window (UTC, e.g. "2021-11-26T12"); overrides STORM_WINDOWS and the automatic rule
# Fixed windows per storm (UTC). A storm listed here uses these hours; one not listed uses the automatic rule.
STORM_WINDOWS = {
    "arwen_2021": ("2021-11-26T12", "2021-11-27T12"),
    "eunice_2022": ("2022-02-18T00", "2022-02-19T00"),
}
# Outputs written per run: "plain" = wind field everywhere, coastlines only;
# "network" = wind field on GB land only, over a sea/land base map, with the transmission network overlaid.
VARIANTS = ("plain", "network")
NETWORK_LINES = DATA_DIR / "lines_overhead.csv"     # overhead line table written by line_exposure.py
NETWORK_COLOUR = "#5c5c5c"      # mid grey: outside the gust colour scale, uniform width
NETWORK_WIDTH = 0.8
HALO_WIDTH = 0.8                # faint white halo under the lines (extra points of width) so they stay readable
HALO_ALPHA = 0.45


# ---------------------------------------------------------------- loading
def load(path: Path) -> xr.Dataset:
    """Return a Dataset with dims (time, latitude, longitude) and variables gust, u10, v10, msl."""
    if path.suffix == ".nc":
        ds = xr.open_dataset(path)
        t = "valid_time" if "valid_time" in ds.dims else "time"
        gust = next(v for v in ("fg10", "i10fg", "instantaneous_10m_wind_gust") if v in ds)
        ds = ds.rename({t: "time", gust: "gust"})
        return ds[[v for v in ("gust", "u10", "v10", "msl") if v in ds]]
    cache = path.with_suffix(".parquet")
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = pd.read_excel(path, sheet_name="hourly_long")
        try:
            df.to_parquet(cache)
        except Exception:
            pass
    tcol = next(c for c in df.columns if c in ("valid_time", "time"))
    gcol = next(c for c in df.columns if c in ("fg10", "i10fg", "gust", "gust_corrected"))
    df = df.rename(columns={tcol: "time", gcol: "gust"})
    df["time"] = pd.to_datetime(df["time"])
    keep = [c for c in ("gust", "u10", "v10", "msl") if c in df]
    return df.set_index(["time", "latitude", "longitude"])[keep].to_xarray()


def load_network():
    """Line end coordinates for the overlay, or None."""
    if not NETWORK_LINES.exists():
        print(f"[warn] network file not found: {NETWORK_LINES}")
        return None
    return pd.read_csv(NETWORK_LINES, index_col=0)


# ---------------------------------------------------------------- land mask
def land_mask(ds) -> np.ndarray:
    """True for cells whose centre is on GB land within the view. Cached beside the data."""
    lon, lat = ds.longitude.values, ds.latitude.values
    cache = DATA_DIR / f"gb_land_mask_{len(lat)}x{len(lon)}.npy"
    if cache.exists():
        return np.load(cache)
    LON, LAT = np.meshgrid(lon, lat)
    m = globe.is_land(LAT, LON)
    m &= ~((LON < -5.3) & (LAT > 51.3) & (LAT < 55.5))      # exclude the island of Ireland: GB land only
    m &= ~((LAT < 50.0) & (LON > -5.0))                      # exclude Normandy and the Channel Islands
    m &= (LON >= EXTENT[0]) & (LON <= EXTENT[1]) & (LAT >= EXTENT[2]) & (LAT <= EXTENT[3])   # inside the view only
    np.save(cache, m)
    return m


def land_peak(ds, mask) -> np.ndarray:
    """Peak gust over GB land, per hour."""
    return ds.gust.where(xr.DataArray(mask, dims=("latitude", "longitude"))).max(("latitude", "longitude")).values


# ---------------------------------------------------------------- drawing
def fine_land(extent):
    """Land/sea on a 0.02 degree grid, for crisp coastlines and land fill."""
    lo, la = np.arange(extent[0], extent[1], 0.02), np.arange(extent[2], extent[3], 0.02)
    LO, LA = np.meshgrid(lo, la)
    return lo, la, globe.is_land(LA, LO).astype(float)


def make_axes(fig):
    # transverse Mercator centred on GB (same family as the OS national grid): true shape, no E-W stretch
    proj = ccrs.TransverseMercator(central_longitude=-2.5, central_latitude=49.0, scale_factor=0.9996)
    ax = fig.add_axes([0.03, 0.10, 0.80, 0.80], projection=proj)
    ax.set_extent(EXTENT, crs=PC)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    return ax


def draw_network(ax, net):
    """Grey lines of one width over the wind field, on a faint white halo."""
    segs = [[(r.lon0, r.lat0), (r.lon1, r.lat1)] for r in net.itertuples()]
    ax.add_collection(LineCollection(segs, colors="white", linewidths=NETWORK_WIDTH + HALO_WIDTH, alpha=HALO_ALPHA,
                                     capstyle="round", zorder=3.4, transform=PC))
    ax.add_collection(LineCollection(segs, colors=NETWORK_COLOUR, linewidths=NETWORK_WIDTH, capstyle="round",
                                     zorder=3.5, transform=PC))


def draw_frame(fig, ds, i, track, mask, land_only=False, net=None):
    cmap = ListedColormap(COLOURS); cmap.set_under(COLOURS[0]); norm = BoundaryNorm(BOUNDS, cmap.N)
    fig.clf(); ax = make_axes(fig)
    g = ds.gust.isel(time=i)
    lon, lat, vals = g.longitude.values, g.latitude.values, g.values
    if land_only:                                     # sea/land base map; gust drawn on GB land only
        ax.set_facecolor(SEA_COLOUR)
        ax.contourf(*fine_land(DATA_EXTENT), levels=[0.5, 1.5], colors=[LAND_COLOUR], zorder=0.5, transform=PC)
    field = np.where(mask, vals, np.nan) if land_only else vals
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="nearest", zorder=1, transform=PC)
    ax.contour(*fine_land(EXTENT), levels=[0.5], colors="#6b6b6b", linewidths=0.7, zorder=3, transform=PC)   # coastline
    if net is not None:
        draw_network(ax, net)

    if "u10" in ds and "v10" in ds:                   # arrows: length and darkness scale with the gust
        st = ARROW_STRIDE
        u, v, gg = ds.u10.isel(time=i).values[::st, ::st], ds.v10.isel(time=i).values[::st, ::st], vals[::st, ::st]
        sp = np.maximum(np.hypot(u, v), 1e-6)
        shade = np.clip((gg - 10) / 30, 0, 1)         # 0 at 10 m/s, 1 at 40 m/s
        length = np.maximum(40 * shade ** 1.6, 4)     # calm cells get short arrows, strong cells long ones
        cols = plt.cm.Greys(0.25 + 0.60 * shade); cols[..., 3] = 0.30 + 0.60 * shade   # fade the calm ones
        LON, LAT = np.meshgrid(lon[::st], lat[::st])
        ax.quiver(LON, LAT, u / sp * length, v / sp * length, color=cols.reshape(-1, 4), scale=360,
                  width=0.0032, headwidth=3.2, headlength=4, pivot="mid", zorder=4, transform=PC)

    if "msl" in ds:                                   # isobars every 4 hPa: show the low and its gradient
        pm = ds.msl.isel(time=i).values / 100.0
        levels = np.arange(np.floor(np.nanmin(pm) / 4) * 4, np.ceil(np.nanmax(pm) / 4) * 4 + 1, 4)
        cs = ax.contour(lon, lat, pm, levels=levels, colors="#555555", linewidths=0.6, alpha=0.8, zorder=3, transform=PC)
        ax.clabel(cs, fmt="%d", fontsize=6.5, colors="#555555", inline=True, inline_spacing=2)

    if track is not None:                            # storm centre (MSLP minimum) and its path so far
        ax.plot(track.lon[: i + 1], track.lat[: i + 1], "-", color="#1f4e9c", lw=1.6, alpha=0.8, zorder=5, transform=PC)
        ax.plot(track.lon[: i + 1], track.lat[: i + 1], ".", color="#1f4e9c", ms=4, alpha=0.8, zorder=5, transform=PC)
        ax.plot(track.lon[i], track.lat[i], "o", ms=16, mfc="#1f4e9c", mec="white", mew=2.0, zorder=6, transform=PC)
        x, y = ax.projection.transform_point(track.lon[i], track.lat[i], PC)
        ax.annotate(f"{track.p[i]:.0f} hPa", (x, y), xytext=(11, 9), textcoords="offset points",
                    fontsize=8.5, color="#1f4e9c", weight="bold", zorder=7)

    # colour bar
    cax = fig.add_axes([0.86, 0.22, 0.025, 0.56])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, boundaries=BOUNDS, ticks=BOUNDS[1:-1])
    cb.ax.set_yticklabels([str(b) for b in BOUNDS[1:-1]]); cb.ax.tick_params(labelsize=8, colors=INK_SOFT)
    cb.set_label("10 m gust (m/s)", fontsize=9, color=INK_SOFT); cb.outline.set_edgecolor(GRID)

    # title and time bar
    t = pd.Timestamp(g.time.values)
    gl = np.where(mask, vals, np.nan)                 # peak over land only
    peak = float(np.nanmax(gl)); w = np.unravel_index(int(np.nanargmax(gl)), g.shape)
    fig.text(0.03, 0.955, t.strftime("%a %d %b %Y  %H:00 UTC"), fontsize=14, color=INK, weight="bold")
    fig.text(0.03, 0.925, f"Peak gust over land {peak:.0f} m/s at {lat[w[0]]:.2f}°N, {abs(lon[w[1]]):.2f}°{'W' if lon[w[1]] < 0 else 'E'}",
             fontsize=9.5, color=INK_SOFT)
    n = ds.sizes["time"]
    bar = fig.add_axes([0.03, 0.045, 0.80, 0.012]); bar.set_axis_off()
    bar.set_xlim(0, n); bar.set_ylim(0, 1)
    bar.barh(0.5, n, height=1, color=GRID); bar.barh(0.5, i + 1, height=1, color=COLOURS[4])
    fig.text(0.03, 0.02, pd.Timestamp(ds.time.values[0]).strftime("%d %b %H:00"), fontsize=8, color=INK_SOFT)
    fig.text(0.83, 0.02, pd.Timestamp(ds.time.values[-1]).strftime("%d %b %H:00"), fontsize=8, color=INK_SOFT, ha="right")


def storm_track(ds):
    if "msl" not in ds:
        return None
    rows = []
    LON, LAT = np.meshgrid(ds.longitude.values, ds.latitude.values)
    prev = None
    for i in range(ds.sizes["time"]):
        p = ds.msl.isel(time=i).values.astype(float)
        if prev is not None:                          # follow the same low: search within ~4 degrees of last position
            far = np.hypot((LON - prev[0]) * np.cos(np.radians(prev[1])), LAT - prev[1]) > 4.0
            p = np.where(far, np.nan, p)
        j = np.unravel_index(int(np.nanargmin(p)), p.shape)
        prev = (float(LON[j]), float(LAT[j]))
        rows.append((prev[0], prev[1], float(p[j]) / 100))
    return pd.DataFrame(rows, columns=["lon", "lat", "p"])


# ---------------------------------------------------------------- interactive viewer (HTML with a time slider)
def write_viewer(path: Path, frames, labels, fps: float) -> None:
    data = ["data:image/png;base64," + base64.b64encode(f.read_bytes()).decode() for f in frames]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{path.stem}</title>
<style>
 body{{margin:0;font-family:system-ui,sans-serif;background:#fff;color:#1E1A26;display:flex;flex-direction:column;align-items:center}}
 img{{max-width:100vw;max-height:88vh}}
 .bar{{display:flex;align-items:center;gap:.6rem;width:min(95vw,900px);padding:.5rem 0}}
 input[type=range]{{flex:1}} button{{font:inherit;padding:.25rem .6rem}} #t{{min-width:11em;text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body>
<img id="f" src="{data[0]}">
<div class="bar">
 <button id="play">Play</button><button id="prev">&#9664;</button><button id="next">&#9654;</button>
 <input type="range" id="s" min="0" max="{len(data) - 1}" value="0" step="1">
 <span id="t"></span>
</div>
<script>
const F={json.dumps(data)}, L={json.dumps(labels)};
const img=document.getElementById('f'), s=document.getElementById('s'), t=document.getElementById('t'), play=document.getElementById('play');
let timer=null;
function show(i){{i=Math.max(0,Math.min(F.length-1,i)); s.value=i; img.src=F[i]; t.textContent=L[i]+' UTC';}}
s.oninput=()=>show(+s.value);
document.getElementById('prev').onclick=()=>show(+s.value-1);
document.getElementById('next').onclick=()=>show(+s.value+1);
function toggle(){{ if(timer){{clearInterval(timer);timer=null;play.textContent='Play';}}
  else{{ play.textContent='Pause'; timer=setInterval(()=>show((+s.value+1)%F.length),{int(1000 / fps)}); }} }}
play.onclick=toggle;
document.addEventListener('keydown',e=>{{ if(e.key==='ArrowLeft')show(+s.value-1); if(e.key==='ArrowRight')show(+s.value+1); if(e.key===' '){{e.preventDefault();toggle();}} }});
show(0);
</script></body></html>"""
    path.write_text(html, encoding="utf-8")
    print(f"[ok] {path}  (slider viewer, open in a browser)")


# ---------------------------------------------------------------- main
def main():
    if INPUT:
        run(Path(INPUT))
        return
    inputs = sorted(p for p in DATA_DIR.glob("era5_*.xlsx") if not p.stem.endswith("_peakgust"))
    inputs = inputs or sorted(p for p in DATA_DIR.glob("era5_*.nc") if not p.stem.endswith("_peakgust"))
    for p in inputs:
        print(f"\n=== {p.name}")
        run(p)


def run(path: Path):
    ds = load(path).sortby("latitude", ascending=False)
    ds = ds.sel(longitude=slice(DATA_EXTENT[0], DATA_EXTENT[1]), latitude=slice(DATA_EXTENT[3], DATA_EXTENT[2]))
    key = path.stem.replace("era5_", "")
    mask = land_mask(ds)
    if START or END:
        ds = ds.sel(time=slice(START, END))
    elif key in STORM_WINDOWS:
        ds = ds.sel(time=slice(*STORM_WINDOWS[key]))
        print(f"[window] fixed for {key}: {STORM_WINDOWS[key][0]} to {STORM_WINDOWS[key][1]} ({ds.sizes['time']} h)")
    else:
        # Active window rule: first to last hour of a run of ACTIVE_RUN hours in which the land peak gust is
        # at or above ACTIVE_MS, plus MARGIN_H hours either side. Quiet hours before and after are dropped.
        dm = land_peak(ds, mask)
        starts = np.flatnonzero(np.convolve(dm >= ACTIVE_MS, np.ones(ACTIVE_RUN, int), "valid") == ACTIVE_RUN)
        if len(starts):
            i0, i1 = max(starts[0] - MARGIN_H, 0), min(starts[-1] + ACTIVE_RUN - 1 + MARGIN_H, len(dm) - 1)
            ds = ds.isel(time=slice(i0, i1 + 1))
            print(f"[window] {pd.Timestamp(ds.time.values[0]):%d %b %H:00} to "
                  f"{pd.Timestamp(ds.time.values[-1]):%d %b %H:00} ({ds.sizes['time']} h; peak >= {ACTIVE_MS} m/s plus {MARGIN_H} h margin)")
    ds = ds.load()
    track = storm_track(ds)
    net = load_network() if "network" in VARIANTS else None
    for variant in VARIANTS:
        if variant == "network":
            render(ds, track, mask, key + "_network", land_only=True, net=net)
        else:
            render(ds, track, mask, key)


def render(ds, track, mask, name, land_only=False, net=None):
    """Frames, GIF and HTML viewer for one variant."""
    frames_dir = FIG_DIR / "frames" / name; frames_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.6, 8.4), dpi=110, facecolor="white")
    paths = []
    for i in range(ds.sizes["time"]):
        draw_frame(fig, ds, i, track, mask, land_only=land_only, net=net)
        p = frames_dir / f"frame_{i:03d}.png"; fig.savefig(p, dpi=110, facecolor="white"); paths.append(p)
    plt.close(fig)
    print(f"[ok] {len(paths)} frames in {frames_dir}")

    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128) for p in paths]
    gif = FIG_DIR / f"{name}_windstorm.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"[ok] {gif}  ({len(imgs)} frames at {FPS} fps)")
    write_viewer(FIG_DIR / f"{name}_viewer.html", paths, [str(pd.Timestamp(t))[:16] for t in ds.time.values], FPS)



if __name__ == "__main__":
    main()
