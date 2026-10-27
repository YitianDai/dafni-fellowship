"""
Download ERA5 hourly 10 m wind gust and wind components over Great Britain
for Storm Arwen (Nov 2021) and Storm Eunice (Feb 2022, with Dudley and Franklin
in the same window), then summarise the peak-gust footprint of each storm.

Setup (once):
    pip install cdsapi xarray netCDF4 matplotlib cartopy
    Create ~/.cdsapirc (or %USERPROFILE%\\.cdsapirc on Windows) containing:
        url: https://cds.climate.copernicus.eu/api
        key: <your personal access token from https://cds.climate.copernicus.eu/profile>
    Accept the ERA5 single-levels licence once in the CDS web interface.

Run:
    python download_era5_storms.py            # download + summary
    python download_era5_storms.py --no-plot  # download only
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figs"
DATA.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

# North, West, South, East. Covers GB and surrounding seas with a margin for track shifting.
GB_AREA = [61.0, -11.0, 49.0, 5.0]

STORMS = {
    # Arwen: named 25 Nov, peak impacts overnight 26 to 27 Nov 2021 (NE England, Scotland).
    "arwen_2021": {"year": "2021", "month": "11", "days": [f"{d:02d}" for d in range(25, 29)]},
    # Eunice: 18 Feb 2022. Window also captures Dudley (16 Feb) and Franklin (20 to 21 Feb).
    "eunice_2022": {"year": "2022", "month": "02", "days": [f"{d:02d}" for d in range(16, 22)]},
}

VARIABLES = [
    "instantaneous_10m_wind_gust",   # fg10, hourly
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",       # useful for track identification
    "geopotential",                  # ERA5 orography (z / 9.80665 = cell mean height), for the exposure correction
]
HOURS = [f"{h:02d}:00" for h in range(24)]


def download(name: str, spec: dict, client: cdsapi.Client) -> Path:
    target = DATA / f"era5_{name}.nc"
    if target.exists():
        print(f"[skip] {target.name} already present")
        return target
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [spec["year"]],
        "month": [spec["month"]],
        "day": spec["days"],
        "time": HOURS,
        "area": GB_AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"[cds] requesting {name} ...")
    client.retrieve("reanalysis-era5-single-levels", request, str(target))
    print(f"[ok]  {target}")
    return target


def summarise(path: Path, plot: bool = True) -> None:
    ds = xr.open_dataset(path)
    # Variable names differ between CDS back ends; resolve defensively.
    gust_name = next(v for v in ("fg10", "i10fg", "instantaneous_10m_wind_gust") if v in ds)
    time_name = "valid_time" if "valid_time" in ds.dims else "time"
    gust = ds[gust_name]
    peak = gust.max(time_name)
    peak_ms = float(peak.max())
    loc = peak.where(peak == peak.max(), drop=True)
    print(f"\n{path.stem}: {gust_name} {gust.sizes[time_name]} h, "
          f"grid {gust.sizes['latitude']}x{gust.sizes['longitude']}")
    print(f"  domain peak gust {peak_ms:.1f} m/s ({peak_ms*2.237:.0f} mph) "
          f"at lat {float(loc.latitude[0]):.2f}, lon {float(loc.longitude[0]):.2f}")
    # Hour of domain-wide maximum
    t_idx = int(gust.max(("latitude", "longitude")).argmax())
    print(f"  peak hour: {str(ds[time_name].values[t_idx])[:16]}")
    peak.to_netcdf(DATA / f"{path.stem}_peakgust.nc")

    if not plot:
        return
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(6, 7.5))
        proj = ccrs.TransverseMercator(central_longitude=-4.0, central_latitude=49.0, scale_factor=0.9996)
        ax = plt.axes(projection=proj)
        ax.set_extent([GB_AREA[1], GB_AREA[3], GB_AREA[2], GB_AREA[0]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.6)
        kw = dict(transform=ccrs.PlateCarree())
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 7.5))
        ax.set_aspect(1 / np.cos(np.radians(55))); kw = {}
    # same classes and colours as animate_windstorm.py so every figure reads the same way
    from matplotlib.colors import ListedColormap, BoundaryNorm
    bounds = [0, 10, 15, 20, 25, 30, 35, 40, 45, 60]
    colours = ["#f4f2ee", "#faf3d8", "#fff1b8", "#fed47a", "#fdae3c", "#f07e12", "#d64f05", "#a62a02", "#5e1000"]
    cmap = ListedColormap(colours); norm = BoundaryNorm(bounds, cmap.N)
    im = ax.pcolormesh(peak.longitude, peak.latitude, peak, cmap=cmap, norm=norm, shading="nearest", **kw)
    cb = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=bounds[1:-1], shrink=0.8)
    cb.set_label("Peak 10 m gust over storm window (m/s)")
    ax.set_title(path.stem.replace("era5_", "ERA5 peak gust: "))
    out = FIGS / f"{path.stem}_peakgust.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  figure: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--storms", nargs="*", default=list(STORMS), choices=list(STORMS))
    args = ap.parse_args()
    client = cdsapi.Client()
    for name in args.storms:
        path = download(name, STORMS[name], client)
        summarise(path, plot=not args.no_plot)


if __name__ == "__main__":
    main()
