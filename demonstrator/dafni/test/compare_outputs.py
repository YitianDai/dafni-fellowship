"""Checks that a container run reproduces the reference results committed in results/<storm>/.
    python compare_outputs.py <storm> <sample> <container_outputs_dir> <reference_results_dir>
Wind damage: same set of (sample, hour, branch) failures. Cascade: same hourly table (numeric columns to 1e-6).
"""
import sys, os, pandas as pd, numpy as np
storm, k, out, ref = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out, ref = os.path.join(out, storm), os.path.join(ref, storm)

a = pd.read_csv(os.path.join(out, "wind_damage.csv")); b = pd.read_csv(os.path.join(ref, "wind_damage.csv"))
ka = set(map(tuple, a[["sample", "h", "branch_idx"]].values)); kb = set(map(tuple, b[["sample", "h", "branch_idx"]].values))
assert ka == kb, f"wind_damage differs: {len(ka - kb)} extra, {len(kb - ka)} missing"
print(f"wind_damage.csv: {len(ka)} failures identical")

a = pd.read_csv(os.path.join(out, f"cascade_sample{k}.csv")); b = pd.read_csv(os.path.join(ref, f"cascade_sample{k}.csv"))
assert list(a.columns) == list(b.columns) and len(a) == len(b), "cascade table shape differs"
cols = ["h", "wind_failures", "cascade_trips", "cascade_branches", "new_dead_buses", "islands", "cascade_iterations", "disconnected_MW"]
diff = (a[cols].fillna("").astype(str) != b[cols].fillna("").astype(str)).any(axis=1)
if diff.any():
    print("rows that differ (container):"); print(a.loc[diff, cols].to_string())
    print("rows that differ (reference):"); print(b.loc[diff, cols].to_string())
for c in a.columns:
    if c == "sec": continue
    if pd.api.types.is_numeric_dtype(a[c]):
        assert np.allclose(a[c].fillna(-1), b[c].fillna(-1), rtol=1e-6, atol=1e-6), f"column {c} differs"
    else:
        assert (a[c].fillna("").astype(str) == b[c].fillna("").astype(str)).all(), f"column {c} differs"
ens = (a["disconnected_MW"]).sum()
print(f"cascade_sample{k}.csv identical: {int(a['wind_failures'].sum())} wind failures, {int(a['cascade_trips'].sum())} cascade trips, {ens:.0f} MWh not served")

figs = os.path.join(out, "figs")
need = [f"{storm}_network.png", f"{storm}_cascade.gif", f"{storm}_cascade_viewer.html", f"{storm}_cascade_stages.png"]
missing = [f for f in need if not os.path.exists(os.path.join(figs, f))]
assert not missing, f"missing figures: {missing}"
print("figures present:", ", ".join(need), f"+ {len(os.listdir(os.path.join(figs, storm + '_cascade_frames')))} frames")
