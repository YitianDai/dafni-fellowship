"""
Workflow 1 (Case 1, storm replay): wind damage -> hourly cascade -> figures, for one storm and one sample.
    python workflows/run_case.py Arwen2021 --sample 330
Workflow 2 (ensemble): the same chain over a range of samples, then the ensemble figures.
    python workflows/run_case.py Arwen2021 --samples 0 99 --figure-sample 330
Each step is a model in models/; this driver only calls them in order.
"""
import os, sys, subprocess, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = lambda *p: os.path.join(ROOT, "models", *p)

def run(cmd):
    print(">", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("storm", choices=["Arwen2021", "Eunice2022"])
    ap.add_argument("--sample", type=int, default=None, help="single wind-damage sample to run and draw")
    ap.add_argument("--samples", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None, help="run samples FIRST..LAST (ensemble)")
    ap.add_argument("--figure-sample", type=int, default=None, help="sample to draw the frames for (ensemble mode)")
    ap.add_argument("--n-wind-samples", type=int, default=1000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-wind", action="store_true", help="reuse results/<storm>/wind_damage.csv")
    a = ap.parse_args(); py = sys.executable
    if not a.skip_wind:
        run([py, M("wind_damage", "wind_damage.py"), a.storm, "--samples", str(a.n_wind_samples), "--seed", str(a.seed), "--root", ROOT])
    if a.samples:
        run([py, M("storm_cascade", "storm_cascade.py"), a.storm, "--samples", str(a.samples[0]), str(a.samples[1]), "--root", ROOT])
        fs = a.figure_sample if a.figure_sample is not None else a.samples[0]
    else:
        cmd = [py, M("storm_cascade", "storm_cascade.py"), a.storm, "--root", ROOT]
        if a.sample is not None: cmd += ["--sample", str(a.sample)]
        run(cmd); fs = a.sample
    if fs is None:
        import glob; fs = int(sorted(glob.glob(os.path.join(ROOT, "results", a.storm, "cascade_sample*.csv")))[-1].split("cascade_sample")[1].split(".")[0].split("_")[0])
    run([py, M("figures", "plot_network.py"), a.storm, "--root", ROOT])
    run([py, M("figures", "plot_cascade.py"), a.storm, "--sample", str(fs), "--root", ROOT])
