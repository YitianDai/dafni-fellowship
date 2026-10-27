"""
Entry point used inside the DAFNI containers. It maps DAFNI's conventions onto the repository layout and runs one model:

  parameters   -> environment variables  STORM, SAMPLE, SAMPLES_FIRST, SAMPLES_LAST, N_SAMPLES, SEED, FIGURE_SAMPLE
  input slots  -> /data/inputs/network/<storm>/...   (dataset: GB storm network cases)
                  /data/inputs/era5/...              (dataset: ERA5 storm fields)
                  /data/inputs/results/<storm>/...   (outputs of the previous model in the workflow, optional)
  outputs      -> /data/outputs/<storm>/...

A working root is assembled under /data/work with the repo structure (data/network, data/era5, results) by
symlinking the input slots and pointing results at /data/outputs, then the model script is called with --root.
    python dafni_entry.py wind_damage | storm_cascade | figures
For a local test set DAFNI_DATA to a folder holding inputs/ and run the same command.
"""
import os, sys, shutil, subprocess
MODEL = sys.argv[1]
DATA = os.environ.get("DAFNI_DATA", "/data")
HERE = os.path.dirname(os.path.abspath(__file__))
env = lambda k, d=None: os.environ.get(k, d)
storm = env("STORM", "Arwen2021")

work = os.path.join(DATA, "work"); os.makedirs(os.path.join(work, "data"), exist_ok=True)
out = os.path.join(DATA, "outputs"); os.makedirs(out, exist_ok=True)
for slot in ("network", "era5"):
    dst = os.path.join(work, "data", slot); src = os.path.join(DATA, "inputs", slot)
    if os.path.lexists(dst): os.remove(dst) if os.path.islink(dst) else shutil.rmtree(dst)
    if os.path.isdir(src): os.symlink(src, dst)
prev = os.path.join(DATA, "inputs", "results")
if os.path.isdir(prev):                                   # carry previous models' outputs forward
    shutil.copytree(prev, out, dirs_exist_ok=True)
res = os.path.join(work, "results")
if os.path.lexists(res): os.remove(res) if os.path.islink(res) else shutil.rmtree(res)
os.symlink(out, res)

py = sys.executable
if MODEL == "wind_damage":
    cmd = [py, os.path.join(HERE, "wind_damage", "wind_damage.py"), storm, "--samples", env("N_SAMPLES", "1000"), "--seed", env("SEED", "42"), "--root", work]
elif MODEL == "storm_cascade":
    cmd = [py, os.path.join(HERE, "storm_cascade", "storm_cascade.py"), storm, "--root", work]
    if env("SAMPLES_FIRST") and env("SAMPLES_LAST"): cmd += ["--samples", env("SAMPLES_FIRST"), env("SAMPLES_LAST")]
    elif env("SAMPLE"): cmd += ["--sample", env("SAMPLE")]
elif MODEL == "figures":
    subprocess.run([py, os.path.join(HERE, "figures", "plot_network.py"), storm, "--root", work], check=True)
    cmd = [py, os.path.join(HERE, "figures", "plot_cascade.py"), storm, "--sample", env("FIGURE_SAMPLE", env("SAMPLE", "0")), "--root", work]
else:
    sys.exit(f"unknown model {MODEL}")
print(">", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd).returncode)
