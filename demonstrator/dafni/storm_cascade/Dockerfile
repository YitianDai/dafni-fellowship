# DAFNI model image for storm_cascade. Built by GitHub Actions (.github/workflows/dafni-images.yml) from the demonstrator/ folder:
#   docker build -f dafni/storm_cascade/Dockerfile -t dafni-storm_cascade .
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models /app/models
# DAFNI mounts dataslots under /data/inputs and collects everything written to /data/outputs
ENTRYPOINT ["python", "/app/models/dafni_entry.py", "storm_cascade"]
