# DAFNI model image for wind_damage. Built by GitHub Actions (.github/workflows/dafni-images.yml) from the demonstrator/ folder:
#   docker build -f dafni/wind_damage/Dockerfile -t dafni-wind_damage .
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models /app/models
# DAFNI mounts dataslots under /data/inputs and collects everything written to /data/outputs
ENTRYPOINT ["python", "/app/models/dafni_entry.py", "wind_damage"]
