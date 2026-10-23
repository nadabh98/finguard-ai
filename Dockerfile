FROM python:3.12-slim
# libgomp1 est nécessaire à LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY models/lightgbm.joblib models/calibrator.joblib models/
ENV PYTHONPATH=/app/src PORT=8080
CMD uvicorn finguard.api:app --host 0.0.0.0 --port ${PORT}
