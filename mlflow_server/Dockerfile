FROM python:3.10-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/secrets/mlops-key.json

WORKDIR /app

# Install MLflow, PostgreSQL driver, GCS support
RUN pip install --upgrade pip && \
    pip install mlflow[gcp] psycopg2-binary gunicorn google-cloud-storage

EXPOSE 8080

# Start MLflow server
CMD ["mlflow", "server",
     "--host", "0.0.0.0",
     "--port", "8080",
     "--backend-store-uri", "postgresql+psycopg2://mlflow-user:Stat%401234@/mlflow?host=/cloudsql/hospital-readmission-ml:us-central1:mlflow-db",
     "--default-artifact-root", "gs://hospital-readmission-ml-mlflow-artifacts"
]
