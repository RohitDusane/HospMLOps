# E2E MLOPS HOSPITAL READMISSION PREDICTION
End-to-end MLOps pipeline for hospital readmission prediction.

## Project Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/HospMLOps.git
cd HospMLOps
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

* **macOS/Linux**

```bash
source venv/bin/activate
```

* **Windows**

```bash
venv\Scripts\activate
```

### 4. Install the project in editable mode

```bash
pip install -e .
```

### 5. (Optional) Install development dependencies

```bash
pip install -e .[dev]
```

This will install additional packages like `pytest`, `black`, and other dev tools.

### 6. Verify installation

```bash
python -c "import components.data_ingestion; print('Components module imported successfully')"
```

### 7. Run the CLI (if defined)

```bash
hospmlops-cli
```

---

### 8. Project Structure

```
HospMLOps/
├── src/                   # Main Python package (src-layout)
│   ├── config/            # Configuration loader
│   ├── components/        # Data ingestion, validation, transformation, feature engineering, model training
│   ├── pipelines/         # Staged pipeline scripts
│   └── core/              # Logging, exceptions, utils
├── configs/               # YAML configuration files
├── app/                   # FastAPI app and static UI
├── scripts/               # Training, deployment scripts
├── docker/                # Docker and docker-compose
├── k8s/                   # Kubernetes manifests
├── artifacts/             # Generated artifacts (raw, processed, transformed, models, logs)
├── tests/                 # Unit and integration tests
├── requirements.txt       # Core dependencies
├── requirements-dev.txt   # Dev dependencies
├── setup.py               # Package installer
├── MANIFEST.in            # Include YAMLs and static files
└── README.md

```

## 6️⃣ Configure Environment Variables (Optional)

Create a `.env` file at the project root:

```env
ENV_MODE=development
DEBUG_LOGS=True
API_HOST=0.0.0.0
API_PORT=8000
```

* These override values in `config.yaml` for local development, staging, or production.

---

## 7️⃣ Running the Pipeline

All pipeline stages are modular and can be run independently or sequentially.

### Run Full Training Pipeline

```bash
python scripts/train_pipeline.py
```

This will:

* Ingest raw data (`artifacts/raw/`)
* Validate data against schema (`artifacts/validated/`)
* Transform and feature-engineer data (`artifacts/transformed/`)
* Train model (`artifacts/models/`)
* Save reports and metrics (`artifacts/reports/`)

### Run Individual Stages

```bash
# Stage 1: Data Ingestion
python src/pipelines/stage_01_data_ingestion.py

# Stage 5: Model Training
python src/pipelines/stage_05_model_training.py
```

---

## 8️⃣ MLflow Tracking

* Default MLflow tracking URI: `file:./mlruns`
* Experiment names are defined in `config.yaml` under `data_preprocessing.mlflow` and `validation.mlflow`.

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

* Visit `http://localhost:5000` to monitor experiments.

---

## 9️⃣ Running the FastAPI App

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* UI is accessible at `http://localhost:8000`
* Predictions can be sent via API or the web UI (`app/static/index.html`).

---

## 🔟 Testing

* Run **unit tests**:

```bash
pytest tests/unit
```

* Run **integration tests**:

```bash
pytest tests/integration
```

* Test coverage:

```bash
pytest --cov=src tests/
```

---

## 1️⃣1️⃣ Deployment

### Using Docker

```bash
# Build production image
docker build -t hospmlops:latest -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 hospmlops:latest
```

### Using Docker Compose (Local Dev)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Apply service (LoadBalancer)
kubectl apply -f k8s/service.yaml

# Apply HPA (Horizontal Pod Autoscaler)
kubectl apply -f k8s/hpa.yaml
```

* ConfigMaps are injected from `k8s/configmap.yaml`

---

## 1️⃣2️⃣ Monitoring and Alerts

* Data drift is monitored via PSI (`drift_monitoring`)
* Fairness metrics are calculated (`fairness`)
* Alerts are sent to `mlops@hospital.org` if thresholds are violated

---

## 1️⃣3️⃣ Notes & Best Practices

1. Use **editable install (`pip install -e .`)** for development.
2. Keep **configs in YAML** and override with environment variables for staging/prod.
3. Logs are written to `logs/`, artifacts to `artifacts/`.
4. Use MLflow for experiment tracking and reproducibility.
5. Follow `src/` package structure for clean MLOps pipelines.

---

