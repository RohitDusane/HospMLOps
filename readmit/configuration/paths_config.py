# readmit/configuration/paths_config.py

import os
from pathlib import Path

# =========================
# BASE DIRECTORIES
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
# Configuration YAML path (points to src/CreditRisk/config/configuration.yaml)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "readmit", "configuration","config.yaml")

# =========================
# RAW DATA PATHS
# =========================
RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")
RAW_REAL_DATA_PATH = os.path.join(RAW_DIR, "diabetic_data.csv")
RAW_SYN_DATA_PATH = os.path.join(RAW_DIR, "synthetic_diabetic_tvae2.csv")

# =========================
# PROCESSED DATA PATHS (After Ingestion)
# =========================
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")

PROCESSED_REAL_DIR = os.path.join(PROCESSED_DIR, "real")
PROCESSED_REAL_TRAIN = os.path.join(PROCESSED_REAL_DIR, "train.csv")
PROCESSED_REAL_TEST = os.path.join(PROCESSED_REAL_DIR, "test.csv")

PROCESSED_SYN_DIR = os.path.join(PROCESSED_DIR, "synthetic")
PROCESSED_SYN_TRAIN = os.path.join(PROCESSED_SYN_DIR, "train.csv")
PROCESSED_SYN_TEST = os.path.join(PROCESSED_SYN_DIR, "test.csv")

PROCESSED_SYN_PRO = os.path.join(PROCESSED_DIR, 'processed')
PROCESSED_SYN_PRO_TRAIN = os.path.join(PROCESSED_SYN_PRO, "train.csv")
PROCESSED_SYN_PRO_TEST = os.path.join(PROCESSED_SYN_PRO, "test.csv")

# =========================
# VALIDATED DATA PATHS (After Validation)
# =========================
VALIDATED_DIR = os.path.join(ARTIFACTS_DIR, "validated")

VALIDATED_REAL_DIR = os.path.join(VALIDATED_DIR, "real")
VALIDATED_REAL_TRAIN = os.path.join(VALIDATED_REAL_DIR, "train.csv")
VALIDATED_REAL_TEST = os.path.join(VALIDATED_REAL_DIR, "test.csv")

VALIDATED_SYN_DIR = os.path.join(VALIDATED_DIR, "synthetic")
VALIDATED_SYN_TRAIN = os.path.join(VALIDATED_SYN_DIR, "train.csv")
VALIDATED_SYN_TEST = os.path.join(VALIDATED_SYN_DIR, "test.csv")

# =========================
# EDA DATA PATHS (After EDA)
# =========================
EDA_DIR = os.path.join(ARTIFACTS_DIR, "eda")
EDA_REAL_DIR = os.path.join(EDA_DIR, "real")
EDA_SYN_DIR = os.path.join(EDA_DIR, "synthetic")

# =========================
# TRANSFORMED DATA PATHS (After Transformation)
# =========================
TRANSFORMED_DIR = os.path.join(ARTIFACTS_DIR, "transformed")

TRANSFORMED_REAL_DIR = os.path.join(TRANSFORMED_DIR, "real")
TRANSFORMED_REAL_TRAIN = os.path.join(TRANSFORMED_REAL_DIR, "train.csv")
TRANSFORMED_REAL_TEST = os.path.join(TRANSFORMED_REAL_DIR, "test.csv")

TRANSFORMED_SYN_DIR = os.path.join(TRANSFORMED_DIR, "synthetic")
TRANSFORMED_SYN_TRAIN = os.path.join(TRANSFORMED_SYN_DIR, "train.csv")
TRANSFORMED_SYN_TEST = os.path.join(TRANSFORMED_SYN_DIR, "test.csv")

# =========================
# MODEL PATHS
# =========================
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROCESSOR_PATH = os.path.join(TRANSFORMED_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# =========================
# REPORTS PATHS
# =========================
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
VALIDATION_REPORT = os.path.join(REPORTS_DIR, "validation_report.json")

# Publication reports DIR
PUB_DIR = os.path.join(ARTIFACTS_DIR, "publication")
TABLE_DIR = os.path.join(PUB_DIR, "tables")
FIG_DIR = os.path.join(PUB_DIR, "figures")
JSON_DIR = os.path.join(PUB_DIR, "json")

for d in [TABLE_DIR, FIG_DIR, JSON_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================
# MODEL / REPORTS (Optional)
# =========================
# MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
# REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
