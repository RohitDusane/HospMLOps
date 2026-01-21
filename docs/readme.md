# 🏥 Hospital Readmission Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**An end-to-end ML system for predicting 30-day hospital readmissions using real patient data**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Synthetic Data Analysis](#-synthetic-data-analysis)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The Hospital Readmission Prediction System is a **production-ready machine learning solution** that predicts the likelihood of patient readmission within 30 days of discharge. Built for healthcare providers, the system combines:

- 🤖 **4 State-of-the-art ML models** (Logistic Regression, Random Forest, LightGBM, XGBoost)
- ⚡ **FastAPI backend** for high-performance inference
- 💻 **React frontend** for intuitive user experience
- 📊 **Comprehensive evaluation tools** with confusion matrices and ROC curves
- 🎨 **Interactive visualizations** for model interpretability

### Why This Matters

Hospital readmissions cost the US healthcare system **$17 billion annually**. Early identification of high-risk patients enables:
- Proactive interventions
- Reduced healthcare costs
- Improved patient outcomes
- Optimized resource allocation

---

## ✨ Features

### Core Functionality
- ✅ **Single Patient Prediction**: Real-time risk assessment with 18 clinical features
- ✅ **Batch Processing**: CSV upload for bulk predictions
- ✅ **Model Evaluation**: Upload test data to evaluate model performance
- ✅ **Feature Importance**: Understand which factors drive predictions
- ✅ **Risk Stratification**: Automatic classification (Low/Medium/High risk)

### Technical Features
- 🚀 **RESTful API**: FastAPI with automatic OpenAPI documentation
- 🔒 **Input Validation**: Pydantic schemas ensure data integrity
- 📈 **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 🎨 **Interactive Dashboards**: Real-time visualizations
- 🔧 **Easy Deployment**: Docker-ready with minimal dependencies

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  • Single Prediction Form  • CSV Upload                 │
│  • Model Evaluation        • Performance Visualizations  │
└─────────────────────────────────────────────────────────┘
                          ↕ REST API
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                         │
│  • Request Validation   • Data Preprocessing            │
│  • Model Management     • Performance Evaluation        │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│              ML Models (Trained on Real Data)            │
│  • Logistic Regression  • Random Forest (Primary)       │
│  • LightGBM            • XGBoost                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip
- Modern web browser (Chrome, Firefox, Safari)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hospital-readmission-prediction.git
cd hospital-readmission-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### Frontend Setup

Simply open `index.html` in your browser or serve it with:

```bash
# Using Python's built-in server
python -m http.server 3000

# Then open: http://localhost:3000
```

---

## 📖 Usage

### 1. Single Patient Prediction

Navigate to the "Single Prediction" tab and enter patient information:

```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[50-60)",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 3,
  "num_lab_procedures": 44,
  "num_procedures": 1,
  "num_medications": 14,
  "number_diagnoses": 8,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 0,
  "A1Cresult": "Not Available",
  "insulin": "Steady",
  "change": "Ch",
  "diabetesMed": "Yes"
}
```

**Expected Output**:
- Prediction: Readmission Likely (1) or Not Likely (0)
- Probability: 0.0 to 1.0
- Risk Level: Low, Medium, or High
- Top contributing features

### 2. Batch Prediction via CSV

Upload a CSV file with the required 18 features:

```csv
race,gender,age,admission_type_id,...,diabetesMed
Caucasian,Female,[50-60),1,...,Yes
AfricanAmerican,Male,[70-80),2,...,No
```

Get results for all patients instantly with summary statistics.

### 3. Model Evaluation

Upload a test dataset (with `readmitted_bin` column) to get:
- Confusion Matrix
- ROC Curve
- Feature Importance Plot
- Detailed Classification Report

### 4. API Usage

#### Using cURL

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST "http://127.0.0.1:8000/predict?model_key=real_to_real_random_forest" \
  -H "Content-Type: application/json" \
  -d @patient_data.json

# Batch prediction
curl -X POST "http://127.0.0.1:8000/predict/csv?model_key=real_to_real_random_forest" \
  -F "file=@patients.csv"
```

#### Using Python

```python
import requests

# Single prediction
url = "http://127.0.0.1:8000/predict"
params = {"model_key": "real_to_real_random_forest"}
data = {
    "race": "Caucasian",
    "gender": "Female",
    # ... other features
}

response = requests.post(url, params=params, json=data)
print(response.json())
```

---

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with health info |
| `/health` | GET | Detailed health check |
| `/models` | GET | List available models |
| `/features` | GET | List required features |
| `/predict` | POST | Single patient prediction |
| `/predict/csv` | POST | Batch CSV prediction |
| `/evaluate/model` | POST | Model performance evaluation |
| `/model/info/{model_key}` | GET | Model metadata |

### Interactive Documentation

Once the server is running, access:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 📊 Model Performance

### Real-to-Real Models (Production)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **92.0%** | **90.4%** | **94.0%** | **92.2%** | **0.892** |
| XGBoost | 91.5% | 89.8% | 93.5% | 91.6% | 0.885 |
| LightGBM | 91.2% | 89.2% | 93.2% | 91.1% | 0.878 |
| Logistic Regression | 87.5% | 85.3% | 89.7% | 87.4% | 0.842 |

**Primary Model**: Random Forest (Best overall performance)

### Feature Importance (Top 5)

1. **number_inpatient** (14.5%) - Previous inpatient visits
2. **time_in_hospital** (13.2%) - Length of stay
3. **num_medications** (9.8%) - Number of medications
4. **number_emergency** (8.7%) - Emergency visits
5. **discharge_disposition_id** (7.9%) - Discharge type

---

## 📁 Project Structure

```
hospital-readmission-prediction/
│
├── app/
│   ├── main.py                 # FastAPI application
│   ├── artifacts/
│   │   ├── models/
│   │   │   └── real_to_real/
│   │   │       ├── logistic_regression.pkl
│   │   │       ├── random_forest.pkl
│   │   │       ├── lightgbm.pkl
│   │   │       └── xgboost.pkl
│   │   └── transformers/
│   │       └── real_transformers.pkl
│   └── __init__.py
│
├── frontend/
│   └── index.html              # React frontend
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_synthetic_data_generation.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned data
│   └── synthetic/              # Synthetic data (not used in production)
│
├── tests/
│   ├── test_api.py
│   ├── test_models.py
│   └── test_preprocessing.py
│
├── docs/
│   ├── HLD.md                  # High Level Design
│   ├── SLD.md                  # System Level Design
│   ├── DIAGNOSTIC_REPORT.md    # Synthetic data analysis
│   └── API_GUIDE.md
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧪 Synthetic Data Analysis

### Motivation

We explored **synthetic data generation** using CTGAN (Conditional Tabular GAN) to:
1. Address potential data scarcity
2. Explore privacy-preserving alternatives
3. Investigate data augmentation possibilities

### Key Findings

#### ✅ Marginal Success: Individual Features

**Statistical Tests Passed**: 18 out of 21 features (85.7%)

```
Features with p > 0.05 (after FDR correction):
✓ time_in_hospital (p=0.271)
✓ num_medications (p=0.264)
✓ num_procedures (p=0.316)
✓ number_emergency (p=0.267)
✓ number_inpatient (p=0.710)
✓ gender (p=0.625)
✓ race (p=0.401)
✓ insulin (p=0.497)
... and 10 more features
```

**Interpretation**: CTGAN successfully preserved individual feature distributions.

#### ❌ Critical Failure: Target Variable

**Most Important Finding**:

| Feature | Test | p-value | FDR-adjusted | Status |
|---------|------|---------|--------------|--------|
| **readmitted** | Chi-square | **<0.001** | **<0.001** | ❌ **FAILED** |

**Distribution Mismatch**:
```
Real Data:
  No Readmission (0):  55.2%
  Readmission (1):     44.8%

Synthetic Data:
  No Readmission (0):  52.1%  (Δ -3.1%)
  Readmission (1):     47.9%  (Δ +3.1%)

Chi-square: p < 0.001 ❌
```

**Why This Matters**: The target variable is what we predict. A mismatch here means models learn **wrong patterns**.

#### ❌ Secondary Issues

**Age Distribution** (p = 0.028):
- Marginally significant before FDR correction
- Age is critical for readmission prediction
- Even small mismatches impact model performance

**Patient IDs** (p = 0.020):
- Systematic differences suggest generation artifacts
- Indicates underlying distribution issues

#### ❌ Model Performance Degradation

| Training → Testing | Accuracy | ROC-AUC | ΔAccuracy | Status |
|-------------------|----------|---------|-----------|--------|
| Real → Real | **92.0%** | **0.892** | - | ✅ Production |
| Synthetic → Synthetic | 88.5% | 0.845 | -3.5% | ⚠️ |
| **Synthetic → Real** | **73.2%** | **0.680** | **-18.8%** | ❌ **CRITICAL** |

**Critical Issue**: Models trained on synthetic data **catastrophically fail** on real patients.

#### ❌ The Deceptive Paradox

**The Problem**: 85.7% of features pass statistical tests, but models fail terribly.

**Why**:
```
Statistical Tests Measure:
  ✓ P(feature) - Individual distributions
  
Models Actually Need:
  ✗ P(target | features) - Conditional probabilities
  ✗ Feature interactions and correlations
  ✗ Joint distributions
```

**Example of Broken Relationships**:
```
Real Data Pattern:
  number_inpatient ≥ 2 → 68% readmission rate
  
Synthetic Data Pattern:
  number_inpatient ≥ 2 → 61% readmission rate (Δ -7%)
  
Result: Wrong predictions for high-risk patients
```

### Diagnostic Report

Full diagnostic analysis with statistical tests, visualizations, and detailed root cause analysis available at [`docs/DIAGNOSTIC_REPORT.md`](docs/DIAGNOSTIC_REPORT.md).

**Key sections include**:
- Complete statistical test results (21 features)
- Target variable mismatch analysis
- Feature correlation degradation
- Model performance across scenarios
- Q-Q plots and distribution comparisons
- Clinical impact assessment
- Technical deep-dive into CTGAN limitations

### Why Synthetic Data Failed

**Root Causes**:

1. **Target Variable Mismatch** (p < 0.001)
   - Generator couldn't preserve exact readmission distribution
   - 3.1% oversampling creates systematic bias
   - Models calibrated to wrong baseline

2. **Joint Distribution Corruption**
   - Individual features match ✓
   - Feature relationships broken ❌
   - Correlation loss: ~25% average
   - Example: `time_in_hospital ↔ num_medications` dropped from r=0.67 to r=0.51

3. **Conditional Probability Failure**
   - P(readmitted | features) learned incorrectly
   - High-risk patient profiles not preserved
   - Clinical relationships lost

**Technical Issues**:
- CTGAN mode collapse on target variable
- Weak conditional generation
- Focus on "looking real" vs "being predictive"

### Clinical Impact

**If Synthetic Data Were Used** (per 10,000 predictions):

```
Additional Errors: +1,880
  • False Negatives: +1,180 (missed high-risk patients)
  • False Positives: +700 (unnecessary interventions)

Clinical Consequences:
  • ~12% more preventable readmissions
  • ~7% wasted intervention resources  
  • Patient safety risk
  • Loss of clinician trust

Estimated Cost: $17.7M additional per 10,000 predictions
```

### Lessons Learned

1. **Statistical Similarity ≠ Predictive Power**
   - 85.7% feature match means nothing if target variable fails
   - Standard statistical tests are insufficient for ML validation
   - Must test model performance, not just distributions

2. **Target Variable Is Sacred**
   - Even 3% mismatch causes 18.8% accuracy drop
   - Models extremely sensitive to outcome variable bias
   - Required: p > 0.5 for target variable

3. **CTGAN's Fundamental Limitation**
   - Good at: Marginal distributions
   - Bad at: Joint distributions, conditional relationships
   - Fatal flaw: Optimizes "fooling discriminator" not "predictive accuracy"

4. **Better Alternatives Exist**
   - Copula-based synthesis (preserves correlations)
   - TVAE (better likelihood optimization)
   - Federated learning (keeps real data distributed)
   - Differential privacy (adds noise to real data)

### Conclusion

**Synthetic data was NOT used in production** due to:
1. ❌ Target variable distribution failure (p < 0.001)
2. ❌ Unacceptable performance degradation (-18.8% accuracy)
3. ❌ Loss of clinical feature relationships
4. ❌ Patient safety risk (1,180 missed high-risk patients per 10,000)
5. ❌ Deceptive statistical tests (85.7% pass rate masked critical failures)

**Production Decision**: Use 100% real data for reliability, accuracy, and patient safety.

**The Paradox**: CTGAN succeeded at making data look statistically similar but failed at making it predictively equivalent. This highlights a crucial insight: **for machine learning applications, predictive fidelity matters more than statistical similarity**.

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run integration tests
pytest tests/integration/ -v

# Load testing
locust -f tests/load_test.py
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t hospital-readmission-api .

# Run container
docker run -d -p 8000:8000 hospital-readmission-api

# Using docker-compose
docker-compose up -d
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_DIR=app/artifacts/models
TRANSFORMER_DIR=app/artifacts/transformers
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## 📈 Monitoring
### FASTApi App Running 
'''bash
# Start FastAPI server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
'''
### REACT APP 
'''
# Windows CMD
start /B python -m http.server 3000 && timeout /t 2 && start http://127.0.0.1:3000/react.html
'''
### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Metrics to Monitor

- Request count per endpoint
- Response time (p50, p95, p99)
- Error rate
- Prediction distribution (Low/Medium/High)
- Model usage statistics

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black app/
isort app/

# Run linting
flake8 app/
pylint app/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

- **Data Science Lead**: [Your Name]
- **ML Engineer**: [Name]
- **Backend Developer**: [Name]
- **Frontend Developer**: [Name]

---

## 📞 Contact

- **Project Lead**: [your.email@example.com]
- **Issues**: [GitHub Issues](https://github.com/yourusername/hospital-readmission-prediction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hospital-readmission-prediction/discussions)

---

## 🙏 Acknowledgments

- Dataset: [Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Inspiration: Healthcare cost reduction initiatives
- Libraries: scikit-learn, FastAPI, React, LightGBM, XGBoost

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{hospital_readmission_prediction,
  title = {Hospital Readmission Prediction System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/hospital-readmission-prediction}
}
```

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ for better healthcare outcomes

</div>