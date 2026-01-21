# High Level Design (HLD)
## Hospital Readmission Prediction System Using Machine Learning

---

## 1. Executive Summary

### 1.1 Project Overview
The Hospital Readmission Prediction System is an end-to-end machine learning solution designed to predict 30-day hospital readmissions for diabetic patients. The system leverages multiple ML algorithms including Logistic Regression, Random Forest, LightGBM, and XGBoost to provide accurate risk assessments.

### 1.2 Business Objectives
- **Primary Goal**: Predict likelihood of patient readmission within 30 days
- **Secondary Goals**: 
  - Identify high-risk patients for proactive intervention
  - Reduce healthcare costs by preventing unnecessary readmissions
  - Optimize resource allocation for patient follow-up care
  - Provide interpretable predictions for clinical decision support

### 1.3 Key Stakeholders
- **Healthcare Providers**: Hospitals, clinics, physicians
- **Healthcare Administrators**: Resource planning teams
- **Data Scientists**: Model development and maintenance
- **Patients**: Indirect beneficiaries through improved care

---

## 2. System Architecture

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          React Web Application (Frontend)                 │  │
│  │  - Single Patient Prediction                             │  │
│  │  - Batch CSV Upload                                       │  │
│  │  - Model Evaluation Dashboard                             │  │
│  │  - Performance Visualizations                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/REST API
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI Backend Server                       │  │
│  │  - RESTful API Endpoints                                  │  │
│  │  - Request Validation (Pydantic)                          │  │
│  │  - Model Manager Service                                  │  │
│  │  - Data Preprocessing Pipeline                            │  │
│  │  - Performance Metrics Calculator                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                         ML MODEL LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Pre-trained ML Models                        │  │
│  │  Real-to-Real Models:                                     │  │
│  │    • Logistic Regression                                  │  │
│  │    • Random Forest (Primary)                              │  │
│  │    • LightGBM                                             │  │
│  │    • XGBoost                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Artifacts & Transformers                     │  │
│  │  - Trained Model Files (.pkl)                             │  │
│  │  - Label Encoders                                         │  │
│  │  - Feature Scalers (StandardScaler)                       │  │
│  │  - Imputers (SimpleImputer)                               │  │
│  │  - Feature Lists & Metadata                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Description

#### 2.2.1 Frontend Layer (React)
- **Technology**: React 18, TailwindCSS
- **Components**:
  - Single Prediction Form (18 input features)
  - Batch CSV Upload Interface
  - Model Evaluation Dashboard
  - Real-time API Health Monitoring
  - Interactive Visualizations (Confusion Matrix, ROC Curves)

#### 2.2.2 Backend Layer (FastAPI)
- **Technology**: Python 3.8+, FastAPI, Uvicorn
- **Core Services**:
  - **Model Manager**: Loads and manages 4 ML models
  - **Preprocessing Pipeline**: Handles feature engineering
  - **Prediction Service**: Generates risk predictions
  - **Evaluation Service**: Calculates performance metrics
  - **Visualization Service**: Creates diagnostic plots

#### 2.2.3 ML Model Layer
- **Models Deployed**: Real-to-Real training scenario
  - Logistic Regression (Baseline)
  - Random Forest (Best Performance)
  - LightGBM (Fast Inference)
  - XGBoost (High Accuracy)

#### 2.2.4 Data Layer
- **Storage**: File-based artifact storage
- **Transformers**: Serialized preprocessing objects
- **Format**: Pickle (.pkl) files

---

## 3. Data Flow Architecture

### 3.1 Prediction Flow

```
User Input → Validation → Preprocessing → Model Inference → 
Risk Calculation → Feature Importance → Response → Visualization
```

### 3.2 Detailed Data Flow

1. **Input Reception**
   - User submits patient data via web form or CSV upload
   - FastAPI receives JSON payload with 18 features
   - Pydantic validates data types and constraints

2. **Preprocessing Pipeline**
   ```
   Raw Features → Label Encoding → Missing Value Imputation → 
   Feature Scaling → Feature Ordering → Model-Ready Data
   ```

3. **Model Inference**
   - Preprocessed data fed to selected model
   - Model generates binary prediction (0/1)
   - Probability scores calculated
   - Risk level categorized (Low/Medium/High)

4. **Feature Importance Extraction**
   - For tree-based models (RF, LightGBM, XGBoost)
   - Top 10 contributing features identified
   - Importance scores normalized

5. **Response Generation**
   - Structured JSON response created
   - Includes: prediction, probability, risk_level, feature_importance
   - Timestamp added for audit trail

---

## 4. Feature Engineering

### 4.1 Input Features (18 Total)

| Category | Features | Type | Description |
|----------|----------|------|-------------|
| **Demographics** | race, gender, age | Categorical | Patient demographics |
| **Admission** | admission_type_id, discharge_disposition_id, admission_source_id | Numerical (Categorical) | Admission context |
| **Clinical** | time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_diagnoses | Numerical | Clinical metrics |
| **Visit History** | number_outpatient, number_emergency, number_inpatient | Numerical | Previous visits |
| **Medical Tests** | A1Cresult, insulin, change, diabetesMed | Categorical | Test results & medications |

### 4.2 Preprocessing Steps

1. **Categorical Encoding**
   - Label Encoding for ordinal features
   - Handles unseen categories with "Missing" class

2. **Missing Value Imputation**
   - Numerical: Mean imputation (SimpleImputer)
   - Categorical: Most frequent value

3. **Feature Scaling**
   - StandardScaler for numerical features
   - Z-score normalization

4. **Feature Ordering**
   - Alphabetical ordering for consistency
   - Ensures model compatibility

---

## 5. Model Architecture

### 5.1 Model Selection Criteria

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **Logistic Regression** | Interpretable, Fast | Baseline comparison |
| **Random Forest** | Robust, Handles non-linearity | Primary production model |
| **LightGBM** | Fast training, Memory efficient | Large-scale deployments |
| **XGBoost** | High accuracy, Regularization | Critical predictions |

### 5.2 Model Training Strategy

**Training Scenarios Explored:**
1. ✅ **Real-to-Real**: Train on real data → Test on real data (DEPLOYED)
2. ❌ **Synthetic-to-Synthetic**: Train on synthetic → Test on synthetic (NOT DEPLOYED)
3. ❌ **Synthetic-to-Real**: Train on synthetic → Test on real (FAILED - Distribution Mismatch)

**Selected Strategy**: Real-to-Real (Best Performance)

### 5.3 Model Performance Metrics

- **Primary Metric**: ROC-AUC Score
- **Secondary Metrics**: 
  - Accuracy
  - Precision (minimize false positives)
  - Recall (minimize false negatives)
  - F1-Score (balanced metric)

---

## 6. API Design

### 6.1 Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | API health check |
| `/models` | GET | List available models |
| `/features` | GET | List required features |
| `/predict` | POST | Single patient prediction |
| `/predict/csv` | POST | Batch predictions |
| `/evaluate/model` | POST | Model performance evaluation |
| `/model/info/{model_key}` | GET | Model metadata |

### 6.2 Request/Response Schema

**Prediction Request:**
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

**Prediction Response:**
```json
{
  "prediction": 1,
  "probability": 0.723,
  "risk_level": "High",
  "model_used": "real_to_real_random_forest",
  "timestamp": "2025-01-12T10:30:00",
  "feature_importance": {
    "number_inpatient": 0.145,
    "time_in_hospital": 0.132,
    "num_medications": 0.098
  }
}
```

---

## 7. Deployment Architecture

### 7.1 Current Deployment

- **Environment**: Local Development
- **Backend**: Uvicorn ASGI Server (Port 8000)
- **Frontend**: Static HTML (Browser-based)
- **CORS**: Enabled for local development

### 7.2 Production Deployment Recommendations

```
┌─────────────────────────────────────────────────────┐
│                 Load Balancer (Nginx)                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              Frontend (Static Hosting)               │
│         AWS S3 + CloudFront / Netlify / Vercel      │
└─────────────────────────────────────────────────────┘
                        ↓ HTTPS
┌─────────────────────────────────────────────────────┐
│              API Gateway (AWS/Azure)                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│           FastAPI Backend (Containerized)            │
│         Docker + Kubernetes / AWS ECS                │
│         Multiple instances for load balancing        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              Model Storage (S3/Azure Blob)           │
│         Versioned model artifacts                    │
└─────────────────────────────────────────────────────┘
```

---

## 8. Security & Compliance

### 8.1 Data Security
- **Encryption**: HTTPS/TLS for data in transit
- **Authentication**: API key-based authentication (future)
- **Authorization**: Role-based access control (RBAC)
- **Data Privacy**: No PHI storage, predictions are stateless

### 8.2 Compliance Considerations
- **HIPAA**: Ensure PHI is properly de-identified
- **GDPR**: Data processing agreements required
- **Audit Trails**: Timestamped prediction logs

---

## 9. Monitoring & Maintenance

### 9.1 System Monitoring
- API health checks (`/health` endpoint)
- Response time monitoring
- Error rate tracking
- Model drift detection (future)

### 9.2 Model Maintenance
- **Retraining Schedule**: Quarterly or when performance degrades
- **A/B Testing**: Test new models against production
- **Rollback Strategy**: Version control for models

---

## 10. Scalability & Performance

### 10.1 Current Limitations
- Single-server deployment
- Synchronous request processing
- File-based model storage

### 10.2 Scalability Improvements
1. **Horizontal Scaling**: Deploy multiple backend instances
2. **Caching**: Redis for frequently requested predictions
3. **Async Processing**: Queue-based batch predictions
4. **Model Optimization**: ONNX runtime for faster inference

---

## 11. Future Enhancements

1. **Real-time Monitoring Dashboard**
   - Live prediction statistics
   - Model performance tracking
   - Alert system for anomalies

2. **Advanced Features**
   - Explainable AI (SHAP values)
   - Multi-model ensemble predictions
   - Time-series analysis for readmission patterns

3. **Integration Capabilities**
   - EHR system integration (HL7/FHIR)
   - Mobile application
   - API versioning (v2 with enhanced features)

4. **MLOps Pipeline**
   - Automated model retraining
   - Continuous integration/deployment (CI/CD)
   - Feature store integration

---

## 12. Technology Stack Summary

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TailwindCSS, Babel |
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **ML Framework** | scikit-learn, LightGBM, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Serialization** | Joblib, Pickle |
| **API Validation** | Pydantic |

---

## 13. Conclusion

The Hospital Readmission Prediction System provides a robust, scalable solution for predicting 30-day readmissions. The real-to-real training approach ensures reliable predictions, while the modular architecture allows for easy maintenance and future enhancements. The system successfully balances accuracy, interpretability, and performance, making it suitable for clinical deployment.

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Author**: Data Science Team  
**Status**: Production Ready