# ============================================================================
# app/main.py - FastAPI Backend for Hospital Readmission Prediction
# ✅ UPDATED: Aligned with model_training.py pipeline
# ============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json
import sys
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="ML-powered API for predicting 30-day hospital readmissions",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# CORS middleware - Allow file:// protocol and all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "null"  # Allows file:// protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS - ✅ MATCHES model_training.py expectations
# ============================================================================

class PatientData(BaseModel):
    """
    ✅ ALIGNED: Patient data matching training pipeline
    Final Features: 18 features (excluding target readmitted_bin)
    """
    # Categorical Features
    race: str = Field(..., description="Patient race", json_schema_extra={"example": "Caucasian"})
    gender: str = Field(..., description="Patient gender", json_schema_extra={"example": "Female"})
    age: str = Field(..., description="Age group", json_schema_extra={"example": "[50-60)"})
    
    # Admission/Discharge Information
    admission_type_id: int = Field(..., ge=1, le=8, description="Type of admission", json_schema_extra={"example": 1})
    discharge_disposition_id: int = Field(..., ge=1, le=29, description="Discharge disposition", json_schema_extra={"example": 1})
    admission_source_id: int = Field(..., ge=1, le=25, description="Source of admission", json_schema_extra={"example": 7})
    
    # Clinical Metrics (Numerical)
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital", json_schema_extra={"example": 3})
    num_lab_procedures: int = Field(..., ge=1, description="Number of lab procedures", json_schema_extra={"example": 44})
    num_procedures: int = Field(..., ge=0, description="Number of procedures", json_schema_extra={"example": 1})
    num_medications: int = Field(..., ge=1, description="Number of medications", json_schema_extra={"example": 14})
    number_diagnoses: int = Field(..., ge=1, description="Number of diagnoses", json_schema_extra={"example": 8})
    
    # Visit History (Numerical)
    number_outpatient: int = Field(0, ge=0, description="Number of outpatient visits", json_schema_extra={"example": 0})
    number_emergency: int = Field(0, ge=0, description="Number of emergency visits", json_schema_extra={"example": 0})
    number_inpatient: int = Field(0, ge=0, description="Number of inpatient visits", json_schema_extra={"example": 0})
    
    # Medical Tests (Categorical)
    A1Cresult: str = Field("Not Available", description="A1C test result", json_schema_extra={"example": "Not Available"})
    
    # Medication Information (Categorical)
    insulin: str = Field("Steady", description="Insulin dosage change", json_schema_extra={"example": "Steady"})
    change: str = Field("Ch", description="Change in medication", json_schema_extra={"example": "Ch"})
    diabetesMed: str = Field("Yes", description="Diabetes medication prescribed", json_schema_extra={"example": "Yes"})

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int
    probability: float
    risk_level: str
    model_used: str
    timestamp: str
    feature_importance: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    total_patients: int
    high_risk_count: int
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    model_type: str
    n_features: int
    has_feature_importance: bool
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: bool
    available_models: List[str]
    n_features_expected: int


# ============================================================================
# MODEL MANAGER - ✅ ALIGNED with model_training.py
# ============================================================================

class ModelManager:
    """
    ✅ UPDATED: Manage ML models and transformers
    Aligned with model_training.py pipeline
    """
    
    # ✅ MUST MATCH: Feature order from training pipeline
    EXPECTED_FEATURES = [
        'A1Cresult', 'admission_source_id', 'admission_type_id', 'age',
        'change', 'diabetesMed', 'discharge_disposition_id', 'gender',
        'insulin', 'num_lab_procedures', 'num_medications', 'num_procedures',
        'number_diagnoses', 'number_emergency', 'number_inpatient',
        'number_outpatient', 'race', 'time_in_hospital'
    ]
    
    def __init__(self):
        self.models = {}
        self.transformers = {}
        self.feature_names = {}  # ✅ NEW: Store feature names per model
        self.model_dir = "app/artifacts/models"
        self.transformer_dir = "app/artifacts/transformers"
        
    def load_models(self):
        """
        ✅ UPDATED: Load models trained by model_training.py
        Only loads real_to_real models (production-ready)
        """
        try:
            # ✅ CHANGED: Only load real_to_real scenario (production)
            scenarios = ["real_to_real"]
            model_types = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
            
            loaded_count = 0
            for scenario in scenarios:
                for model_type in model_types:
                    model_path = os.path.join(self.model_dir, scenario, f"{model_type}.pkl")
                    
                    if os.path.exists(model_path):
                        try:
                            model = joblib.load(model_path)
                            model_key = f"{scenario}_{model_type}"
                            self.models[model_key] = model
                            
                            # ✅ NEW: Store feature names from model
                            if hasattr(model, 'n_features_in_'):
                                self.feature_names[model_key] = self.EXPECTED_FEATURES[:model.n_features_in_]
                            else:
                                self.feature_names[model_key] = self.EXPECTED_FEATURES
                            
                            print(f"✓ Loaded: {model_key}")
                            loaded_count += 1
                        except Exception as e:
                            print(f"✗ Failed to load {model_type}: {str(e)}")
            
            # Load transformers
            transformer_loaded = False
            for data_type in ["real"]:
                transformer_path = os.path.join(self.transformer_dir, f"{data_type}_transformers.pkl")
                
                if os.path.exists(transformer_path):
                    try:
                        self.transformers[data_type] = joblib.load(transformer_path)
                        print(f"✓ Loaded transformers: {data_type}")
                        transformer_loaded = True
                    except Exception as e:
                        print(f"✗ Failed to load transformer: {str(e)}")
            
            print(f"\n{'='*70}")
            print(f"Models loaded: {loaded_count}/{len(model_types)}")
            print(f"Transformers loaded: {'Yes' if transformer_loaded else 'No'}")
            print(f"Expected features: {len(self.EXPECTED_FEATURES)}")
            print(f"{'='*70}\n")
            
            if loaded_count == 0:
                print("⚠️  WARNING: No models loaded! Predictions will fail.")
            if not transformer_loaded:
                print("⚠️  WARNING: No transformers loaded! Preprocessing will fail.")
            
        except Exception as e:
            print(f"✗ Error in load_models: {str(e)}")
            raise
    
    def get_model(self, model_key: str = "real_to_real_logistic_regression"):
        """✅ UPDATED: Get a specific model with validation"""
        if model_key not in self.models:
            available = list(self.models.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_key}' not found. Available models: {available}")
        return self.models[model_key]
    
    def get_transformer(self, data_type: str = "real"):
        """✅ UPDATED: Get transformer with validation"""
        if data_type not in self.transformers:
            raise HTTPException(
                status_code=404,
                detail=f"Transformer '{data_type}' not found. Available: {list(self.transformers.keys())}"
            )
        return self.transformers[data_type]
    
    def preprocess_data(self, data: pd.DataFrame, data_type: str = "real") -> pd.DataFrame:
        """
        ✅ UPDATED: Preprocess input data using saved transformers
        MUST match the preprocessing in model_training.py
        """
        try:
            # Force real data type (production models)
            data_type = "real"
            transformer = self.get_transformer(data_type)
            
            # Drop target if present
            if 'readmitted_bin' in data.columns:
                data = data.drop(columns=['readmitted_bin'])
            
            # Get feature lists from transformer
            numerical_features = transformer.get('numerical_features', [])
            categorical_features = transformer.get('categorical_features', [])
            
            # ✅ CRITICAL: Validate we have all expected features
            all_features = set(numerical_features + categorical_features)
            expected_set = set(self.EXPECTED_FEATURES)
            
            missing_in_transformer = expected_set - all_features
            if missing_in_transformer:
                print(f"⚠️  Warning: Features in EXPECTED but not in transformer: {missing_in_transformer}")
            
            # Ensure all required features exist in input data
            for feat in numerical_features + categorical_features:
                if feat not in data.columns:
                    if feat in numerical_features:
                        data[feat] = 0
                        print(f"⚠️  Added missing numerical feature '{feat}' with default 0")
                    else:
                        data[feat] = "Not Available"
                        print(f"⚠️  Added missing categorical feature '{feat}' with default 'Not Available'")
            
            # ✅ STEP 1: Encode categorical features (SAME AS TRAINING)
            label_encoders = transformer.get('label_encoders', {})
            for col in categorical_features:
                if col in data.columns and col in label_encoders:
                    le = label_encoders[col]
                    # Handle unseen categories
                    data[col] = data[col].fillna('Missing').astype(str)
                    unseen_mask = ~data[col].isin(le.classes_)
                    if unseen_mask.any():
                        print(f"⚠️  Unseen categories in '{col}': {data.loc[unseen_mask, col].unique().tolist()}")
                        data.loc[unseen_mask, col] = 'Missing'
                    data[col] = le.transform(data[col])
            
            # ✅ STEP 2: Impute missing values (SAME AS TRAINING)
            cat_imputer = transformer.get('cat_imputer')
            num_imputer = transformer.get('num_imputer')
            
            if cat_imputer is not None and categorical_features:
                cat_features_present = [f for f in categorical_features if f in data.columns]
                if cat_features_present:
                    data[cat_features_present] = cat_imputer.transform(
                        data[cat_features_present]
                    )
            
            if num_imputer is not None and numerical_features:
                num_features_present = [f for f in numerical_features if f in data.columns]
                if num_features_present:
                    data[num_features_present] = num_imputer.transform(
                        data[num_features_present]
                    )
            
            # ✅ STEP 3: Scale numerical features (SAME AS TRAINING)
            scaler = transformer.get('scaler')
            if scaler is not None and numerical_features:
                num_features_present = [f for f in numerical_features if f in data.columns]
                if num_features_present:
                    data[num_features_present] = scaler.transform(
                        data[num_features_present]
                    )
            
            # ✅ STEP 4: Ensure correct feature order (CRITICAL!)
            # This MUST match the order used during training
            data = data[self.EXPECTED_FEATURES]
            
            # ✅ VALIDATION: Check for NaN and non-numeric
            if data.isnull().any().any():
                nan_cols = data.columns[data.isnull().any()].tolist()
                raise ValueError(f"NaN values found after preprocessing in columns: {nan_cols}")
            
            # Check data types
            non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                raise ValueError(f"Non-numeric columns found after preprocessing: {non_numeric}")
            
            return data
            
        except Exception as e:
            print(f"✗ Preprocessing error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Preprocessing error: {str(e)}"
            )


# ✅ Initialize model manager
model_manager = ModelManager()


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "="*70)
    print("Starting Hospital Readmission Prediction API")
    print("="*70)
    try:
        model_manager.load_models()
        print("✓ Startup complete")
    except Exception as e:
        print(f"✗ Startup error: {str(e)}")
        print("⚠️  API started but predictions may fail")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n✓ Shutting down API...\n")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    return Path("app/static/react.html").read_text()

# @app.get("/", response_model=HealthResponse)
# async def root():
#     """Root endpoint - API health check"""
#     return {
#         "status": "healthy" if len(model_manager.models) > 0 else "degraded",
#         "timestamp": datetime.now().isoformat(),
#         "models_loaded": len(model_manager.models) > 0,
#         "available_models": list(model_manager.models.keys()),
#         "n_features_expected": len(model_manager.EXPECTED_FEATURES)
#     }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if len(model_manager.models) > 0 else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_manager.models) > 0,
        "available_models": list(model_manager.models.keys()),
        "n_features_expected": len(model_manager.EXPECTED_FEATURES)
    }


@app.get("/features", response_model=List[str])
async def list_features():
    """List all expected features in correct order"""
    return model_manager.EXPECTED_FEATURES


@app.get("/models", response_model=List[str])
async def list_models():
    """List all available models"""
    return list(model_manager.models.keys())


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    patient: PatientData,
    model_key: str = "real_to_real_random_forest"
):
    """
    ✅ UPDATED: Predict readmission risk for a single patient
    Aligned with model_training.py preprocessing
    """
    try:
        # Validate model key
        if not model_key.startswith("real_to_real"):
            raise HTTPException(
                status_code=400,
                detail="Only real_to_real models are supported in production"
            )
        
        # Convert to DataFrame
        patient_dict = patient.dict()
        df = pd.DataFrame([patient_dict])
        
        # ✅ CRITICAL: Preprocess using SAME pipeline as training
        data_type = "real"
        df_processed = model_manager.preprocess_data(df, data_type)
        
        # Get model and predict
        model = model_manager.get_model(model_key)
        
        # ✅ VALIDATION: Check feature count matches
        if hasattr(model, 'n_features_in_'):
            if df_processed.shape[1] != model.n_features_in_:
                raise HTTPException(
                    status_code=500,
                    detail=f"Feature mismatch: got {df_processed.shape[1]}, expected {model.n_features_in_}"
                )
        
        # Predict
        prediction = int(model.predict(df_processed)[0])
        probability = float(model.predict_proba(df_processed)[0][1])
        probability = round(probability, 3)
        
        # Determine risk level (SAME THRESHOLDS AS TRAINING)
        model_thresholds = {
            "real_to_real_logistic_regression": {"high": 0.6, "medium": 0.3},
            "real_to_real_xgboost": {"high": 0.7, "medium": 0.4},
            "real_to_real_random_forest": {"high": 0.7, "medium": 0.4},
        }
        thresholds = model_thresholds.get(model_key, {"high":0.7,"medium":0.4})

        if probability >= thresholds["high"]:
            risk_level = "High"
        elif probability >= thresholds["medium"]:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(
                df_processed.columns,
                model.feature_importances_
            ))
            # Top 10 features
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            feature_importance = {k: round(v, 3) for k, v in sorted_features}
        
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "model_used": model_key,
            "timestamp": datetime.now().isoformat(),
            "feature_importance": feature_importance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    patients: List[PatientData],
    model_key: str = "real_to_real_random_forest"
):
    """
    ✅ UPDATED: Predict readmission risk for multiple patients
    """
    try:
        # Validate model key
        if not model_key.startswith("real_to_real"):
            raise HTTPException(
                status_code=400,
                detail="Only real_to_real models are supported"
            )
        
        predictions = []
        high_risk_count = 0
        
        for patient in patients:
            result = await predict_single(patient, model_key)
            predictions.append(result)
            if result.risk_level == "High":
                high_risk_count += 1
        
        return {
            "predictions": predictions,
            "total_patients": len(patients),
            "high_risk_count": high_risk_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.post("/predict/csv")
async def predict_from_csv(
    file: UploadFile = File(...),
    model_key: str = "real_to_real_random_forest"
):
    """
    ✅ UPDATED: Predict readmission risk from uploaded CSV file
    """
    try:
        # Validate model key
        if not model_key.startswith("real_to_real"):
            raise HTTPException(
                status_code=400,
                detail="Only real_to_real models are supported"
            )
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Drop target if present
        if 'readmitted_bin' in df.columns:
            df = df.drop(columns=['readmitted_bin'])
        
        # Validate features
        missing_features = set(model_manager.EXPECTED_FEATURES) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)}"
            )
        
        # Preprocess
        data_type = "real"
        df_processed = model_manager.preprocess_data(df, data_type)
        
        # Get model and predict
        model = model_manager.get_model(model_key)
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Add predictions to original dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.4, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Summary statistics
        summary = {
            "total_patients": len(df),
            "predicted_readmissions": int(predictions.sum()),
            "high_risk_count": int((df['risk_level'] == 'High').sum()),
            "medium_risk_count": int((df['risk_level'] == 'Medium').sum()),
            "low_risk_count": int((df['risk_level'] == 'Low').sum()),
            "average_probability": float(probabilities.mean()),
            "model_used": model_key,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "summary": summary,
            "predictions": df.to_dict(orient='records')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ CSV prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"CSV prediction error: {str(e)}"
        )


@app.get("/model/info/{model_key}", response_model=ModelInfo)
async def get_model_info(model_key: str):
    """✅ UPDATED: Get information about a specific model"""
    try:
        model = model_manager.get_model(model_key)
        
        info = {
            "model_name": model_key,
            "model_type": type(model).__name__,
            "n_features": getattr(model, 'n_features_in_', len(model_manager.EXPECTED_FEATURES)),
            "has_feature_importance": hasattr(model, 'feature_importances_'),
            "timestamp": datetime.now().isoformat()
        }
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )