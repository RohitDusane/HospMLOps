import joblib

MODEL_PATH = "app/artifacts/transformers/readmission_model.pkl"

model = joblib.load(MODEL_PATH)
