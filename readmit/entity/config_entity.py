# Core 17 selected variables (excluding identifiers and target)
CORE_FEATURES = [
    # Demographics
    'age', 'gender', 'race',
    
    # Hospital stay characteristics
    'time_in_hospital', 'admission_type_id', 
    'discharge_disposition_id', 'admission_source_id',
    
    # Clinical measures
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_diagnoses', 'A1Cresult',
    
    # Prior utilization
    'number_outpatient', 'number_emergency', 'number_inpatient',
    
    # Medication management
    'insulin', 'diabetesMed', 'change'
]

# Identifiers (kept but not used in modeling)
IDENTIFIERS = ['encounter_id', 'patient_nbr']

# Target variables
TARGET_COLS = ['readmitted']

# ======================
# Expected Schema
# ======================
REAL_EXPECTED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "age",
    "gender",
    "race",
    "time_in_hospital",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "A1Cresult",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "insulin",
    "diabetesMed",
    "change",
    "readmitted",
    "readmitted_bin"
]

SYNTHETIC_EXPECTED_COLUMNS = [
    # ❌ no IDs
    "age",
    "gender",
    "race",
    "time_in_hospital",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "A1Cresult",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "insulin",
    "diabetesMed",
    "change",
    "readmitted",
    "readmitted_bin"
]

# SEMANTIC_COLUMN_MAP = {
#     "age": "age_group"
# }
# AGE_GROUP_VALUES = {"young", "middle", "senior"}