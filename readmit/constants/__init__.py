
# ===== DATA INGESTION STEP =====
TEST_SIZE = 0.2
RANDOM_STATE = 24

# ===== DATA PROCESSING =====
SELECTED_VARIABLES = [
                'encounter_id', 'patient_nbr', 'age', 'gender', 'race',
                'time_in_hospital', 'admission_type_id', 'discharge_disposition_id',
                'admission_source_id', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_diagnoses', 'A1Cresult',
                'number_outpatient', 'number_emergency', 'number_inpatient',
                'insulin', 'diabetesMed', 'change', 'readmitted']
