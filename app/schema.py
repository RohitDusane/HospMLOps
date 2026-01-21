from pydantic import BaseModel

class PatientInput(BaseModel):
    A1Cresult: str
    admission_source_id: int
    admission_type_id: int
    age: str
    change: str
    diabetesMed: str
    discharge_disposition_id: int
    gender: str
    insulin: str
    num_lab_procedures: int
    num_medications: int
    num_procedures: int
    number_diagnoses: int
    number_emergency: int
    number_inpatient: int
    number_outpatient: int
    race: str
    time_in_hospital: int
