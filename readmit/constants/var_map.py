# ======================
# Age bucket mapping
# ======================
AGE_MAP = {
    "[0-10)": "0-10",
    "[10-20)": "10-20",
    "[20-30)": "20-30",
    "[30-40)": "30-40",
    "[40-50)": "40-50",
    "[50-60)": "50-60",
    "[60-70)": "60-70",
    "[70-80)": "70-80",
    "[80-90)": "80+",
    "[90-100)": "80+"
}

# ======================
# Binary / reduced maps
# ======================
BINARY_MAPS = {
    "gender": {
        "Male": 0,
        "Female": 1,
        "Unknown/Invalid":0
    },
    "diabetesMed": {
        "Yes": 1,
        "No": 0
    },
    "change": {
        "Ch": 1,
        "No": 0
    },
    "insulin": {
        "Up": 1,
        "Down": 1,
        "Steady": 1,
        "No": 0
    },
    "A1Cresult": {
        ">8": 1,
        ">7": 1,
        "Norm": 0,
        "None": 0
    },
    "readmitted": {
        "<30": 1,
        ">30": 0,
        "NO": 0
    }
}

# ======================
# Race grouping
# ======================
RACE_MAP = {
    "Caucasian": "Caucasian",
    "AfricanAmerican": "AfricanAmerican",
    "Asian": "Other",
    "Hispanic": "Other",
    "Other": "Other"
}
