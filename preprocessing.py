# preprocessing.py

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_loan(data: dict):
    """Encode categoricals and scale numerics for loan data."""
    
    # --- Encode categorical fields ---
    emp_map  = {"salaried": 2, "self-employed": 1, "contract": 0}
    coll_map = {"property": 3, "vehicle": 2, "fixed_deposit": 1, "none": 0}
    purp_map = {"home": 3, "auto": 2, "business": 1, "personal": 0}
    
    data["employment_type_enc"] = emp_map.get(data["employment_type"], 0)
    data["collateral_type_enc"] = coll_map.get(data["collateral_type"], 0)
    data["loan_purpose_enc"]    = purp_map.get(data["loan_purpose"], 0)
    
    # --- Derived features (Step 4 feeds into here) ---
    monthly_income = data["annual_income"] / 12
    data["debt_to_income"]      = (data["existing_debt"] / monthly_income) if monthly_income > 0 else 0
    data["loan_to_value"]       = (data["loan_amount"] / data["collateral_value"]) if data["collateral_value"] > 0 else 99
    data["monthly_emi_estimate"]= data["loan_amount"] / (data["loan_term_years"] * 12)
    data["emi_to_income"]       = data["monthly_emi_estimate"] / monthly_income if monthly_income > 0 else 0
    
    # --- Final feature vector (order matters for model) ---
    features = [
        data["credit_score"],
        data["payment_history"],
        data["num_late_payments"],
        data["num_defaults"],
        data["credit_age_years"],
        data["annual_income"],
        data["employment_years"],
        data["employment_type_enc"],
        data["existing_debt"],
        data["monthly_expenses"],
        data["loan_amount"],
        data["collateral_value"],
        data["collateral_type_enc"],
        data["loan_term_years"],
        data["loan_purpose_enc"],
        data["debt_to_income"],
        data["loan_to_value"],
        data["emi_to_income"],
    ]
    
    return np.array(features).reshape(1, -1), data


def preprocess_insurance(data: dict):
    """Encode and prepare insurance applicant features."""
    
    gender_map  = {"male": 0, "female": 1, "other": 2}
    binary_map  = {"yes": 1, "no": 0}
    
    features = [
        data["age"],
        gender_map.get(data["gender"], 0),
        data["bmi"],
        binary_map.get(data["smoker"], 0),
        data["annual_income"],
        data["property_value"],
        data["property_age_years"],
        binary_map.get(data["flood_zone"], 0),
        data["num_dependents"],
        binary_map.get(data["existing_diseases"], 0),
        data["previous_claims"],
        data["claim_amount_history"],
    ]
    
    return np.array(features).reshape(1, -1), data


def preprocess_fraud(data: dict):
    """Encode and prepare fraud claim features."""
    
    claim_type_map = {"health": 0, "property": 1, "vehicle": 2, "life": 3}
    binary_map     = {"yes": 1, "no": 0}
    
    features = [
        data["claim_amount"],
        data["policy_premium"],
        data["claim_to_premium_ratio"],
        data["days_since_policy_start"],
        data["num_previous_claims"],
        claim_type_map.get(data["claim_type"], 0),
        binary_map.get(data["hospital_name_provided"], 0),
        data["documents_submitted"],
        data["claim_hour"],
        data["num_witnesses"],
    ]
    
    return np.array(features).reshape(1, -1), data