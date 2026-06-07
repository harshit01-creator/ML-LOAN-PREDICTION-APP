# data_input.py

def collect_loan_applicant():
    """Collect loan applicant data via manual input."""
    print("\n=== LOAN APPLICANT DATA ENTRY ===\n")
    
    data = {}
    
    # --- CREDIT ---
    print("--- Credit ---")
    data["credit_score"]        = int(input("Credit score (300–850): "))
    data["payment_history"]     = float(input("On-time payment rate (0.0–1.0): "))
    data["num_late_payments"]   = int(input("Number of late payments (last 2 yrs): "))
    data["num_defaults"]        = int(input("Number of past defaults: "))
    data["credit_age_years"]    = float(input("Credit history age (years): "))
    
    # --- CAPACITY ---
    print("\n--- Capacity ---")
    data["annual_income"]       = float(input("Annual income (₹ or $): "))
    data["employment_years"]    = float(input("Years at current employer: "))
    data["employment_type"]     = input("Employment type (salaried/self-employed/contract): ").strip().lower()
    data["existing_debt"]       = float(input("Total existing monthly debt payments: "))
    data["monthly_expenses"]    = float(input("Total monthly living expenses: "))
    
    # --- COLLATERAL ---
    print("\n--- Collateral ---")
    data["loan_amount"]         = float(input("Requested loan amount: "))
    data["collateral_value"]    = float(input("Collateral/property value: "))
    data["collateral_type"]     = input("Collateral type (property/vehicle/fixed_deposit/none): ").strip().lower()
    data["loan_term_years"]     = int(input("Loan term (years): "))
    data["loan_purpose"]        = input("Loan purpose (home/auto/business/personal): ").strip().lower()
    
    return data


def collect_insurance_applicant():
    """Collect insurance applicant data via manual input."""
    print("\n=== INSURANCE APPLICANT DATA ENTRY ===\n")
    
    data = {}
    data["age"]                 = int(input("Applicant age: "))
    data["gender"]              = input("Gender (male/female/other): ").strip().lower()
    data["bmi"]                 = float(input("BMI: "))
    data["smoker"]              = input("Smoker? (yes/no): ").strip().lower()
    data["annual_income"]       = float(input("Annual income: "))
    data["property_value"]      = float(input("Property value (0 if none): "))
    data["property_age_years"]  = float(input("Property age in years (0 if none): "))
    data["flood_zone"]          = input("Property in flood zone? (yes/no): ").strip().lower()
    data["num_dependents"]      = int(input("Number of dependents: "))
    data["existing_diseases"]   = input("Existing diseases? (yes/no): ").strip().lower()
    data["previous_claims"]     = int(input("Number of previous insurance claims: "))
    data["claim_amount_history"]= float(input("Total past claim amount (0 if none): "))
    
    return data


def collect_fraud_claim():
    """Collect insurance claim data for fraud detection."""
    print("\n=== INSURANCE CLAIM DATA ENTRY ===\n")
    
    data = {}
    data["claim_amount"]            = float(input("Claim amount: "))
    data["policy_premium"]          = float(input("Annual policy premium: "))
    data["claim_to_premium_ratio"]  = data["claim_amount"] / max(data["policy_premium"], 1)
    data["days_since_policy_start"] = int(input("Days since policy started: "))
    data["num_previous_claims"]     = int(input("Number of previous claims by this user: "))
    data["claim_type"]              = input("Claim type (health/property/vehicle/life): ").strip().lower()
    data["hospital_name_provided"]  = input("Hospital/provider name given? (yes/no): ").strip().lower()
    data["documents_submitted"]     = int(input("Number of documents submitted: "))
    data["claim_hour"]              = int(input("Hour of claim filing (0–23): "))
    data["num_witnesses"]           = int(input("Number of witnesses: "))
    
    return data