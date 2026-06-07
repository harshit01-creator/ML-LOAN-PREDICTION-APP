# model_training.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import pickle

# ──────────────────────────────────────────
# SYNTHETIC TRAINING DATA GENERATOR
# (Replace with real CSV data if available)
# ──────────────────────────────────────────

def generate_loan_training_data(n=2000, seed=42):
    """Generate realistic synthetic loan applicant records."""
    np.random.seed(seed)
    
    credit_score     = np.random.randint(300, 851, n)
    payment_history  = np.clip(np.random.normal(0.85, 0.15, n), 0, 1)
    late_payments    = np.random.poisson(1.5, n)
    defaults         = np.random.poisson(0.3, n)
    credit_age       = np.random.uniform(0.5, 30, n)
    annual_income    = np.random.lognormal(11, 0.5, n)   # ~60k median
    emp_years        = np.random.uniform(0, 30, n)
    emp_type         = np.random.randint(0, 3, n)
    existing_debt    = annual_income / 12 * np.random.uniform(0.1, 0.6, n)
    monthly_expenses = annual_income / 12 * np.random.uniform(0.2, 0.5, n)
    loan_amount      = annual_income * np.random.uniform(0.5, 5, n)
    collateral_value = loan_amount * np.random.uniform(0.6, 2, n)
    collateral_type  = np.random.randint(0, 4, n)
    loan_term        = np.random.choice([5, 10, 15, 20, 30], n)
    loan_purpose     = np.random.randint(0, 4, n)
    
    monthly_income   = annual_income / 12
    dti              = existing_debt / np.maximum(monthly_income, 1)
    ltv              = loan_amount / np.maximum(collateral_value, 1)
    monthly_emi      = loan_amount / (loan_term * 12)
    emi_to_income    = monthly_emi / np.maximum(monthly_income, 1)
    
    X = np.column_stack([
        credit_score, payment_history, late_payments, defaults,
        credit_age, annual_income, emp_years, emp_type,
        existing_debt, monthly_expenses, loan_amount, collateral_value,
        collateral_type, loan_term, loan_purpose, dti, ltv, emi_to_income
    ])
    
    # Label: 0 = Reject, 1 = Approve, 2 = Conditional
    risk = (
        (850 - credit_score) / 550 * 0.35 +
        late_payments / 10 * 0.15 +
        dti * 0.25 +
        ltv * 0.15 +
        emi_to_income * 0.10
    )
    y = np.where(risk < 0.3, 1, np.where(risk < 0.55, 2, 0))
    
    return X, y


def train_loan_model():
    X, y = generate_loan_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False,
                         eval_metric="mlogloss", random_state=42)
    xgb.fit(X_train, y_train)
    
    for name, model in [("LR", lr), ("RF", rf), ("XGB", xgb)]:
        preds = model.predict(X_test)
        print(f"\n{name} Report:\n", classification_report(y_test, preds,
              target_names=["Reject", "Approve", "Conditional"]))
    
    # Save best model (XGBoost)
    with open("loan_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    
    print("Loan model saved as loan_model.pkl")
    return xgb


# ──────────────────────────────────────────
# INSURANCE TRAINING
# ──────────────────────────────────────────

def generate_insurance_training_data(n=2000, seed=42):
    np.random.seed(seed)
    
    age          = np.random.randint(18, 80, n)
    gender       = np.random.randint(0, 3, n)
    bmi          = np.random.normal(25, 5, n)
    smoker       = np.random.randint(0, 2, n)
    income       = np.random.lognormal(11, 0.5, n)
    prop_value   = np.random.choice([0, 1], n, p=[0.3, 0.7]) * np.random.lognormal(13, 0.4, n)
    prop_age     = np.random.uniform(0, 40, n)
    flood        = np.random.randint(0, 2, n)
    dependents   = np.random.randint(0, 6, n)
    diseases     = np.random.randint(0, 2, n)
    prev_claims  = np.random.poisson(0.8, n)
    claim_hist   = prev_claims * np.random.lognormal(8, 0.5, n)
    
    X = np.column_stack([age, gender, bmi, smoker, income, prop_value,
                          prop_age, flood, dependents, diseases, prev_claims, claim_hist])
    
    # Insurance type: 0=health, 1=life, 2=property, 3=flood
    y = np.where(
        diseases == 1, 0,               # health
        np.where(age > 40, 1,           # life
        np.where(prop_value > 0, 2,     # property
        np.where(flood == 1, 3, 0)))    # flood else health
    )
    
    return X, y


def train_insurance_model():
    X, y = generate_insurance_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=200, use_label_encoder=False,
                           eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)
    
    print("\nInsurance Model:\n", classification_report(
        y_test, model.predict(X_test),
        target_names=["Health", "Life", "Property", "Flood"]))
    
    with open("insurance_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Insurance model saved.")
    return model


# ──────────────────────────────────────────
# FRAUD DETECTION TRAINING
# ──────────────────────────────────────────

def generate_fraud_training_data(n=2000, seed=42):
    np.random.seed(seed)
    
    claim_amount     = np.random.lognormal(9, 1, n)
    premium          = np.random.lognormal(8, 0.4, n)
    ratio            = claim_amount / np.maximum(premium, 1)
    days_since       = np.random.randint(1, 3650, n)
    prev_claims      = np.random.poisson(1, n)
    claim_type       = np.random.randint(0, 4, n)
    provider_given   = np.random.randint(0, 2, n)
    docs_submitted   = np.random.randint(0, 10, n)
    claim_hour       = np.random.randint(0, 24, n)
    witnesses        = np.random.randint(0, 5, n)
    
    X = np.column_stack([claim_amount, premium, ratio, days_since, prev_claims,
                          claim_type, provider_given, docs_submitted, claim_hour, witnesses])
    return X


def train_fraud_model():
    X = generate_fraud_training_data()
    
    # IsolationForest — unsupervised anomaly detection
    model = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
    model.fit(X)
    
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Fraud detection model saved.")
    return model