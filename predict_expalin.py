# predict_explain.py

import pickle
import numpy as np
import shap

LOAN_FEATURE_NAMES = [
    "Credit score", "Payment history", "Late payments", "Defaults",
    "Credit age", "Annual income", "Employment years", "Employment type",
    "Existing debt", "Monthly expenses", "Loan amount", "Collateral value",
    "Collateral type", "Loan term", "Loan purpose", "Debt-to-income",
    "Loan-to-value", "EMI-to-income"
]

INSURANCE_FEATURE_NAMES = [
    "Age", "Gender", "BMI", "Smoker", "Annual income",
    "Property value", "Property age", "Flood zone",
    "Dependents", "Existing diseases", "Previous claims", "Claim history"
]

FRAUD_FEATURE_NAMES = [
    "Claim amount", "Policy premium", "Claim/premium ratio",
    "Days since policy start", "Previous claims", "Claim type",
    "Provider given", "Documents submitted", "Claim hour", "Witnesses"
]

LOAN_LABELS = {0: "REJECTED", 1: "APPROVED", 2: "CONDITIONALLY APPROVED"}
INSURANCE_LABELS = {0: "Health", 1: "Life", 2: "Property", 3: "Flood"}


def predict_loan(feature_vector: np.ndarray, data: dict):
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    pred_class = model.predict(feature_vector)[0]
    pred_proba = model.predict_proba(feature_vector)[0]
    
    from feature_engineering import compute_risk_score, recommend_loan_amount, suggest_interest_rate
    risk_score     = compute_risk_score(data)
    recommended    = recommend_loan_amount(data)
    rate_range     = suggest_interest_rate(risk_score)
    
    print("\n========== LOAN DECISION ==========")
    print(f"  Decision      : {LOAN_LABELS[pred_class]}")
    print(f"  Confidence    : {pred_proba[pred_class]*100:.1f}%")
    print(f"  Risk Score    : {risk_score}/100")
    print(f"  Recommended   : {recommended:,.0f}")
    print(f"  Interest Rate : {rate_range[0]}% – {rate_range[1]}%")
    print("====================================")
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(feature_vector)
    
    print("\n--- Top Factors Influencing Decision ---")
    if isinstance(shap_vals, list):
        shap_for_class = shap_vals[pred_class][0]
    elif len(shap_vals.shape) == 3:
        shap_for_class = shap_vals[0, :, pred_class]
    else:
        shap_for_class = shap_vals[0]
    top_idx = np.argsort(np.abs(shap_for_class))[::-1][:5]
    for i in top_idx:
        direction = "↑ increases risk" if shap_for_class[i] > 0 else "↓ reduces risk"
        print(f"  {LOAN_FEATURE_NAMES[i]:<22} SHAP={shap_for_class[i]:+.3f}  {direction}")
    
    return pred_class, risk_score


def predict_insurance(feature_vector: np.ndarray, data: dict):
    with open("insurance_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    pred_class = model.predict(feature_vector)[0]
    pred_proba = model.predict_proba(feature_vector)[0]
    
    # Premium estimate (simple heuristic; replace with a trained regressor)
    base_premium = data["annual_income"] * 0.03
    age_factor   = 1 + max(0, data["age"] - 35) * 0.02
    risk_factor  = 1 + data["previous_claims"] * 0.15
    premium_est  = round(base_premium * age_factor * risk_factor, 2)
    
    print("\n========== INSURANCE RECOMMENDATION ==========")
    print(f"  Recommended Type  : {INSURANCE_LABELS[pred_class]}")
    print(f"  Confidence        : {pred_proba[pred_class]*100:.1f}%")
    print(f"  Estimated Premium : {premium_est:,.2f}")
    print("===============================================")
    
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(feature_vector)
    if isinstance(shap_vals, list):
        shap_for = shap_vals[pred_class][0]
    elif len(shap_vals.shape) == 3:
        shap_for = shap_vals[0, :, pred_class]
    else:
        shap_for = shap_vals[0]
    top_idx    = np.argsort(np.abs(shap_for))[::-1][:5]
    
    print("\n--- Key Factors ---")
    for i in top_idx:
        print(f"  {INSURANCE_FEATURE_NAMES[i]:<22} SHAP={shap_for[i]:+.3f}")
    
    return pred_class, premium_est


def predict_fraud(feature_vector: np.ndarray):
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # IsolationForest: -1 = anomaly (fraud), 1 = normal
    pred  = model.predict(feature_vector)[0]
    score = model.decision_function(feature_vector)[0]  # more negative = more anomalous
    
    # Normalise to 0–100 fraud probability
    fraud_prob = round(max(0, min(100, (1 - score) * 50)), 1)
    is_fraud   = pred == -1
    
    print("\n========== FRAUD DETECTION RESULT ==========")
    print(f"  Fraud Flag       : {'⚠ FLAGGED FOR REVIEW' if is_fraud else '✓ LOOKS NORMAL'}")
    print(f"  Fraud Score      : {fraud_prob}/100")
    print(f"  Anomaly Score    : {score:.4f}")
    print("=============================================")
    
    return is_fraud, fraud_prob