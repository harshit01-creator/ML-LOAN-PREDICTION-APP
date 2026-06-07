# feature_engineering.py

def compute_risk_score(data: dict) -> float:
    """
    Compute a composite risk score (0–100, higher = riskier)
    based on the three C's.
    """
    score = 0
    
    # Credit component (0–40 pts)
    credit_norm = max(0, (850 - data["credit_score"]) / 550)  # 0 = best, 1 = worst
    score += credit_norm * 20
    score += min(data["num_late_payments"] * 3, 10)
    score += min(data["num_defaults"] * 5, 10)
    
    # Capacity component (0–35 pts)
    dti = data.get("debt_to_income", 0)
    score += min(dti * 20, 20)  # DTI > 1 = maxed out
    score += min(data.get("emi_to_income", 0) * 15, 10)
    score += (5 if data.get("employment_years", 5) < 1 else 0)
    
    # Collateral component (0–25 pts)
    ltv = data.get("loan_to_value", 0)
    score += min(ltv * 15, 20)  # LTV > 1 = undercollateralised
    score += (5 if data.get("collateral_type", "none") == "none" else 0)
    
    return round(min(score, 100), 2)


def recommend_loan_amount(data: dict) -> float:
    """Suggest a safe loan amount based on income and collateral."""
    monthly_income  = data["annual_income"] / 12
    max_emi         = monthly_income * 0.4  # 40% of income rule
    max_by_emi      = max_emi * data["loan_term_years"] * 12
    max_by_collateral = data["collateral_value"] * 0.8  # 80% LTV cap
    return round(min(max_by_emi, max_by_collateral), 2)


def suggest_interest_rate(risk_score: float) -> tuple:
    """Return (min_rate%, max_rate%) based on risk score."""
    if risk_score <= 20:
        return (6.5, 8.0)
    elif risk_score <= 40:
        return (8.0, 10.5)
    elif risk_score <= 60:
        return (10.5, 13.0)
    elif risk_score <= 80:
        return (13.0, 16.0)
    else:
        return (16.0, 20.0)