# main.py

from data_input import collect_loan_applicant, collect_insurance_applicant, collect_fraud_claim
from preprocessing import preprocess_loan, preprocess_insurance, preprocess_fraud
from predict_expalin import predict_loan, predict_insurance, predict_fraud
from model_training import train_loan_model, train_insurance_model, train_fraud_model
import os


def train_all_models():
    """Train and save all models (run once)."""
    print("Training all models — this takes about 30 seconds...")
    train_loan_model()
    train_insurance_model()
    train_fraud_model()
    print("\nAll models trained and saved!")


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║  AI Underwriting Decision System      ║")
    print("╚══════════════════════════════════════╝")
    
    # Train models if not already saved
    if not os.path.exists("loan_model.pkl"):
        train_all_models()
    
    while True:
        print("\nWhat would you like to do?")
        print("  1. Evaluate a loan application")
        print("  2. Get insurance recommendation")
        print("  3. Check an insurance claim for fraud")
        print("  4. Retrain all models")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1–5): ").strip()
        
        if choice == "1":
            raw        = collect_loan_applicant()
            features, processed = preprocess_loan(raw)
            predict_loan(features, processed)
        
        elif choice == "2":
            raw        = collect_insurance_applicant()
            features, processed = preprocess_insurance(raw)
            predict_insurance(features, processed)
        
        elif choice == "3":
            raw        = collect_fraud_claim()
            features, processed = preprocess_fraud(raw)
            predict_fraud(features)
        
        elif choice == "4":
            train_all_models()
        
        elif choice == "5":
            print("Exiting. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1–5.")


if __name__ == "__main__":
    main()