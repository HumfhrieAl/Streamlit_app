import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("Loan_default_prediction_project_model1.h5")

def main():
    st.title("Loan Default Prediction App")
    st.sidebar.title("User Data Input")

    # Sidebar input widgets
    age = st.sidebar.slider("Age", 18, 100, 30)
    income = st.sidebar.number_input("Income ($)", min_value=0, max_value=1_000_000, value=50_000)
    loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0, max_value=1_000_000, value=10_000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
    months_employed = st.sidebar.number_input("Months Employed", min_value=0, max_value=360, value=24)
    num_credit_lines = st.sidebar.number_input("Number of Credit Lines", min_value=0, max_value=20, value=5)
    interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 20.0, 5.0)
    loan_term = st.sidebar.slider("Loan Term (months)", 6, 120, 36)
    dti_ratio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.4)
    education = st.sidebar.selectbox("Education", ["High School", "Bachelor's", "Master's", "Ph.D."])
    employment_type = st.sidebar.selectbox("Employment Type", ["Full-Time", "Part-Time", "Self-Employed"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage = st.sidebar.checkbox("Has Mortgage")
    has_dependents = st.sidebar.checkbox("Has Dependents")
    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Home Purchase", "Debt Consolidation", "Education", "Other"])
    has_cosigner = st.sidebar.checkbox("Has Co-Signer")

    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner]
    })

    # One-hot encode categorical features
    categorical_columns = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
    user_data = pd.get_dummies(user_data, columns=categorical_columns)

    # Convert input data to float32
    user_data = user_data.astype(np.float32)

    # Make predictions
    prediction = model.predict(user_data)
    default_probability = prediction[0][0]

     # Display prediction with risk level classification
    if prediction > 0.5:
        st.error("Risk of default: High")
    else:
        st.success("Risk of default: Low")

if __name__ == "__main__":
    main()
