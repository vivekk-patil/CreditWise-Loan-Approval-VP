import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

st.set_page_config(page_title="CreditWise | Loan Approval System", page_icon="💳", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
    .stApp { background-color: #F5F4F0; font-family: 'DM Sans', sans-serif; }
    header[data-testid="stHeader"] { background: transparent; }
    .hero { background: #1A1A2E; border-radius: 20px; padding: 2.5rem 2.5rem 2rem; margin-bottom: 1.5rem; color: white; }
    .hero h1 { font-family: 'DM Serif Display', serif; font-size: 2.4rem; margin: 0 0 0.4rem; color: white; }
    .hero p { color: #A0AEC0; font-size: 15px; margin: 0; }
    .form-card { background: white; border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; border: 1px solid #E8E4DC; box-shadow: 0 2px 12px rgba(0,0,0,0.05); }
    .form-card h3 { font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #1A1A2E; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #F5F4F0; }
    .result-approved { background: linear-gradient(135deg, #0F4C2A, #1A6B3A); border-radius: 16px; padding: 2rem; color: white; text-align: center; margin-top: 1rem; }
    .result-rejected { background: linear-gradient(135deg, #7B1E1E, #C0392B); border-radius: 16px; padding: 2rem; color: white; text-align: center; margin-top: 1rem; }
    .result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-title { font-family: 'DM Serif Display', serif; font-size: 2rem; margin-bottom: 0.3rem; }
    .result-sub { opacity: 0.85; font-size: 15px; }
    .stButton > button { background: #1A1A2E !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.65rem 2rem !important; font-size: 15px !important; font-weight: 600 !important; width: 100% !important; }
    .footer { text-align: center; padding: 1.5rem 0 0.5rem; color: #4B5563; font-size: 14px; font-weight: 600; }
    .footer a { color: #1A1A2E; text-decoration: none; font-weight: 700; }
    .badge { display: inline-block; background: #E8F5E9; color: #1B5E20; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
    <div class='badge'>AI POWERED</div>
    <h1>💳 CreditWise</h1>
    <p>Intelligent Loan Approval Prediction System — fill in applicant details to get an instant decision.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    if not os.path.exists("loan_approval_data.csv"):
        st.error("❌ Dataset `loan_approval_data.csv` not found.")
        st.stop()

    df = pd.read_csv("loan_approval_data.csv")
    df = df.drop(columns=["Applicant_ID"], errors="ignore")

    cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Education_Level","Gender","Employer_Category","Loan_Approved"]
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # Encode target & education
    df["Loan_Approved"]  = (df["Loan_Approved"] == "Yes").astype(int)
    df["Education_Level"] = (df["Education_Level"] == "Graduate").astype(int)

    # OHE
    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature engineering — drop originals
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df = df.drop(columns=["Credit_Score","DTI_Ratio"])

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, ohe, X.columns.tolist()

model, scaler, ohe, feature_cols = train_model()

# ── Form ──────────────────────────────────────────────────
st.markdown("<div class='form-card'><h3>👤 Personal Information</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender     = st.selectbox("Gender", ["Female", "Male"])
    marital    = st.selectbox("Marital Status", ["Married", "Single"])
    education  = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    age        = st.number_input("Age", min_value=18, max_value=80, value=30)
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
with col2:
    applicant_income   = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
    employment   = st.selectbox("Employment Status", ["Contract", "Salaried", "Self-employed", "Unemployed"])
    employer_cat = st.selectbox("Employer Category", ["Business", "Government", "MNC", "Private", "Unemployed"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='form-card'><h3>💰 Financial Details</h3>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    credit_score  = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    dti_ratio     = st.number_input("DTI Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    savings       = st.number_input("Savings (₹)", min_value=0, value=100000, step=5000)
with col4:
    loan_amount      = st.number_input("Loan Amount (₹)", min_value=0, value=500000, step=10000)
    loan_term        = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=60)
    existing_loans   = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0, value=200000, step=10000)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='form-card'><h3>🏠 Loan Details</h3>", unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    loan_purpose  = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
with col6:
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
st.markdown("</div>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval"):

    # OHE transform
    cat_input = pd.DataFrame([[employment, marital, loan_purpose, property_area, gender, employer_cat]],
                               columns=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"])
    ohe_encoded = ohe.transform(cat_input)
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out(
                          ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]))

    num_row = pd.DataFrame([{
        "Applicant_Income":   applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age":                age,
        "Dependents":         dependents,
        "Existing_Loans":     existing_loans,
        "Savings":            savings,
        "Collateral_Value":   collateral_value,
        "Loan_Amount":        loan_amount,
        "Loan_Term":          loan_term,
        "Education_Level":    1 if education == "Graduate" else 0,
        "DTI_Ratio_sq":       dti_ratio ** 2,
        "Credit_Score_sq":    credit_score ** 2,
    }])

    input_df = pd.concat([num_row.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    scaled     = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]
    proba      = model.predict_proba(scaled)[0]

    if prediction == 1:
        confidence = round(proba[1] * 100, 1)
        st.markdown(f"""
        <div class='result-approved'>
            <div class='result-icon'>✅</div>
            <div class='result-title'>Loan Approved</div>
            <div class='result-sub'>Confidence: {confidence}% — This applicant meets the credit criteria.</div>
        </div>""", unsafe_allow_html=True)
    else:
        confidence = round(proba[0] * 100, 1)
        st.markdown(f"""
        <div class='result-rejected'>
            <div class='result-icon'>❌</div>
            <div class='result-title'>Loan Rejected</div>
            <div class='result-sub'>Confidence: {confidence}% — This applicant does not meet the credit criteria.</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<footer style="
    text-align: center;
    padding: 15px;
    margin-top: 20px;
    border-top: 1px solid #ddd;
    color: #666;
    font-size: 14px;
">
    © 2026 AI Assistant | Developed by Vivek Satish Patil
</footer>
""", unsafe_allow_html=True)