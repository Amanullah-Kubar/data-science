import streamlit as st
import pickle
import numpy as np

# Load model and scaler
def load_model():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None


model, scaler = load_model()

st.title("Loan Prediction System")
st.write("Fill the form below to check loan approval")

if model is None or scaler is None:
    st.error("Model or scaler not found. Please place best_model.pkl and scaler.pkl in the same folder.")
else:
    # User Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Co-applicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (days)", min_value=0)
    
    credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Convert input to features
    def preprocess_input(data):
        gender = 1 if data['gender'] == 'Male' else 0
        married = 1 if data['married'] == 'Yes' else 0
        dependents = 4 if data['dependents'] == '3+' else int(data['dependents'])
        education = 1 if data['education'] == 'Graduate' else 0
        self_employed = 1 if data['self_employed'] == 'Yes' else 0
        
        property_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        property_area = property_mapping[data['property_area']]
        
        features = np.array([[
            gender, married, dependents, education, self_employed,
            float(data['applicant_income']),
            float(data['coapplicant_income']),
            float(data['loan_amount']),
            float(data['loan_term']),
            float(data['credit_history']),
            property_area
        ]])
        
        return features

    # Button
    if st.button("Predict Loan Status"):
        data = {
            "gender": gender,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_history": credit_history,
            "property_area": property_area
        }

        features = preprocess_input(data)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        # Confidence (optional)
        try:
            prob = model.predict_proba(scaled)[0]
            confidence = prob[prediction] * 100
        except:
            confidence = None

        # Show result
        if prediction == 1:
            st.success(f"Loan Approved ✓")
        else:
            st.error("Loan Rejected ✗")

        if confidence is not None:
            st.write(f"**Confidence:** {confidence:.2f}%")

