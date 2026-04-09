import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sagarravindra/Tourism-Churn-Prediction-04042026", filename="best_tourism-prediction-model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Churn Prediction App")
st.write("""
This application predicts the likelihood of a customer taking the Tourism Product based on customer profile and product features offered.
Please enter the customer and tourism product data below to get a Customer Churn prediction.
""")

# =========================
# USER INPUTS (MATCH DATA)
# =========================

Age = st.number_input("Age", 18, 100, 30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input("Duration Of Pitch", 0.0, 100.0, 10.0)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 2)

NumberOfFollowups = st.number_input("Number of Followups", 0, 10, 2)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input("Number of Trips", 0, 20, 2)

Passport = st.selectbox("Passport", [0, 1])

PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

OwnCar = st.selectbox("Own Car", [0, 1])

NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 10, 0)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input("Monthly Income", 0.0, 500000.0, 20000.0)

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# =========================
# PREDICTION
# =========================

if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "Product NOT Taken"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

