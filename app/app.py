import streamlit as st
import joblib
import pandas as pd
import os

# Load model and saved feature list
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir,"..","models","house_price_model.pkl")
features_path = os.path.join(current_dir,"..","models","model_features.pkl")
model = joblib.load(model_path)
features = joblib.load(features_path)

st.title("🏠 House Price Predictor")

st.write("Enter house details")

overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sqft)", 500, 5000, 1500)
garage_cars = st.number_input("Garage Cars", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area", 0, 3000, 800)
year_built = st.number_input("Year Built", 1900, 2025, 2000)

if st.button("Predict Price"):

    # Create empty feature vector
    input_data = pd.DataFrame([[0]*len(features)], columns=features)

    # Fill the fields collected from UI
    if "Overall Qual" in input_data.columns:
        input_data.at[0, "Overall Qual"] = overall_qual

    if "Gr Liv Area" in input_data.columns:
        input_data.at[0, "Gr Liv Area"] = gr_liv_area

    if "Garage Cars" in input_data.columns:
        input_data.at[0, "Garage Cars"] = garage_cars

    if "Total Bsmt SF" in input_data.columns:
        input_data.at[0, "Total Bsmt SF"] = total_bsmt_sf

    if "Year Built" in input_data.columns:
        input_data.at[0, "Year Built"] = year_built

    # Ensure correct column order
    input_data = input_data[features]

    prediction = model.predict(input_data)

    st.success(f"Estimated House Price: ${prediction[0]:,.0f}")
