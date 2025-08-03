import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and scaler using pickle
@st.cache_resource
def load_model_scaler():
    try:
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Feature list used during training (57 features)
ALL_FEATURES = [
    "width", "curb_weight", "engine_size", "horsepower", "cit_-mpg",  # numeric
    "make_audi", "make_bmw", "make_chevrolet", "make_dodge", "make_honda", "make_isuzu", "make_jaguar", "make_mazda",
    "make_mercedes-benz", "make_mercury", "make_mitsubishi", "make_nissan", "make_peugot", "make_plymouth", "make_porsche",
    "make_renault", "make_saab", "make_subaru", "make_toyota", "make_volkswagen", "make_volvo",
    "fuel_type_gas", "aspiration_turbo", "num_of_doors_two",
    "body_style_hardtop", "body_style_hatchback", "body_style_sedan", "body_style_wagon",
    "drive_wheels_fwd", "drive_wheels_rwd", "engine-location_rear",
    "engine_type_l", "engine_type_ohc", "engine_type_ohcf", "engine_type_ohcv", "engine_type_rotor",
    "num_of_cylinders_five", "num_of_cylinders_four", "num_of_cylinders_six", "num_of_cylinders_three",
    "num_of_cylinders_twelve", "num_of_cylinders_two",
    "fuel_system_2bbl", "fuel_system_4bbl", "fuel_system_idi", "fuel_system_mfi", "fuel_system_mpfi",
    "fuel_system_spdi", "fuel_system_spfi"
]

# Input form
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")

# Load model and scaler
model, scaler = load_model_scaler()
if model is None or scaler is None:
    st.stop()

# Numeric input
st.subheader("Enter Car Specifications:")
width = st.number_input("Width", min_value=50.0, max_value=80.0, value=65.0)
curb_weight = st.number_input("Curb Weight", min_value=1000.0, max_value=5000.0, value=2500.0)
engine_size = st.number_input("Engine Size", min_value=50.0, max_value=350.0, value=130.0)
horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=100.0)
city_mpg = st.number_input("City MPG", min_value=5.0, max_value=60.0, value=25.0)

# Categorical input
make = st.selectbox("Make", [
    "audi", "bmw", "chevrolet", "dodge", "honda", "isuzu", "jaguar", "mazda",
    "mercedes-benz", "mercury", "mitsubishi", "nissan", "peugot", "plymouth",
    "porsche", "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"
])
fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.selectbox("Aspiration", ["std", "turbo"])
num_of_doors = st.selectbox("Number of Doors", ["four", "two"])
body_style = st.selectbox("Body Style", ["sedan", "hatchback", "wagon", "hardtop"])
drive_wheels = st.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
engine_location = st.selectbox("Engine Location", ["front", "rear"])
engine_type = st.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "l", "rotor"])
num_of_cylinders = st.selectbox("Number of Cylinders", ["four", "six", "five", "three", "twelve", "two"])
fuel_system = st.selectbox("Fuel System", ["mpfi", "2bbl", "idi", "mfi", "4bbl", "spdi", "spfi"])

# Create input DataFrame
def prepare_input():
    data = pd.DataFrame([[0]*len(ALL_FEATURES)], columns=ALL_FEATURES)

    # Fill numeric
    data["width"] = width
    data["curb_weight"] = curb_weight
    data["engine_size"] = engine_size
    data["horsepower"] = horsepower
    data["cit_-mpg"] = city_mpg

    # Fill encoded columns
    if f"make_{make}" in data.columns:
        data[f"make_{make}"] = 1
    if f"fuel_type_{fuel_type}" in data.columns:
        data[f"fuel_type_{fuel_type}"] = 1
    if f"aspiration_{aspiration}" in data.columns:
        data[f"aspiration_{aspiration}"] = 1
    if f"num_of_doors_{num_of_doors}" in data.columns:
        data[f"num_of_doors_{num_of_doors}"] = 1
    if f"body_style_{body_style}" in data.columns:
        data[f"body_style_{body_style}"] = 1
    if f"drive_wheels_{drive_wheels}" in data.columns:
        data[f"drive_wheels_{drive_wheels}"] = 1
    if f"engine-location_{engine_location}" in data.columns:
        data[f"engine-location_{engine_location}"] = 1
    if f"engine_type_{engine_type}" in data.columns:
        data[f"engine_type_{engine_type}"] = 1
    if f"num_of_cylinders_{num_of_cylinders}" in data.columns:
        data[f"num_of_cylinders_{num_of_cylinders}"] = 1
    if f"fuel_system_{fuel_system}" in data.columns:
        data[f"fuel_system_{fuel_system}"] = 1

    return data

# Predict
if st.button("Predict Price"):
    input_df = prepare_input()

    if input_df.shape[1] != scaler.n_features_in_:
        st.error("🚫 Feature mismatch. Please check column alignment.")
        st.stop()

    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.success(f"💰 Predicted Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
