import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Breast Cancer Prediction app")
# st.help(st.selectbox)
with st.form(key = "form1", clear_on_submit =True):

    
    symmetry_mean = st.number_input("symmetry_mean")
    texture_se = st.number_input("texture_se")
    smoothness_se = st.number_input("smoothness_se")
    concave_points_se = st.number_input("concave points_se'")
    symmetry_se = st.number_input("symmetry_se")
    fractal_dimension_se = st.number_input("fractal_dimension_se")
    texture_worst = st.number_input("texture_worst")
    area_worst = st.number_input("area_worst")
    smoothness_worst = st.number_input("smoothness_worst")
    concave_points_worst = st.number_input("concave points_worst")
    symmetry_worst = st.number_input("symmetry_worst")
    fractal_dimension_worst = st.number_input("fractal_dimension_worst")
        
    submit = st.form_submit_button(label = "Predict")


model = joblib.load("breast_cancer.pkl")

result = model.predict([[symmetry_mean, texture_se, smoothness_se, concave_points_se,symmetry_se, fractal_dimension_se, texture_worst, area_worst,smoothness_worst, concave_points_worst, symmetry_worst,fractal_dimension_worst]])

# st.success("Run the model")
if result==0:
    out = "Malignant"
else:
     out = "Benign"

st.title("Predictions")
st.success(f"you have {out}")