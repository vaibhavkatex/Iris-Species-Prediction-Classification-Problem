import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("rf_tuned_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page title
st.title("🌸 Iris Flower Prediction")

st.write("Enter the flower measurements and click Predict.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, value=0.2)

# Predict button
if st.button("Predict"):

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    # Scale data
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)[0]

    st.success(f"Predicted Species: {prediction}")
