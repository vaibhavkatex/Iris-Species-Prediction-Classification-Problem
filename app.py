import streamlit as st
import joblib
import numpy as np

# Load Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸")

st.title("🌸 Iris Flower Prediction")
st.write("Predict the species of an Iris flower using a KNN Machine Learning model.")

st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 3.0, 0.2)

features = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

scaled_features = scaler.transform(features)

prediction = model.predict(scaled_features)[0]

species = {
    0: "🌼 Iris-setosa",
    1: "🌺 Iris-versicolor",
    2: "🌹 Iris-virginica"
}

if st.button("Predict"):
    st.success(f"Predicted Species: **{species[prediction]}**")

st.markdown("---")
st.write("Model: K-Nearest Neighbors (KNN)")