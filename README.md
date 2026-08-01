# 🌸 Iris Flower Prediction using KNN

A Machine Learning web application built with **Streamlit** that predicts the species of an Iris flower using the **K-Nearest Neighbors (KNN)** algorithm.

## 🚀 Live Demo

Try the deployed Streamlit application:

🔗 **Streamlit App:**   
https://iris-species-predictionx.streamlit.app/

## Features

- Predicts Iris Species
- User-friendly Streamlit interface
- Machine Learning model trained using Scikit-learn
- Uses StandardScaler for feature scaling
- Deployable on Streamlit Cloud

## Dataset

- Iris Dataset
- 150 samples
- 4 Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

Target Classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Iris-KNN-Classifier.git
```

Move into the project

```bash
cd Iris-KNN-Classifier
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train_model.py
```

Run the Streamlit app

```bash
streamlit run app.py
```

## Model

Algorithm:

- K-Nearest Neighbors (KNN)

Hyperparameter:

- n_neighbors = 11

Train/Test Split:

- test_size = 0.33
- random_state = 42

## Project Structure

```
Iris-KNN-Classifier/
│
├── app.py
├── train_model.py
├── model.pkl
├── scaler.pkl
├── Iris.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Deployment

1. Push the project to GitHub.
2. Open Streamlit Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Click **Deploy**.

## Author

Vaibhav Katex
