import streamlit as st
import pandas as pd 
import numpy as np
import joblib


# Load Dataset
data = pd.read_csv('./Data/health_brest_cancer_dataSet.csv')

st.title('Breast Cancer Prediction')

st.write('Raw Data')
st.write(data)

# 0 Texture
# 1 Perimeter
# 2 Area
# 3 Smoothness
# 4 Compactness
# 5 Concavity
# 6 Concave Points
# 7 Symmetry
# 8 Fractal Dimension
st.header('Input Data for Prediction')
col1, col2, col3 = st.columns(3)
with col1:
    texture = st.number_input('Texture', min_value=9.0, max_value=40.0, step=1.0)
    perimeter = st.number_input('Perimeter', min_value=43.0, max_value=200.0, step=1.0)
    area = st.number_input('Area', min_value=143.0, max_value=2501.0, step=1.0)
with col2:
    smoothness = st.number_input('Smoothness', min_value=0.0, max_value=0.3, step=0.01)
    compactness = st.number_input('Compactness', min_value=0.0, max_value=0.4, step=0.01)
    concavity = st.number_input('Concavity', min_value=0.0, max_value=0.5, step=0.01)
with col3:
    concave_points = st.number_input('Concave Points', min_value=0.0, max_value=0.3, step=0.01)
    symmetry = st.number_input('Symmetry', min_value=0.0, max_value=0.4, step=0.01)
    fractal_dimension = st.number_input('Fractal Dimension', min_value=0.0, max_value=0.1, step=0.01)
    
    
input_features = pd.DataFrame({
    '0': [texture],
    '1': [perimeter],
    '2': [area],
    '3': [smoothness],
    '4': [compactness],
    '5': [concavity],
    '6': [concave_points],
    '7': [symmetry],
    '8': [fractal_dimension]
})


model = joblib.load('brest_cancer_app.pkl')

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

st.write('Input Features')
st.write(input_features)

if st.button ('Predict Breast Cancer'):
    prediction = predict(input_features)
    st.write(prediction)