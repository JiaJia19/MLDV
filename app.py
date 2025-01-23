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
    smoothness = st.number_input('Smoothness', min_value=0.00000, max_value=0.300000, step=0.010000)
    compactness = st.number_input('Compactness', min_value=0.000000, max_value=0.400000, step=0.010000)
    concavity = st.number_input('Concavity', min_value=0.000000, max_value=0.500000, step=0.010000)
with col3:
    concave_points = st.number_input('Concave Points', min_value=0.000000, max_value=0.300000, step=0.010000)
    symmetry = st.number_input('Symmetry', min_value=0.000000, max_value=0.400000, step=0.010000)
    fractal_dimension = st.number_input('Fractal Dimension', min_value=0.000000, max_value=0.100000, step=0.010000)
    
    
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

st.header('Input Features')
st.write(input_features)

if st.button ('Predict Breast Cancer'):
    prediction = predict(input_features)
    if prediction == "M":
        st.header('You have Malignant Cancer')
        st.write("A malignant tumor is cancerous. These tumors tend to grow rapidly, invade nearby tissues, and can spread to other parts of the body through the bloodstream or lymphatic system. This process of spreading is known as metastasis. Malignant tumors can be life-threatening and often require aggressive treatment, such as surgery, radiation, or chemotherapy.")
        
    else:
        st.header('You have Benign Cancer')
        st.write("A benign tumor is non-cancerous. These tumors usually grow slowly and do not invade nearby tissues or spread to other parts of the body. They are typically less dangerous than malignant tumors. However, depending on their size and location, benign tumors can still cause problems if they press against organs or structures. They are often removed surgically for this reason, but they usually do not recur after removal.")
