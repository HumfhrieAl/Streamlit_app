import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Loading trained DL model
model = load_model(r'Loan_default_prediction_project_model.h5')

# Defining the features of model expects
features = ['InterestRate', 'LoanTerm', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed']

st.title('Loan Default Prediction')

# Displays the image
st.image('images.jpeg')

# Creating input fields for each feature
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(f'Enter {feature}', value=0.0)

# When the 'Predict' button is clicked, make a prediction and display it
if st.button('Predict'):
    # Prepares the feature vector for prediction
    feature_vector = np.array([inputs[feature] for feature in features]).reshape(1, -1)

    # Makes the prediction
    prediction = model.predict(feature_vector)

    # Displays the prediction
    if prediction[0] > 0.5:
        st.write('The loan is likely to default.')
    else:
        st.write('The loan is likely not to default.')
