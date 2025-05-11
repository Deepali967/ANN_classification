import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load models and encoders
with open('scaler_reg.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encode_gender.pkl', 'rb') as f:
    label_gender = pickle.load(f)

with open('label_encode_geography.pkl', 'rb') as f:
    onehot_geography = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load the regression model
model_reg = tf.keras.models.load_model("model_regression.h5")

def regression_app():
    st.title("Salary Prediction using ANN Regression")

    # Collect user input
    gender = st.selectbox("Gender", ['Male', 'Female'])
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    credit_score = st.number_input("Credit Score", value=650)
    age = st.number_input("Age", value=35)
    tenure = st.number_input("Tenure", value=3)
    balance = st.number_input("Balance", value=100000.0)
    products = st.number_input("Number of Products", value=1)
    has_card = st.selectbox("Has Credit Card", ['Yes', 'No'])
    is_active = st.selectbox("Is Active Member", ['Yes', 'No'])
    is_exited = st.selectbox("Is Exited", ['Yes', 'No'])

    # Prepare input data
    input_dict = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': label_gender.transform([gender])[0],
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': 1 if has_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active == 'Yes' else 0,
        'Exited': 1 if is_exited == 'Yes' else 0
    }])

    st.write("Input Data:", input_dict)

    # One-hot encode Geography
    geo_encoded = onehot_geography.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_dict.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure all expected columns are present (some may be missing due to one-hot encoding)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with 0

    # Reorder columns to match training
    input_data = input_data[feature_columns]

    if st.button("Predict Salary"):
        st.write("Input Data after processing:", input_data)

        #Scale the data
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model_reg.predict(input_scaled)[0][0]

        #Display result
        st.success(f"Predicted Salary: ₹{prediction:,.2f}")