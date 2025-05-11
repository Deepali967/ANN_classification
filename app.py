import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

print(tf.__version__)

model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encode_gender.pkl', 'rb') as f:
    label_gender = pickle.load(f)

with open('label_encode_geography.pkl', 'rb') as f:
    onehot_geography = pickle.load(f)

# Load the encoder

st.title('Customer Churn Prediction')

st.write("This app predicts whether a customer will leave the bank based on their information.")

st.write("Please enter the following information:")

# Input fields
age = st.number_input('Age', min_value=18, max_value=100, value=30)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
geography = st.selectbox('Geography', onehot_geography.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, max_value=100000.0, value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0)
active_member = st.selectbox('Active Member', ['Yes', 'No'])

gender_encoded = label_gender.transform([gender])[0]

if has_cr_card == 'Yes':
    has_cr_card = 1
else:
    has_cr_card = 0

if active_member == 'Yes':
    active_member = 1
else:
    active_member = 0

input_data = pd.DataFrame([{
    "CreditScore" : credit_score,
    "Geography": geography,
    "Gender": gender_encoded,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": active_member,
    "EstimatedSalary": estimated_salary
}])

# Convert input to appropriate format
geo_encoded = onehot_geography.transform(np.array([[geography]])).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography.get_feature_names_out(['Geography']))

print('geo_encoded_df', geo_encoded_df)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = input_data.drop(columns=['Geography'])

st.write("Input Data:", input_data)

if st.button("Predict"):
    input_data_scaled = scaler.transform(input_data)  # Scale
    st.write("Scaled Input Data:", input_data_scaled)
    prediction = model.predict(input_data_scaled)  # Predict
    prediction_prod = prediction[0][0]

    st.write("Prediction Probability:", prediction_prod)

    if prediction_prod > 0.5:
        st.write("The model predicts that the customer will leave the bank.")
    else:
        st.write("The model predicts that the customer will stay with the bank.")