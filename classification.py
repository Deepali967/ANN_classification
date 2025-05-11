import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load models and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encode_gender.pkl', 'rb') as f:
    label_gender = pickle.load(f)

with open('label_encode_geography.pkl', 'rb') as f:
    onehot_geography = pickle.load(f)

# Load the classification model
model = tf.keras.models.load_model('model.h5')

def classification_app():
    st.title('Customer Churn Prediction')
    st.write("This app predicts whether a customer will leave the bank based on their information.")

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

    # Encode categorical variables
    gender_encoded = label_gender.transform([gender])[0]
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    active_member = 1 if active_member == 'Yes' else 0

    # Prepare input data
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
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

    geo_encoded = onehot_geography.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geography.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.drop(columns=['Geography']).reset_index(drop=True), geo_encoded_df], axis=1)

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]
        st.write(f"Prediction Probability: **{prediction:.2f}**")
        if prediction > 0.5:
            st.success("✅ The model predicts that the customer will **leave** the bank.")
        else:
            st.success("✅ The model predicts that the customer will **stay** with the bank.")
