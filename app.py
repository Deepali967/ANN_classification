import streamlit as st
from classification import classification_app
from regression import regression_app

# Navigation bar
nav_selection = st.sidebar.radio("Navigation", ["Classification", "Regression"])

if nav_selection == "Classification":
    classification_app()

elif nav_selection == "Regression":
    regression_app()
