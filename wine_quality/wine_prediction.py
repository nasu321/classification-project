import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('wine_model.pkl', 'rb'))

# App title
st.title("🍷 Wine Type Prediction App")
st.write("Enter the wine chemical properties below to predict whether it's **Red** or **White**.")

# Collect input features
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 6.0)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 30.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 100.0)
density = st.number_input("Density", 0.9900, 1.0100, 0.9950)
pH = st.number_input("pH", 2.0, 4.0, 3.2)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.6)
alcohol = st.number_input("Alcohol", 5.0, 15.0, 10.0)

# Prediction
if st.button("Predict"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                            density, pH, sulphates, alcohol]])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🍇 The wine is **White Wine**!")
    else:
        st.error("🍷 The wine is **Red Wine**!")
6