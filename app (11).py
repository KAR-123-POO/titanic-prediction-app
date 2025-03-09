import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("🚢 Titanic Survival Prediction App")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert Inputs to Model Format
sex = 1 if sex == "Female" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
embarked_C = 0  # Not needed as it's the reference category

# Prepare input array
input_data = np.array([[pclass, sex, age, fare, sibsp, parch, embarked_Q, embarked_S]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]
    result = "Survived 🟢" if prediction == 1 else "Did NOT Survive 🔴"
    st.subheader(f"Prediction: {result}")
