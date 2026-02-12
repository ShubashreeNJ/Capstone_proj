import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_heart_model.pkl')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# Header
st.markdown("<h1 style='text-align: center; color: darkred;'>Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Enter your details below to predict the risk of heart disease.</p>",
    unsafe_allow_html=True)

# Create a form for inputs
with st.form(key='heart_form'):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=0)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", index=0)
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=0)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=0)
        chol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=0)
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No", index=0)

    with col2:
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], index=0)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, value=0)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                             index=0)
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2], index=0)
        ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3, 4], index=0)
        thal = st.selectbox("Thalassemia", options=[1, 2, 3], index=0)

    submit_button = st.form_submit_button(label="Predict")

# Prediction
if submit_button:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # If your model supports probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]  # probability of heart disease
    else:
        prob = None

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown(
            f"""
            <div style='
                background-color:#8B0000;
                padding:20px;
                border-radius:10px;
                color:white;
                text-align:center;
                font-size:18px;
            '>
            ⚠️ <b>High Risk!</b> The model predicts that the person <u>may have heart disease</u>.<br>
            {f"<b>Probability:</b> {prob * 100:.1f}%" if prob else ""}
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("**Recommendation:** Consult a doctor and consider lifestyle changes 💓")
    else:
        st.markdown(
            f"""
            <div style='
                background-color:#006400;
                padding:20px;
                border-radius:10px;
                color:white;
                text-align:center;
                font-size:18px;
            '>
            ✅ <b>Low Risk!</b> The model predicts that the person is <u>unlikely to have heart disease</u>.<br>
            {f"<b>Probability:</b> {(1 - prob) * 100:.1f}%" if prob else ""}
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("**Recommendation:** Maintain a healthy lifestyle 🥗🏃‍♂️")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)

