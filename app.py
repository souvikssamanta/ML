# import streamlit as st
# import pandas as pd
# import joblib
# model=joblib.load('Logistic_heart.pkl')
# scaler=joblib.load('scaler.pkl')
# expected_columns=joblib.load('columns.pkl')
# st.title("Heart disease Prediction model")
# st.markdown("Provide the following details to check your heart stroke risk:")

# # Collect user input
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ["M", "F"])
# chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# # When Predict is clicked
# if st.button("Predict"):

#     # Create a raw input dictionary
#     raw_input = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chest_pain: 1,
#         'RestingECG_' + resting_ecg: 1,
#         'ExerciseAngina_' + exercise_angina: 1,
#         'ST_Slope_' + st_slope: 1
#     }

#     # Create input dataframe
#     input_df = pd.DataFrame([raw_input])

#     # Fill in missing columns with 0s
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Reorder columns
#     input_df = input_df[expected_columns]

#     # Scale the input
#     scaled_input = scaler.transform(input_df)

#     # Make prediction
#     prediction = model.predict(scaled_input)[0]

#     # Show result
#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")


import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time

# Load model and assets
model = joblib.load('Logistic_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# App configuration
st.set_page_config(
    page_title="Heart Health Check",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .title {
        color: #ff4b4b;
        text-align: center;
        font-size: 2.5em !important;
    }
    .subheader {
        color: #555;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        transform: scale(1.05);
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
    }
    .progress-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 10px 0;
    }
    .progress {
        height: 100%;
        border-radius: 5px;
        background-color: #ff4b4b;
        transition: width 1s;
    }
</style>
""", unsafe_allow_html=True)

# Header with image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.markdown('<h1 class="title">Heart Health Check</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Assess your risk of heart disease with our predictive model</p>', unsafe_allow_html=True)

# Information expander
with st.expander("ℹ️ About this tool"):
    st.write("""
    This tool uses machine learning to predict your risk of heart disease based on several health indicators. 
    It's designed for educational purposes only and should not replace professional medical advice.
    
    The model considers factors like:
    - Age and gender
    - Blood pressure and cholesterol
    - ECG results and heart rate
    - Exercise-induced symptoms
    """)

# Risk factors sidebar
st.sidebar.header("Heart Health Tips")
st.sidebar.write("""
- 🚬 Avoid smoking
- � Maintain healthy weight
- 🏃‍♂️ Exercise regularly
- 🥗 Eat balanced diet
- 😴 Get enough sleep
- 🧘 Manage stress
""")

st.sidebar.header("Recommended Tests")
st.sidebar.write("""
- Blood pressure check
- Cholesterol test
- Blood sugar test
- BMI calculation
- ECG/EKG
""")

# User input section
st.header("Your Health Information")
st.markdown("Please provide accurate information for the most reliable prediction.")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40, help="Your current age in years")
    sex = st.radio("Sex", ["Male", "Female"], format_func=lambda x: "♂️ Male" if x == "Male" else "♀️ Female")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], 
                            help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, 
                               help="Your blood pressure at rest")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200,
                                help="Your cholesterol level")

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                            format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
                             help="Results of electrocardiogram at rest")
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150,
                      help="Maximum heart rate achieved during exercise")
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"],
                                 format_func=lambda x: "✅ Yes" if x == "Yes" else "❌ No")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1,
                       help="ST depression induced by exercise relative to rest")
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"],
                           help="Slope of the peak exercise ST segment")

# Convert inputs to model format
sex = "M" if sex == "Male" else "F"
exercise_angina = "Y" if exercise_angina == "Yes" else "N"

# Prediction button
if st.button("🔍 Analyze My Heart Health"):
    with st.spinner("Analyzing your data..."):
        # Show progress bar animation
        progress_bar = st.empty()
        for percent_complete in range(100):
            progress_bar.markdown(f"""
            <div class="progress-bar">
                <div class="progress" style="width:{percent_complete + 1}%"></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.01)
        
        # Create input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1] * 100

        # Show result with animation
        st.balloons()
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box high-risk">
                ⚠️ High Risk of Heart Disease ({prediction_proba:.1f}% probability)
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("""
            **Recommendations:**
            - Consult a cardiologist soon
            - Monitor your blood pressure regularly
            - Consider lifestyle changes
            - Schedule a comprehensive heart check-up
            """)
        else:
            st.markdown(f"""
            <div class="result-box low-risk">
                ✅ Low Risk of Heart Disease ({100 - prediction_proba:.1f}% probability)
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            **Good job! To maintain heart health:**
            - Continue healthy habits
            - Get regular check-ups
            - Stay physically active
            - Maintain balanced diet
            """)
        
        # Show risk factors visualization
        st.subheader("Risk Factors Overview")
        factors = {
            "Age": min(age / 100, 1),
            "Blood Pressure": min((resting_bp - 80) / 120, 1),
            "Cholesterol": min((cholesterol - 100) / 500, 1),
            "Heart Rate": 1 - min((max_hr - 60) / 160, 1),
            "ST Depression": min(oldpeak / 6, 1)
        }
        
        for factor, value in factors.items():
            st.write(f"{factor}")
            st.progress(value)
        
        # Disclaimer
        st.info("""
        **Disclaimer:** This prediction is for educational purposes only and should not replace 
        professional medical advice. Always consult with a healthcare provider for personal health concerns.
        """)



