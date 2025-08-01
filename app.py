import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
model = joblib.load('model.pkl')

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #3498db;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .feature-input {
        margin-bottom: 1em;
    }
    .result-box {
        padding: 2em;
        border-radius: 10px;
        margin-top: 2em;
        background-color: #ecf0f1;
    }
    .positive {
        color: #e74c3c;
        font-weight: bold;
    }
    .negative {
        color: #2ecc71;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("🧠 Parkinson's Disease Detection System")
st.markdown("""
    This application uses a Naive Bayes machine learning model to predict the likelihood of Parkinson's disease 
    based on voice measurement features. The model was trained on the [Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification).
    """)
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("About Parkinson's Disease")
    st.markdown("""
    Parkinson's disease is a progressive nervous system disorder that affects movement. 
    Symptoms start gradually, sometimes starting with a barely noticeable tremor in just one hand.
    """)
    
    st.header("How It Works")
    st.markdown("""
    1. Enter the voice measurement parameters
    2. Click the 'Predict' button
    3. View the prediction result
    """)
    
    st.header("Model Information")
    st.markdown("""
    - Algorithm: Gaussian Naive Bayes
    - Accuracy: ~93%
    - Features: 10 voice measurements
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Voice Measurement Features")
    
    # Feature inputs
    mdvp_fo = st.number_input("MDVP:Fo(Hz) - Average vocal fundamental frequency", 
                             min_value=80.0, max_value=400.0, value=150.0, step=0.1)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz) - Maximum vocal fundamental frequency", 
                              min_value=100.0, max_value=600.0, value=200.0, step=0.1)
    mdvp_flo = st.number_input("MDVP:Flo(Hz) - Minimum vocal fundamental frequency", 
                              min_value=50.0, max_value=300.0, value=100.0, step=0.1)
    mdvp_jitter = st.number_input("MDVP:Jitter(%) - Measures of variation in fundamental frequency", 
                                 min_value=0.0, max_value=0.1, value=0.005, step=0.0001, format="%.4f")
    mdvp_shimmer = st.number_input("MDVP:Shimmer - Measures of variation in amplitude", 
                                  min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%.3f")
    
with col2:
    st.subheader("")
    nhr = st.number_input("NHR - Noise-to-harmonics ratio", 
                         min_value=0.0, max_value=0.5, value=0.02, step=0.001, format="%.3f")
    hnr = st.number_input("HNR - Harmonics-to-noise ratio", 
                         min_value=0.0, max_value=30.0, value=20.0, step=0.1)
    rpde = st.number_input("RPDE - Nonlinear dynamical complexity measure", 
                          min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
    dfa = st.number_input("DFA - Signal fractal scaling exponent", 
                         min_value=0.5, max_value=1.0, value=0.7, step=0.001, format="%.3f")
    ppe = st.number_input("PPE - Nonlinear measure of fundamental frequency variation", 
                         min_value=0.0, max_value=0.5, value=0.1, step=0.001, format="%.3f")

# Prediction button
if st.button("Predict Parkinson's Disease"):
    # Create input array
    input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, 
                           mdvp_shimmer, nhr, hnr, rpde, dfa, ppe]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #e74c3c;'>⚠️ Positive for Parkinson's Disease</h3>
            <p>Probability: {prediction_proba[0][1]*100:.2f}%</p>
            <p>This result suggests the presence of Parkinson's disease symptoms based on the voice measurements. 
            Please consult with a healthcare professional for further evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box">
            <h3 style='color: #2ecc71;'>✅ Negative for Parkinson's Disease</h3>
            <p>Probability: {prediction_proba[0][0]*100:.2f}%</p>
            <p>The analysis does not indicate signs of Parkinson's disease based on the provided voice measurements. 
            However, if you have concerns about your health, please consult with a medical professional.</p>
        </div>
        """, unsafe_allow_html=True)

# Additional information
st.markdown("---")
st.subheader("Understanding the Features")
st.markdown("""
The model uses the following voice measurement features to detect Parkinson's disease:

- **MDVP:Fo(Hz)**: Average vocal fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
- **MDVP:Jitter(%)**: Measures of variation in fundamental frequency
- **MDVP:Shimmer**: Measures of variation in amplitude
- **NHR**: Noise-to-harmonics ratio
- **HNR**: Harmonics-to-noise ratio
- **RPDE**: Nonlinear dynamical complexity measure
- **DFA**: Signal fractal scaling exponent
- **PPE**: Nonlinear measure of fundamental frequency variation
""")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for informational purposes only and is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any 
questions you may have regarding a medical condition.
""")