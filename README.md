# Parkinson-s-Disease-Detection-App
🎯 Project Objective This project aims to detect the presence of Parkinson's Disease using voice measurements and a Naive Bayes Machine Learning model. The project includes a fully interactive Streamlit web app to make predictions and explore the disease's indicators.

🔍 Dataset Overview
The model was trained on the Parkinson's Disease Dataset from the UCI Machine Learning Repository. The dataset contains voice measurements from both healthy individuals and those with Parkinson's.

Filename: parkinsons.csv

Features used for prediction:

Feature	Description
MDVP:Fo(Hz)	Average vocal fundamental frequency
MDVP:Fhi(Hz)	Maximum vocal fundamental frequency
MDVP:Flo(Hz)	Minimum vocal fundamental frequency
MDVP:Jitter(%)	Variation in frequency
MDVP:Shimmer	Variation in amplitude
NHR	Noise-to-harmonics ratio
HNR	Harmonics-to-noise ratio
RPDE	Nonlinear dynamical complexity measure
DFA	Fractal scaling exponent
PPE	Fundamental frequency variation
