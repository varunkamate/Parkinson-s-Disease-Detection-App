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

🧪 Model Details
Algorithm: Gaussian Naive Bayes

Accuracy: ~93%

Model File: model.pkl

Notebook: parkinsons_with_Naive_baye.ipynb (for training and evaluation)

💻 Web Application: Streamlit
🔧 Features
Clean and responsive UI with custom CSS

Sidebar with medical info and instructions

Real-time predictions based on user input

Displays probability and medical disclaimer

Educates users about voice features used

Front streamlit page

<img width="1908" height="1004" alt="Screenshot 2025-08-01 122003" src="https://github.com/user-attachments/assets/fb7a4960-6546-41fc-9f60-83b27b1a93a8" />


Output prediction page
<img width="1896" height="448" alt="Screenshot 2025-08-01 122024" src="https://github.com/user-attachments/assets/0f4ac060-d0b7-461d-8906-3483817f0a1a" />


🚀 Getting Started
🛠 Requirements
bash
Copy code
pip install -r requirements.txt
Or manually install:

bash
Copy code
pip install streamlit pandas numpy joblib scikit-learn
▶️ Run the App
bash
Copy code
streamlit run app.py
