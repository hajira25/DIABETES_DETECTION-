# Diabetes Detection Using Streamlit

## Overview
This project utilizes machine learning random forest algorithm to predict diabetes outcomes based on patient data.
It features a user-friendly web interface built with Streamlit Community Cloud, allowing users to input their health metrics and receive instant visualized predictions.

### Supervised by 
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu) ,
(Assistant Professor) Department of CSE, MIT Mysore

## Website

[Diabetes-Detection](https://diabetesdetection-ss.streamlit.app/)

## Project Structure and Description  

- Diabetes Detection/
│
├── app.py                          # Main application file
├── train_model.py                  # Script to train the model
├── dataset/
│   └── diabetes.csv                # Dataset for training
├── templates/
│   ├── index.html                  # Input form
│   └── results.html                # Results display
├── models/
│   ├── diabetes_detection_model.h5  # Pre-trained model .h5 extension
│   └── rf_model.pkl                # Random Forest model
└── requirements.txt                # Dependencies

## Prerequisites

- Python 3.12
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- TensorFlow (if using .h5 model)
- HTML/CSS for frontend design

## Installation

###Step 1: Clone the Repository
Clone the repository to your local machine:

git clone <repository_url>
cd Diabetes Detection

###Step 2: Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment:

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
### Step 3: Install Dependencies
Install the required libraries:
'''bash
 pip install -r requirements.txt

### Step 4: Run the Application
1. **Ensure you're in the project directory.**

2. **Run the Streamlit app:**

   ```bash
   streamlit run app.py

## Usage

*   Input patient data using the sliders in the sidebar.
*   Click the "Submit" button to get predictions and visualizations.
*   The app will display the patient's data along with the prediction and accuracy of the model.

## Hosting 
Hosting is done using RENDER.


## Conclusion

The Diabetes Detection application leverages machine learning to provide insights into an individual's diabetes risk based on their health data. By using a Random Forest Classifier, the model can accurately predict outcomes, empowering users to make informed health decisions. This tool not only aids in early detection but also promotes awareness and proactive health management.

Diabetes Detection Using Streamlit
Overview
This project leverages a machine learning Random Forest Classifier to predict diabetes outcomes based on patient data. It features a user-friendly web interface built with Streamlit, allowing users to input health metrics and receive instant visualized predictions.

Supervised by
Prof. Agughasi Victor Ikechukwu
(Assistant Professor, Department of CSE, MIT Mysore)

Website
Diabetes Detection

Project Structure and Description
plaintext
Copy code
Diabetes-Detection/
│
├── app.py                          # Main Streamlit application file
├── train_model.py                  # Script to train the Random Forest model
├── dataset/
│   └── diabetes.csv                # Dataset used for training and testing
├── templates/
│   ├── index.html                  # Input form for patient data
│   └── results.html                # Results display with predictions
├── models/
│   ├── diabetes_detection_model.h5 # Pre-trained deep learning model (optional)
│   └── rf_model.pkl                # Random Forest machine learning model
└── requirements.txt                # Python dependencies
Prerequisites
Python 3.12 or higher
Libraries:
Streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
TensorFlow (if using the .h5 model)
Basic knowledge of HTML/CSS for frontend enhancements (optional)
Installation
Step 1: Clone the Repository
Clone the repository to your local machine:

bash
Copy code
git clone <repository_url>
cd Diabetes-Detection
Step 2: Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment:

bash
Copy code
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Step 4: Run the Application
Navigate to the project directory.
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Usage
Input Data: Enter the patient’s health metrics (e.g., glucose levels, BMI) using the sliders in the sidebar.
Predict: Click the "Submit" button to generate predictions and visualizations.
View Results: The app will display:
The patient's data.
The prediction outcome (e.g., diabetes risk).
The model's accuracy.
Hosting
This application is hosted using Render for seamless access.

Conclusion
The Diabetes Detection Tool empowers users to assess their diabetes risk based on health metrics. Using a Random Forest Classifier, the model delivers accurate predictions and promotes proactive health management. This application fosters awareness and supports early detection, helping individuals make informed health decisions.
