## Loan Default Prediction App

This Streamlit application predicts the likelihood of loan default based on user-provided financial and personal information.
It leverages a pre-trained TensorFlow model for efficient predictions.

Deployment

A live version of the app is currently deployed at: https://appappgit-6l5oehywuneuczbnhh5ejv.streamlit.app/

Local Development

To run the app locally, follow these steps:

Prerequisites:

Python 3.6 or later (https://www.python.org/downloads/)
pip package installer (usually included with Python)
Install dependencies:
Open a terminal or command prompt and navigate to your project directory. Then, run:

Bash
pip install streamlit pandas numpy tensorflow
Use code with caution.
Clone or download the project files.

Run the application:
In your terminal, execute:

Bash
streamlit run Streamlitapp.py  # Replace with your main script name if different
Use code with caution.
This will launch the app in your web browser, typically at http://localhost:8501.

## Usage
On the app's main page, you'll find various input fields for user data such as age, income, loan amount, credit score, employment type, and more.
Enter your information into the corresponding fields.
The app automatically update as you enter values.
The app will display the predicted probability of loan default and classify the risk as "High" (above 0.5 probability) or "Low" (below 0.5 probability).

## Disclaimer

This application is for demonstration purposes only and should not be used for making real-world financial decisions. 
The model's accuracy may vary depending on the data used for training.
