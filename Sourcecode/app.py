from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import webbrowser, threading

app = Flask(__name__)

# Load the entire pipeline instead of separate components
# This ensures consistent preprocessing between training and prediction
with open('churn_model_pipeline.pkl', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)
with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/form', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    
    if request.method == 'POST':
        # Collect data from form
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Use the pipeline for prediction (it handles all preprocessing internally)
            prediction_result = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0, 1]
            
            prediction = "Churn" if prediction_result == 1 else "No Churn"
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = None
    
    return render_template('index.html', prediction=prediction, probability=probability)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, use_reloader=False)