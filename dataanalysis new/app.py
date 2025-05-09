from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Power BI Embed URL
POWERBI_REPORT_URL = "https://app.powerbi.com/reportEmbed?reportId=cde0a06b-afd0-4724-924d-e06f27378250&autoAuth=true&ctid=d8a63e7a-515b-414d-ae44-9febcfb99c8b"

# Global variables for model and features
model = None
feature_columns = None
class_mapping = None

def load_model():
    global model, feature_columns, class_mapping
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'insurance_claim_model.joblib')
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return False
            
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        class_mapping = model_data['class_mapping']
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Class mapping: {class_mapping}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Load model at startup
if not load_model():
    print("Failed to load model. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html', powerbi_url=POWERBI_REPORT_URL)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
    try:
        data = request.json
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]
        
        # Map prediction to class name
        result = class_mapping[prediction]
        
        print(f"Prediction: {result}, Confidence: {confidence:.2f}")
        return jsonify({
            'prediction': result,
            'confidence': float(confidence)
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Add current date for submission and approval dates
    current_date = datetime.now().strftime('%Y-%m-%d')
    df['SubmissionDate'] = current_date
    df['ApprovalDate'] = current_date
    
    # Process date columns
    df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
    df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'])
    df['DaysToProcess'] = (df['ApprovalDate'] - df['SubmissionDate']).dt.days
    
    # Initialize numeric columns with default values
    numeric_columns = ['ClaimAmount', 'ApprovedAmount', 'DeniedAmount', 'CoPayAmount', 
                      'DeductibleAmount', 'ReimbursedAmount', 'YearsInPractice']
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Handle categorical variables with proper prefixing
    categorical_mappings = {
        'memberGender': ['memberGender_Male', 'memberGender_Female'],
        'category': ['Category_Cardiac', 'Category_Gastro', 'Category_Neuro', 
                    'Category_Ortho', 'Category_Other', 'Category_Pulmonary'],
        'claimType': ['ClaimType_Inpatient', 'ClaimType_Outpatient', 'ClaimType_Pharmacy'],
        'providerType': ['Type_Unknown']  # Default to Unknown if not specified
    }
    
    # Create dummy variables with proper prefixes
    for col, possible_values in categorical_mappings.items():
        if col in df.columns:
            value = df[col].iloc[0]
            for possible_value in possible_values:
                df[possible_value] = 0
            # Set the appropriate column to 1
            matching_col = f"{col}_{value}"
            if matching_col in possible_values:
                df[matching_col] = 1
            else:
                # If value doesn't match any known category, set to Unknown
                df[f"{col}_Unknown"] = 1
    
    # Ensure all training features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features used in training
    return df[feature_columns]

if __name__ == '__main__':
    app.run(debug=True) 