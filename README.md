# Medical Insurance Claim Prediction System

This web application predicts whether a medical insurance claim will be approved or denied based on various factors such as patient demographics, medical procedures, diagnosis codes, and chronic conditions.

## Project Structure

```
.
├── app.py                 # Flask backend
├── model_training.py      # Model training script
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css         # CSS styles
├── templates/
│   └── index.html        # Frontend template
└── README.md             # This file
```

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - The application expects CSV files with the following data:
     - MedicalClaims.csv: Contains claim details, amounts, and status
     - PatientInfo.csv: Contains patient demographics and chronic conditions
     - ProviderInfo.csv: Contains provider details and specialties
     - MedicalProcedures.csv: Contains procedure codes and descriptions
     - DiagnosisCodes.csv: Contains diagnosis codes and descriptions
   - Required columns in MedicalClaims.csv:
     - ClaimID, PatientID, ProviderID, ProcedureCode, DiagnosisCode
     - ClaimAmount, ClaimStatus (Approved/Denied/Pending/Rejected)
     - ServiceDate, SubmissionDate, ApprovalDate
   - Required columns in PatientInfo.csv:
     - PatientID, DateOfBirth, Gender
     - HasDiabetes, HasHypertension, HasCancer, HasHeartDisease, HasCOPD
   - Required columns in ProviderInfo.csv:
     - ProviderID, ProviderType, Specialty
   - Required columns in MedicalProcedures.csv:
     - ProcedureCode, ProcedureDescription, Category
   - Required columns in DiagnosisCodes.csv:
     - DiagnosisCode, DiagnosisDescription, Category

4. Train the model:
   ```bash
   python model_training.py
   ```
   This will train the models and save the best performing model as `medical_insurance_model.joblib`

5. Run the Flask application:
   ```bash
   python app.py
   ```

6. Access the web application:
   - Open your web browser and navigate to `http://localhost:5000`
   - Fill out the form with the required information
   - Click "Predict Claim Status" to get the prediction

## Features

- Clean and responsive web interface
- Real-time prediction without page reload
- Handles both numeric and categorical inputs
- Preprocesses data consistently with the training pipeline
- Displays prediction results with clear formatting
- Includes chronic condition tracking
- Supports medical procedure and diagnosis codes

## Model Details

The application uses a machine learning pipeline that:
- Preprocesses the data (scaling, encoding)
- Handles medical-specific features (procedures, diagnoses, chronic conditions)
- Trains both Random Forest and Logistic Regression models
- Selects the best performing model based on F1 score
- Handles missing values and outliers
- Processes date features automatically

## Notes

- The model will be saved as `medical_insurance_model.joblib` after training
- The web application runs in debug mode by default
- For production use, consider:
  - Disabling debug mode
  - Adding proper error handling
  - Implementing user authentication
  - Adding input validation
  - Using a production-grade web server 
