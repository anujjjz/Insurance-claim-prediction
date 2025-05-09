import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from datetime import datetime

def load_and_join_data():
    try:
        # Define file paths for the existing data files
        files = {
            'claims': r"C:\Users\archi\Downloads\FactClaims.csv",
            'date': r"C:\Users\archi\Downloads\DimDate.csv",
            'category': r"C:\Users\archi\Downloads\ClaimCategorySummary.csv",
            'procedure': r"C:\Users\archi\Downloads\DimProcedure.csv",
            'claim_type': r"C:\Users\archi\Downloads\DimClaimType.csv",
            'provider': r"C:\Users\archi\Downloads\DimProvider.csv",
            'member': r"C:\Users\archi\Downloads\DimMember.csv"
        }
        
        # Load all files
        print("Loading data files...")
        data = {}
        for name, path in files.items():
            print(f"Loading {name}...")
            data[name] = pd.read_csv(path)
            print(f"Shape: {data[name].shape}")
            print(f"Columns: {data[name].columns.tolist()}\n")
        
        # Join the tables
        print("Joining tables...")
        
        # Start with claims as the fact table
        df = data['claims']
        
        # Join with date dimension
        if 'DateID' in df.columns and 'DateID' in data['date'].columns:
            df = pd.merge(df, data['date'], on='DateID', how='left')
        
        # Join with category
        if 'CategoryID' in df.columns and 'CategoryID' in data['category'].columns:
            df = pd.merge(df, data['category'], on='CategoryID', how='left')
        
        # Join with procedure
        if 'ProcedureID' in df.columns and 'ProcedureID' in data['procedure'].columns:
            df = pd.merge(df, data['procedure'], on='ProcedureID', how='left')
        
        # Join with claim type
        if 'ClaimTypeID' in df.columns and 'ClaimTypeID' in data['claim_type'].columns:
            df = pd.merge(df, data['claim_type'], on='ClaimTypeID', how='left')
        
        # Join with provider
        if 'ProviderID' in df.columns and 'ProviderID' in data['provider'].columns:
            df = pd.merge(df, data['provider'], on='ProviderID', how='left')
        
        # Join with member (patient) information
        if 'MemberID' in df.columns and 'MemberID' in data['member'].columns:
            df = pd.merge(df, data['member'], on='MemberID', how='left')
        
        print(f"\nFinal joined dataset shape: {df.shape}")
        print("Final columns:", df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"Error in load_and_join_data: {str(e)}")
        raise

def preprocess_data(df):
    try:
        # Create target variable from StatusID
        print("\nCreating target variable from StatusID")
        print("StatusID values:", df['StatusID'].unique())
        print("StatusID distribution:", df['StatusID'].value_counts())
        
        # Map StatusID to binary target (1 for Approved, 0 for Denied)
        status_mapping = {
            1: 1,  # Approved
            2: 0,  # Denied
            3: 0,  # Pending (treat as denied for training)
            4: 0   # Rejected (treat as denied for training)
        }
        df['ClaimApproved'] = df['StatusID'].map(status_mapping)
        print("\nTarget variable distribution:")
        print(df['ClaimApproved'].value_counts(normalize=True))
        
        # Process date columns
        date_columns = ['SubmissionDate', 'ApprovalDate']
        for col in date_columns:
            if col in df.columns:
                print(f"\nProcessing {col}")
                df[col] = pd.to_datetime(df[col])
        
        # Calculate days between submission and approval
        if 'SubmissionDate' in df.columns and 'ApprovalDate' in df.columns:
            print("\nCalculating days to process")
            df['DaysToProcess'] = (df['ApprovalDate'] - df['SubmissionDate']).dt.days
        
        # Calculate patient age if DateOfBirth is available
        if 'DateOfBirth' in df.columns:
            df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'])
            df['Age'] = (pd.Timestamp.now() - df['DateOfBirth']).dt.days / 365.25
        
        # Create chronic condition flags if available
        chronic_conditions = [
            'HasDiabetes', 'HasHypertension', 'HasCancer',
            'HasHeartDisease', 'HasCOPD'
        ]
        for condition in chronic_conditions:
            if condition in df.columns:
                df[condition] = df[condition].fillna(0).astype(int)
        
        # Calculate total chronic conditions if available
        if all(condition in df.columns for condition in chronic_conditions):
            df['ChronicConditionCount'] = df[chronic_conditions].sum(axis=1)
        
        # Drop identifier columns and unnecessary columns
        columns_to_drop = [
            'ProviderID', 'MemberID', 'StatusID', 'DateID', 'CategoryID', 
            'ProcedureID', 'ClaimTypeID', 'SubmissionDate', 'ApprovalDate',
            'DenialReason'  # Drop denial reason as it's highly correlated with the target
        ]
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        print("\nDropping columns:", columns_to_drop)
        df = df.drop(columns=columns_to_drop)
        
        # Handle missing values
        print("\nHandling missing values...")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        print("Numeric columns:", numeric_columns.tolist())
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        print("Categorical columns:", categorical_columns.tolist())
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # Convert categorical variables to dummy variables
        print("\nConverting categorical variables to dummy variables...")
        df = pd.get_dummies(df)
        
        print("\nFinal preprocessed dataset shape:", df.shape)
        print("Final columns:", df.columns.tolist())
        
        return df
    except Exception as e:
        print(f"Error in preprocess_data: {str(e)}")
        raise

def train_models(X, y):
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("\nClass distribution in training set:")
        print(y_train.value_counts(normalize=True))
        print("\nClass distribution in test set:")
        print(y_test.value_counts(normalize=True))
        
        # Initialize models
        print("\nInitializing models...")
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            
            print(f"{name} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            # Update best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} with F1 score {best_score:.4f}")
        
        # Save the best model and feature columns
        model_data = {
            'model': best_model,
            'feature_columns': X.columns.tolist(),
            'class_mapping': {0: 'Denied', 1: 'Approved'}
        }
        
        joblib.dump(model_data, 'insurance_claim_model.joblib')
        print("\nModel and feature columns saved")
        
        return best_model
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting model training process...")
        
        # Load and join data
        print("\nLoading and joining data files...")
        df = load_and_join_data()
        
        # Preprocess data
        print("\nPreprocessing data...")
        df = preprocess_data(df)
        
        # Prepare features and target
        print("\nPreparing features and target...")
        y = df['ClaimApproved']
        X = df.drop('ClaimApproved', axis=1)
        
        # Handle missing values before creating dummy variables
        print("\nHandling missing values...")
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        print("Numeric columns:", numeric_columns.tolist())
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
        
        print("Categorical columns:", categorical_columns.tolist())
        X[categorical_columns] = X[categorical_columns].fillna('Unknown')
        
        # Convert categorical variables to dummy variables
        print("\nConverting categorical variables to dummy variables...")
        X = pd.get_dummies(X)
        
        # Handle any remaining NaN values after dummy conversion
        print("\nHandling any remaining NaN values...")
        print("NaN values before filling:", X.isna().sum().sum())
        X = X.fillna(0)  # Fill any remaining NaN values with 0
        print("NaN values after filling:", X.isna().sum().sum())
        print(f"Final feature matrix shape: {X.shape}")
        
        # Train models
        best_model = train_models(X, y)
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise 