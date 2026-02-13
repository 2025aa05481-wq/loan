import pandas as pd
import opendatasets as od
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LoanPredictionDataLoader:
    def __init__(self):
        self.data = None
        self.file_path = 'loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv'
        
    def download_dataset(self):
        """Download the loan prediction dataset from Kaggle"""
        try:
            print("Downloading loan prediction dataset...")
            od.download("https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/")
            print("Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False
    
    def load_data(self, download_if_missing=True):
        """Load the loan prediction dataset"""
        if not os.path.exists(self.file_path):
            if download_if_missing:
                if not self.download_dataset():
                    raise FileNotFoundError("Could not download dataset")
            else:
                raise FileNotFoundError("Dataset file not found")
        
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully!")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def preprocess_data(self):
        """Preprocess the loan prediction dataset"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        # Handle missing values
        # For categorical variables: fill with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        for col in categorical_cols:
            if col in df.columns:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_value)
        
        # For numerical variables: fill with median
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for col in numerical_cols:
            if col in df.columns:
                median_value = df[col].median() if not pd.isna(df[col].median()) else 0
                df[col] = df[col].fillna(median_value)
        
        # Handle 'Dependents' column specifically (replace '3+' with 3)
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace('3+', '3')
            df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
            median_value = df['Dependents'].median() if not pd.isna(df['Dependents'].median()) else 0
            df['Dependents'] = df['Dependents'].fillna(median_value)
        
        # Convert Loan_Status to binary (Y=1, N=0)
        if 'Loan_Status' in df.columns:
            print(f"   Original Loan_Status dtype: {df['Loan_Status'].dtype}")
            print(f"   Original Loan_Status unique values: {df['Loan_Status'].unique()}")
            
            # If values are already 0/1, keep them as is
            if df['Loan_Status'].dtype in [np.int64, np.float64, int, float]:
                # Values are already numeric, just ensure no NaN
                df['Loan_Status'] = df['Loan_Status'].fillna(0)
            else:
                # Convert string values to binary
                df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
                # Fill any remaining NaN values in target
                df['Loan_Status'] = df['Loan_Status'].fillna(0)
            
            print(f"   Processed Loan_Status unique values: {df['Loan_Status'].unique()}")
            print(f"   Processed Loan_Status value counts: {df['Loan_Status'].value_counts()}")
        
        # Encode categorical variables
        label_encoders = {}
        categorical_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        
        for col in categorical_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        # Drop unnecessary columns
        cols_to_drop = ['Loan_ID']  # Loan_ID is just an identifier
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        self.processed_data = df
        self.label_encoders = label_encoders
        
        print(f"Data preprocessing completed. Final shape: {df.shape}")
        return df
    
    def get_feature_target(self, target_column='Loan_Status'):
        """Separate features and target"""
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        X = self.processed_data.drop(columns=[target_column])
        y = self.processed_data[target_column]
        
        return X, y
    
    def get_data_info(self):
        """Get information about the dataset"""
        if self.data is None:
            return "No data loaded"
        
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'target_distribution': self.data['Loan_Status'].value_counts().to_dict() if 'Loan_Status' in self.data.columns else None
        }
        
        return info
