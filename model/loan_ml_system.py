#!/usr/bin/env python3
"""
Loan Prediction ML System
Combined file containing LoanPredictionModels class and training script
This file contains all ML models, data processing, and training functionality
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import xgboost as xgb
import joblib
import os
import sys

# Add parent directory to path for importing data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import LoanPredictionDataLoader

class LoanPredictionModels:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.evaluation_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data_loader = LoanPredictionDataLoader()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the loan prediction dataset"""
        print("Loading and preprocessing loan prediction dataset...")
        
        # Load data
        data = self.data_loader.load_data()
        
        # Preprocess data
        processed_data = self.data_loader.preprocess_data()
        
        # Get features and target
        X, y = self.data_loader.get_feature_target()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data loaded and preprocessed. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        return processed_data
    
    def train_models(self):
        """Train all classification models"""
        print("Training all models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Determine whether to use scaled features
            if name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
                # These models benefit from scaled features
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                # Tree-based models don't need scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Store trained model
            self.trained_models[name] = model
            
            # Store results for evaluation
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        print("All models trained successfully!")
        return results
    
    def evaluate_models(self):
        """Evaluate all trained models with comprehensive metrics"""
        print("Evaluating models...")
        
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            if name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate all evaluation metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='binary')
            recall = recall_score(self.y_test, y_pred, average='binary')
            f1 = f1_score(self.y_test, y_pred, average='binary')
            mcc = matthews_corrcoef(self.y_test, y_pred)
            
            # Calculate AUC score
            if y_pred_proba is not None:
                try:
                    auc_score = roc_auc_score(self.y_test, y_pred_proba)
                except ValueError:
                    auc_score = np.nan
            else:
                auc_score = np.nan
            
            # Generate classification report
            class_report = classification_report(self.y_test, y_pred, output_dict=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Store all results
            self.evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc_score': mcc,
                'auc_score': auc_score,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        print("Model evaluation completed!")
        return self.evaluation_results
    
    def get_evaluation_summary(self):
        """Get a summary of all evaluation metrics"""
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_models() first."
        
        summary = []
        for model_name, results in self.evaluation_results.items():
            summary.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'MCC Score': results['mcc_score'],
                'AUC Score': results['auc_score']
            })
        
        return pd.DataFrame(summary)
    
    def save_models(self, directory='model'):
        """Save all trained models to the specified directory as separate .pkl files"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        print(f"Saving models to {directory} directory...")
        
        for name, model in self.trained_models.items():
            # Create safe filename with .pkl extension
            filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
            filepath = os.path.join(directory, filename)
            
            # Save the model as .pkl file
            joblib.dump(model, filepath)
            print(f"Saved {name} to {filepath}")
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.pkl'))
        
        # Save evaluation results
        joblib.dump(self.evaluation_results, os.path.join(directory, 'evaluation_results.pkl'))
        
        print("All models and components saved successfully as .pkl files!")
    
    def load_models(self, directory='model'):
        """Load trained models from the specified directory"""
        print(f"Loading models from {directory} directory...")
        
        # Load scaler
        scaler_path = os.path.join(directory, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load evaluation results
        eval_path = os.path.join(directory, 'evaluation_results.pkl')
        if os.path.exists(eval_path):
            self.evaluation_results = joblib.load(eval_path)
        
        # Load models
        model_files = {
            'logistic_regression.pkl': 'Logistic Regression',
            'decision_tree.pkl': 'Decision Tree',
            'k-nearest_neighbors.pkl': 'K-Nearest Neighbors',
            'naive_bayes.pkl': 'Naive Bayes',
            'random_forest.pkl': 'Random Forest',
            'xgboost.pkl': 'XGBoost'
        }
        
        for filename, model_name in model_files.items():
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                self.trained_models[model_name] = joblib.load(filepath)
                print(f"Loaded {model_name}")
        
        print("Models loaded successfully!")
    
    def predict(self, model_name, input_data):
        """Make predictions using a specific model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        
        # Preprocess input data
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Scale if necessary
        if model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'SVM']:
            input_scaled = self.scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'class_names': model.classes_.tolist()
        }
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model based on specified metric"""
        if not self.evaluation_results:
            return None
        
        best_model = None
        best_score = -np.inf
        
        for model_name, results in self.evaluation_results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score


def main():
    """Main training script function"""
    print("=" * 60)
    print("LOAN PREDICTION - ML MODELS TRAINING")
    print("=" * 60)
    
    # Initialize the models
    loan_models = LoanPredictionModels()
    
    try:
        # Step 1: Load and preprocess data
        print("\n📁 Step 1: Loading and preprocessing data...")
        processed_data = loan_models.load_and_preprocess_data()
        print(f"✅ Data loaded successfully! Shape: {processed_data.shape}")
        
        # Show data info
        print("\n📊 Dataset Information:")
        print(f"   - Total rows: {processed_data.shape[0]}")
        print(f"   - Total columns: {processed_data.shape[1]}")
        print(f"   - Target column: Loan_Status")
        print(f"   - Missing values: {processed_data.isnull().sum().sum()}")
        
        # Show target distribution
        target_dist = processed_data['Loan_Status'].value_counts()
        print(f"\n🎯 Target Distribution:")
        print(f"   - Approved (1): {target_dist.get(1, 0)}")
        print(f"   - Rejected (0): {target_dist.get(0, 0)}")
        
        # Step 2: Train all models
        print("\n🎯 Step 2: Training all 6 classification models...")
        training_results = loan_models.train_models()
        print("✅ All models trained successfully!")
        
        # Step 3: Evaluate models
        print("\n📈 Step 3: Evaluating models with comprehensive metrics...")
        evaluation_results = loan_models.evaluate_models()
        print("✅ Model evaluation completed!")
        
        # Step 4: Display results
        print("\n📊 Step 4: Model Performance Summary")
        print("=" * 80)
        
        # Get summary dataframe
        summary_df = loan_models.get_evaluation_summary()
        
        # Format the summary for better display
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'MCC':<10} {'AUC':<10}")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1 Score']:<10.4f} {row['MCC Score']:<10.4f} {row['AUC Score']:<10.4f}")
        
        # Step 5: Find best models
        print("\n🏆 Best Performing Models:")
        print("-" * 40)
        
        best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
        best_f1 = summary_df.loc[summary_df['F1 Score'].idxmax()]
        best_mcc = summary_df.loc[summary_df['MCC Score'].idxmax()]
        
        print(f"Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        print(f"Best F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
        print(f"Best MCC Score: {best_mcc['Model']} ({best_mcc['MCC Score']:.4f})")
        
        # Step 6: Save models
        print("\n💾 Step 5: Saving models to 'model' directory...")
        
        # Create model directory if it doesn't exist
        if not os.path.exists('model'):
            os.makedirs('model')
            print("   Created 'model' directory")
        
        # Save all models
        loan_models.save_models()
        
        print("\n✅ Training completed successfully!")
        print("=" * 60)
        print("SUMMARY:")
        print(f"• 6 models trained and evaluated")
        print(f"• Models saved to 'model' directory")
        print(f"• Evaluation metrics: Accuracy, Precision, Recall, F1, MCC, AUC")
        print(f"• Best model: {best_accuracy['Model']} (Accuracy: {best_accuracy['Accuracy']:.4f})")
        print("=" * 60)
        
        # Save summary to CSV
        summary_df.to_csv('model/model_performance_summary.csv', index=False)
        print("📄 Performance summary saved to 'model/model_performance_summary.csv'")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! You can now run the Streamlit app with:")
        print("   streamlit run loan_app.py")
    else:
        print("\n💥 Training failed. Please check the error message above.")
