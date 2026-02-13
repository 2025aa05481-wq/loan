#!/usr/bin/env python3
"""
Create PKL Files Script
This script downloads the loan prediction dataset, trains all 6 models,
and saves each model as a separate .pkl file with evaluation metrics
"""

import os
import sys
import pandas as pd
import numpy as np

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from loan_ml_system import LoanPredictionModels

def main():
    print("=" * 80)
    print("CREATING PKL FILES FOR ALL MODELS WITH EVALUATION METRICS")
    print("=" * 80)
    
    try:
        # Initialize the ML system
        print("\n[INFO] Initializing ML System...")
        loan_models = LoanPredictionModels()
        
        # Step 1: Download and load dataset
        print("\n[STEP 1] Downloading and loading dataset...")
        
        # Check if dataset exists, if not download it
        if not os.path.exists('loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv'):
            print("   Dataset not found. Downloading from Kaggle...")
            success = loan_models.data_loader.download_dataset()
            if not success:
                print("[ERROR] Failed to download dataset")
                return False
        else:
            print("   Dataset already exists!")
        
        # Load and display dataset info
        file_path = 'loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv'
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print("\n[INFO] First 5 rows:")
        print(df.head())
        
        # Step 2: Preprocess data
        print("\n[STEP 2] Preprocessing data...")
        processed_data = loan_models.load_and_preprocess_data()
        print(f"[SUCCESS] Data preprocessed successfully!")
        print(f"   Final shape: {processed_data.shape}")
        
        # Step 3: Train all models
        print("\n[STEP 3] Training all 6 classification models...")
        training_results = loan_models.train_models()
        print("[SUCCESS] All models trained successfully!")
        
        # Step 4: Evaluate models with all metrics
        print("\n[STEP 4] Evaluating models with comprehensive metrics...")
        evaluation_results = loan_models.evaluate_models()
        print("[SUCCESS] Model evaluation completed!")
        
        # Step 5: Display detailed results
        print("\n[STEP 5] Model Performance Summary")
        print("=" * 90)
        
        # Get summary dataframe
        summary_df = loan_models.get_evaluation_summary()
        
        # Display detailed metrics table
        print(f"\n{'Model':<20} {'Accuracy':<10} {'AUC Score':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'MCC Score':<10}")
        print("-" * 90)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['AUC Score']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1 Score']:<10.4f} {row['MCC Score']:<10.4f}")
        
        # Step 6: Save models as individual .pkl files
        print("\n[STEP 6] Saving models as individual .pkl files...")
        
        # Ensure model directory exists
        model_dir = 'model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"   Created '{model_dir}' directory")
        
        # Save all models
        loan_models.save_models(directory=model_dir)
        
        # Step 7: List all created files
        print("\n[STEP 7] Created Files Summary")
        print("-" * 40)
        
        pkl_files = []
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(model_dir, file)
                file_size = os.path.getsize(file_path)
                pkl_files.append((file, file_size))
                print(f"   [FILE] {file} ({file_size:,} bytes)")
        
        # Step 8: Save detailed evaluation metrics to CSV
        print("\n[STEP 8] Saving detailed evaluation metrics...")
        
        # Create detailed metrics dataframe
        detailed_metrics = []
        for model_name, results in evaluation_results.items():
            detailed_metrics.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'AUC_Score': results['auc_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'MCC_Score': results['mcc_score'],
                'True_Negatives': results['confusion_matrix'][0][0],
                'False_Positives': results['confusion_matrix'][0][1],
                'False_Negatives': results['confusion_matrix'][1][0],
                'True_Positives': results['confusion_matrix'][1][1]
            })
        
        detailed_df = pd.DataFrame(detailed_metrics)
        detailed_csv_path = os.path.join(model_dir, 'detailed_evaluation_metrics.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"   [SUCCESS] Detailed metrics saved to '{detailed_csv_path}'")
        
        # Step 9: Save summary metrics to CSV
        summary_csv_path = os.path.join(model_dir, 'model_performance_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"   [SUCCESS] Summary metrics saved to '{summary_csv_path}'")
        
        # Step 10: Final summary
        print("\n[STEP 10] Process Completed Successfully!")
        print("=" * 80)
        print("SUMMARY:")
        print(f"• Dataset: Loan Prediction ({df.shape[0]} rows, {df.shape[1]} columns)")
        print(f"• Models trained: {len(loan_models.trained_models)}")
        print(f"• PKL files created: {len(pkl_files)}")
        print(f"• Evaluation metrics: 6 (Accuracy, AUC, Precision, Recall, F1, MCC)")
        print(f"• Files saved in: '{model_dir}/' directory")
        
        # Show best performing model
        best_model, best_score = loan_models.get_best_model('accuracy')
        print(f"• Best model: {best_model} (Accuracy: {best_score:.4f})")
        
        print("\n[INFO] All Required Evaluation Metrics Included:")
        print("   [DONE] 1. Accuracy")
        print("   [DONE] 2. AUC Score") 
        print("   [DONE] 3. Precision")
        print("   [DONE] 4. Recall")
        print("   [DONE] 5. F1 Score")
        print("   [DONE] 6. Matthews Correlation Coefficient (MCC Score)")
        
        print("=" * 80)
        print("[READY] Ready for deployment! You can now run:")
        print("   streamlit run loan_app.py")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during model creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] All PKL files created successfully with evaluation metrics!")
    else:
        print("\n[FAILED] Process failed. Please check the error messages above.")
