import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add model directory to path for importing loan_ml_system
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from loan_ml_system import LoanPredictionModels

# Set page configuration
st.set_page_config(
    page_title="Loan Prediction ML Models",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2E86AB;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'loan_models' not in st.session_state:
    st.session_state.loan_models = LoanPredictionModels()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Loan Prediction ML Models</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Machine Learning Classification for Loan Approval Prediction</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox("Choose a page:", [
            "Dataset Loading",
            "Model Training",
            "Model Evaluation",
            "Make Predictions",
            "Model Comparison"
        ])
    
    if page == "Dataset Loading":
        dataset_loading_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Make Predictions":
        make_predictions_page()
    elif page == "Model Comparison":
        model_comparison_page()

def dataset_loading_page():
    st.header("📁 Dataset Loading & Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Loan Prediction Dataset")
        st.markdown("""
        <div class="model-card">
            <h4>Dataset Information</h4>
            <p>• Source: Kaggle Loan Prediction Dataset</p>
            <p>• Target: Loan_Status (Y/N)</p>
            <p>• Features: 12 columns including income, credit history, etc.</p>
            <p>• Size: ~614 rows</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Download Dataset", help="Download the loan prediction dataset from Kaggle"):
            with st.spinner("Downloading dataset..."):
                try:
                    # Check if dataset already exists
                    if os.path.exists('loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv'):
                        st.success("✅ Dataset already exists!")
                    else:
                        success = st.session_state.loan_models.data_loader.download_dataset()
                        if success:
                            st.success("✅ Dataset downloaded successfully!")
                        else:
                            st.error("❌ Failed to download dataset")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Load and Preprocess Data")
        
        if st.button("🔄 Load & Preprocess Data", help="Load and preprocess the loan prediction dataset"):
            with st.spinner("Loading and preprocessing data..."):
                try:
                    # Load and preprocess data
                    processed_data = st.session_state.loan_models.load_and_preprocess_data()
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Data loaded and preprocessed successfully!")
                    
                    # Display data info
                    data_info = st.session_state.loan_models.data_loader.get_data_info()
                    
                    # Show dataset preview
                    st.subheader("📊 Dataset Preview")
                    st.dataframe(processed_data.head())
                    
                    # Show basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", data_info['shape'][0])
                    with col2:
                        st.metric("Columns", data_info['shape'][1])
                    with col3:
                        total_missing = sum(data_info['missing_values'].values())
                        st.metric("Missing Values", total_missing)
                    
                    # Show target distribution
                    st.subheader("🎯 Loan Status Distribution")
                    if data_info['target_distribution']:
                        fig = px.pie(
                            values=list(data_info['target_distribution'].values()),
                            names=['Approved (Y)', 'Rejected (N)'],
                            title="Loan Status Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature correlations
                    st.subheader("🔗 Feature Correlations")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = processed_data.corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    plt.title('Feature Correlation Matrix')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")

def model_training_page():
    st.header("🎯 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load and preprocess data first!")
        return
    
    st.subheader("Available Classification Models")
    
    models_info = {
        'Logistic Regression': 'Linear classifier for binary classification',
        'Decision Tree': 'Tree-based classifier with if-then rules',
        'K-Nearest Neighbors': 'Instance-based learning algorithm',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem',
        'Random Forest': 'Ensemble method using multiple decision trees',
        'XGBoost': 'Gradient boosting ensemble method'
    }
    
    # Display model cards
    cols = st.columns(3)
    for i, (model_name, description) in enumerate(models_info.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="model-card">
                <h4>{model_name}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Train models button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Train All Models", type="primary", help="Train all 6 classification models"):
            with st.spinner("Training models... This may take a moment..."):
                try:
                    # Train models
                    results = st.session_state.loan_models.train_models()
                    st.session_state.models_trained = True
                    
                    st.success("✅ All models trained successfully!")
                    
                    # Show training results
                    st.subheader("📊 Training Results")
                    
                    # Create metrics table
                    metrics_data = []
                    for model_name in results.keys():
                        metrics_data.append({
                            'Model': model_name,
                            'Status': '✅ Trained'
                        })
                    
                    st.dataframe(pd.DataFrame(metrics_data))
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
    
    with col2:
        if st.button("💾 Save Models", help="Save trained models to model directory"):
            if st.session_state.models_trained:
                try:
                    st.session_state.loan_models.save_models()
                    st.success("✅ Models saved successfully to 'model' directory!")
                except Exception as e:
                    st.error(f"❌ Error saving models: {str(e)}")
            else:
                st.warning("⚠️ Please train models first!")

def model_evaluation_page():
    st.header("📈 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    # Evaluate models if not already done
    if not st.session_state.loan_models.evaluation_results:
        with st.spinner("Evaluating models..."):
            try:
                st.session_state.loan_models.evaluate_models()
            except Exception as e:
                st.error(f"❌ Error evaluating models: {str(e)}")
                return
    
    # Model selection for detailed evaluation
    available_models = list(st.session_state.loan_models.evaluation_results.keys())
    selected_model = st.selectbox(
        "Select Model for Detailed Evaluation",
        available_models
    )
    
    if selected_model:
        results = st.session_state.loan_models.evaluation_results[selected_model]
        
        # Display all metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
            st.metric("Precision", f"{results['precision']:.4f}")
        
        with col2:
            st.metric("Recall", f"{results['recall']:.4f}")
            st.metric("F1 Score", f"{results['f1_score']:.4f}")
        
        with col3:
            st.metric("MCC Score", f"{results['mcc_score']:.4f}")
            st.metric("AUC Score", f"{results['auc_score']:.4f}" if not np.isnan(results['auc_score']) else "N/A")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = results['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
        
        # ROC Curve (if AUC is available)
        if results['probabilities'] is not None and not np.isnan(results['auc_score']):
            st.subheader("📈 ROC Curve")
            
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(st.session_state.loan_models.y_test, results['probabilities'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {results["auc_score"]:.4f})',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

def make_predictions_page():
    st.header("🔮 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    # Select model for prediction
    available_models = list(st.session_state.loan_models.trained_models.keys())
    selected_model = st.selectbox(
        "Select Model for Prediction",
        available_models
    )
    
    if selected_model:
        st.subheader(f"🎯 Using {selected_model}")
        
        # Get feature names from the training data
        feature_names = st.session_state.loan_models.X_train.columns.tolist()
        
        # Create input fields for each feature
        st.subheader("📝 Input Applicant Information")
        
        input_data = {}
        cols = st.columns(2)
        
        # Feature descriptions for loan prediction
        feature_descriptions = {
            'Gender': 'Gender (0=Female, 1=Male)',
            'Married': 'Marital Status (0=No, 1=Yes)',
            'Dependents': 'Number of Dependents',
            'Education': 'Education (0=Graduate, 1=Not Graduate)',
            'Self_Employed': 'Self Employed (0=No, 1=Yes)',
            'ApplicantIncome': 'Applicant Income',
            'CoapplicantIncome': 'Coapplicant Income',
            'LoanAmount': 'Loan Amount (in thousands)',
            'Loan_Amount_Term': 'Loan Term (in months)',
            'Credit_History': 'Credit History (0=Bad, 1=Good)',
            'Property_Area': 'Property Area (0=Rural, 1=Semiurban, 2=Urban)'
        }
        
        for i, feature in enumerate(feature_names):
            with cols[i % 2]:
                description = feature_descriptions.get(feature, feature)
                
                if feature in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
                    input_data[feature] = st.number_input(
                        f"{description}",
                        value=0.0,
                        step=100.0 if feature in ['ApplicantIncome', 'CoapplicantIncome'] else 1.0,
                        key=f"input_{feature}"
                    )
                elif feature in ['Dependents']:
                    input_data[feature] = st.number_input(
                        f"{description}",
                        value=0,
                        min=0,
                        max=10,
                        step=1,
                        key=f"input_{feature}"
                    )
                elif feature in ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History']:
                    input_data[feature] = st.selectbox(
                        f"{description}",
                        options=[0, 1],
                        format_func=lambda x: 'No' if x == 0 else 'Yes',
                        key=f"input_{feature}"
                    )
                elif feature == 'Property_Area':
                    input_data[feature] = st.selectbox(
                        f"{description}",
                        options=[0, 1, 2],
                        format_func=lambda x: ['Rural', 'Semiurban', 'Urban'][x],
                        key=f"input_{feature}"
                    )
        
        # Make prediction button
        if st.button("🚀 Make Prediction", type="primary"):
            try:
                prediction_result = st.session_state.loan_models.predict(selected_model, input_data)
                
                # Display prediction results
                prediction = prediction_result['prediction']
                probability = prediction_result['probability']
                
                # Determine result
                loan_status = "Approved ✅" if prediction == 1 else "Rejected ❌"
                confidence = probability[1] if prediction == 1 else probability[0]
                
                st.markdown(f"""
                <div class="success-message">
                    <h3>Loan Status: {loan_status}</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probability breakdown
                st.subheader("📊 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Status': ['Rejected', 'Approved'],
                    'Probability': probability
                })
                
                fig = px.bar(
                    prob_df,
                    x='Status',
                    y='Probability',
                    title="Loan Status Probabilities",
                    color='Probability',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")

def model_comparison_page():
    st.header("⚖️ Model Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
        return
    
    # Get evaluation summary
    summary_df = st.session_state.loan_models.get_evaluation_summary()
    
    # Display comparison table
    st.subheader("📊 Performance Comparison")
    st.dataframe(summary_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = px.bar(
            summary_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Multi-metric comparison
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                x=summary_df['Model'],
                y=summary_df[metric],
                name=metric,
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title="Multi-Metric Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # MCC and AUC comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            summary_df,
            x='Model',
            y='MCC Score',
            title="Matthews Correlation Coefficient",
            color='MCC Score',
            color_continuous_scale='plasma'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Filter out NaN AUC scores
        auc_df = summary_df.dropna(subset=['AUC Score'])
        if not auc_df.empty:
            fig = px.bar(
                auc_df,
                x='Model',
                y='AUC Score',
                title="AUC Score Comparison",
                color='AUC Score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No AUC scores available")
    
    # Best model recommendation
    best_model_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_model_f1 = summary_df.loc[summary_df['F1 Score'].idxmax()]
    
    st.subheader("🏆 Best Model Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best by Accuracy</h4>
            <h3>{best_model_accuracy['Model']}</h3>
            <p>Accuracy: {best_model_accuracy['Accuracy']:.4f}</p>
            <p>F1 Score: {best_model_accuracy['F1 Score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best by F1 Score</h4>
            <h3>{best_model_f1['Model']}</h3>
            <p>F1 Score: {best_model_f1['F1 Score']:.4f}</p>
            <p>Accuracy: {best_model_f1['Accuracy']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
