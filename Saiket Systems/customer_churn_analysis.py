"""
Customer Churn Analysis and Prediction
=====================================

Creator: Harikesh Das (SKS/A2/C48188)
Date: 12/09/2025
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Try to use interactive backend, fallback to non-interactive if needed
try:
    matplotlib.use('TkAgg')  # Use interactive backend for displaying plots
    INTERACTIVE_MODE = True
    print("Using interactive backend for plot display")
except:
    matplotlib.use('Agg')  # Fallback to non-interactive backend
    INTERACTIVE_MODE = False
    print("Using non-interactive backend - plots will only be saved to files")
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CustomerChurnAnalyzer:
    """Main class for customer churn analysis and prediction."""
    
    def __init__(self, data_path):
        """Initialize the analyzer with data path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset and display basic information."""
        print("=" * 60)
        print("TASK 1: DATA PREPARATION")
        print("=" * 60)
        
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Display basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Check for missing values
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Check data types
        print("\nData Types:")
        print(self.df.dtypes)
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n" + "=" * 40)
        print("DATA PREPROCESSING")
        print("=" * 40)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Handle missing values in TotalCharges (convert to numeric first)
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
        
        # Remove customerID as it's not useful for prediction
        df_processed = df_processed.drop('customerID', axis=1)
        
        # Encode categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Churn distribution:\n{y.value_counts()}")
        print(f"Churn percentage: {y.mean() * 100:.2f}%")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def split_data(self):
        """Split data into training and testing sets."""
        print("\n" + "=" * 60)
        print("TASK 2: DATA SPLITTING")
        print("=" * 60)
        
        # Split the data (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Training set churn rate: {self.y_train.mean() * 100:.2f}%")
        print(f"Testing set churn rate: {self.y_test.mean() * 100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_analysis(self):
        """Analyze and select relevant features."""
        print("\n" + "=" * 60)
        print("TASK 3: FEATURE SELECTION")
        print("=" * 60)
        
        # Feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance (Random Forest):")
        print(feature_importance)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save and display the plot
        try:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"Error saving feature importance plot: {e}")
        
        plt.show()  # Display the plot
        if INTERACTIVE_MODE:
            if INTERACTIVE_MODE:
                input("Press Enter to continue to the next plot...")  # Pause for user
        plt.close()  # Close the figure after displaying
        
        # Select top features (you can adjust this threshold)
        top_features = feature_importance.head(10)['feature'].tolist()
        print(f"\nSelected top 10 features: {top_features}")
        
        # Update training and testing sets with selected features
        self.X_train_selected = self.X_train[top_features]
        self.X_test_selected = self.X_test[top_features]
        self.X_train_selected_scaled = self.scaler.fit_transform(self.X_train_selected)
        self.X_test_selected_scaled = self.scaler.transform(self.X_test_selected)
        
        return top_features
    
    def model_selection(self):
        """Select and train multiple models."""
        print("\n" + "=" * 60)
        print("TASK 4 & 5: MODEL SELECTION AND TRAINING")
        print("=" * 60)
        
        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                model.fit(self.X_train_selected_scaled, self.y_train)
                y_pred = model.predict(self.X_test_selected_scaled)
                y_pred_proba = model.predict_proba(self.X_test_selected_scaled)[:, 1]
            else:
                model.fit(self.X_train_selected, self.y_train)
                y_pred = model.predict(self.X_test_selected)
                y_pred_proba = model.predict_proba(self.X_test_selected)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return self.results
    
    def model_evaluation(self):
        """Evaluate model performance comprehensively."""
        print("\n" + "=" * 60)
        print("TASK 6: MODEL EVALUATION")
        print("=" * 60)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1-Score': [self.results[model]['f1'] for model in self.results.keys()],
            'ROC-AUC': [self.results[model]['roc_auc'] for model in self.results.keys()]
        })
        
        print("Model Performance Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        best_model = self.results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1-Score: {results_df['F1-Score'].max():.4f}")
        
        # Create separate plots to avoid complexity issues
        try:
            # Plot 1: Model Performance Comparison
            plt.figure(figsize=(10, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            x = np.arange(len(metrics))
            width = 0.2
            
            for i, model in enumerate(self.results.keys()):
                values = []
                for metric in metrics:
                    if metric == 'F1-Score':
                        values.append(self.results[model]['f1'])
                    else:
                        values.append(self.results[model][metric.lower().replace('-', '_')])
                plt.bar(x + i*width, values, width, label=model)
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width*1.5, metrics, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('model_performance.png', dpi=200, bbox_inches='tight')
            plt.show()  # Display the plot
            if INTERACTIVE_MODE:
                input("Press Enter to continue to the next plot...")  # Pause for user
            plt.close()
            print("Model performance plot saved as 'model_performance.png'")
            
            # Plot 2: Confusion Matrix
            plt.figure(figsize=(6, 5))
            cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
            plt.show()  # Display the plot
            if INTERACTIVE_MODE:
                input("Press Enter to continue to the next plot...")  # Pause for user
            plt.close()
            print("Confusion matrix plot saved as 'confusion_matrix.png'")
            
            # Plot 3: ROC Curve
            plt.figure(figsize=(8, 6))
            for model_name, result in self.results.items():
                fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.3f})")
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('roc_curves.png', dpi=200, bbox_inches='tight')
            plt.show()  # Display the plot
            if INTERACTIVE_MODE:
                input("Press Enter to continue to the next plot...")  # Pause for user
            plt.close()
            print("ROC curves plot saved as 'roc_curves.png'")
            
            # Plot 4: Feature Importance for best model
            if hasattr(best_model, 'feature_importances_'):
                plt.figure(figsize=(8, 6))
                feature_importance = pd.DataFrame({
                    'feature': self.X_train_selected.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                plt.barh(feature_importance['feature'], feature_importance['importance'])
                plt.title(f'Feature Importance - {best_model_name}')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig('best_model_features.png', dpi=200, bbox_inches='tight')
                plt.show()  # Display the plot
                if INTERACTIVE_MODE:
                    input("Press Enter to continue...")  # Pause for user
                plt.close()
                print("Best model feature importance plot saved as 'best_model_features.png'")
            
            print(f"\nAll model evaluation plots saved successfully!")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Continuing with analysis...")
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(self.y_test, self.results[best_model_name]['predictions']))
        
        return best_model_name, results_df
    
    def generate_insights(self):
        """Generate business insights and recommendations."""
        print("\n" + "=" * 60)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("=" * 60)
        
        # Analyze the original dataset for insights
        print("Key Insights from Data Analysis:")
        print("-" * 40)
        
        # Churn rate by contract type
        contract_churn = self.df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        print("\n1. Churn Rate by Contract Type:")
        print(contract_churn)
        
        # Churn rate by tenure groups
        self.df['tenure_group'] = pd.cut(self.df['tenure'], bins=[0, 12, 24, 48, 72], 
                                        labels=['0-12', '13-24', '25-48', '49+'])
        tenure_churn = self.df.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack()
        print("\n2. Churn Rate by Tenure Group:")
        print(tenure_churn)
        
        # Monthly charges analysis
        churn_charges = self.df.groupby('Churn')['MonthlyCharges'].describe()
        print("\n3. Monthly Charges Analysis:")
        print(churn_charges)
        
       
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("CUSTOMER CHURN ANALYSIS AND PREDICTION")
        print("=" * 60)
        if INTERACTIVE_MODE:
            print("📊 INTERACTIVE MODE: All plots will be displayed during execution")
            print("💡 Press Enter after each plot to continue to the next one")
        else:
            print("📁 FILE MODE: Plots will be saved to files only")
            print("💡 Check the generated PNG files for visualizations")
        print("=" * 60)
        
        # Task 1: Data Preparation
        self.load_data()
        self.preprocess_data()
        
        # Task 2: Data Splitting
        self.split_data()
        
        # Task 3: Feature Selection
        self.feature_analysis()
        
        # Task 4 & 5: Model Selection and Training
        self.model_selection()
        
        # Task 6: Model Evaluation
        best_model, results_df = self.model_evaluation()
        
        # Generate insights
        self.generate_insights()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Best performing model: {best_model}")
        
        # Check if plots were created
        import os
        plots_created = []
        plot_files = [
            'feature_importance.png',
            'model_performance.png', 
            'confusion_matrix.png',
            'roc_curves.png',
            'best_model_features.png'
        ]
        
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                plots_created.append(plot_file)
        
        if plots_created:
            print(f"Generated plots: {', '.join(plots_created)}")
        else:
            print("Warning: No plots were generated. Check for matplotlib backend issues.")
        
        return best_model, results_df

def main():
    """Main function to run the analysis."""
    # Initialize the analyzer
    analyzer = CustomerChurnAnalyzer('Telco_Customer_Churn_Dataset  (1).csv')
    
    # Run complete analysis
    best_model, results = analyzer.run_complete_analysis()
    
    return analyzer, best_model, results

if __name__ == "__main__":
    analyzer, best_model, results = main()



