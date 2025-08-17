"""
AlphaCare Insurance Solutions - Machine Learning Modeling Module

This module contains all machine learning utilities for the insurance analytics project.
Provides comprehensive modeling capabilities including feature engineering, model training,
evaluation, and interpretability analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')


class InsuranceModelingPipeline:
    """
    Comprehensive machine learning pipeline for AlphaCare Insurance Analytics.
    
    Provides end-to-end modeling capabilities:
    - Data preprocessing and feature engineering
    - Multiple model training and comparison
    - Model evaluation and validation
    - Feature importance and SHAP analysis
    - Model persistence and deployment
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the modeling pipeline.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.feature_names = []
        self.target_name = ""
        self.scaler = None
        self.encoders = {}
        
    def prepare_features(self, df: pd.DataFrame, target_column: str, 
                        feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            feature_columns (List[str]): Specific features to use. If None, use all except target.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Validate feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        # Store feature names and target name
        self.feature_names = list(X.columns)
        self.target_name = target_column
        
        print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        X_clean = X.copy()
        
        # Numerical columns - fill with median
        numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X_clean[col].isnull().sum() > 0:
                X_clean[col].fillna(X_clean[col].median(), inplace=True)
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = X_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X_clean[col].isnull().sum() > 0:
                mode_value = X_clean[col].mode()
                if len(mode_value) > 0:
                    X_clean[col].fillna(mode_value[0], inplace=True)
                else:
                    X_clean[col].fillna('Unknown', inplace=True)
        
        return X_clean
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for machine learning."""
        X_encoded = X.copy()
        
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_values = X_encoded[col].nunique()
            
            if unique_values <= 10:  # One-hot encode low cardinality
                dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded.drop(col, axis=1, inplace=True)
                
                # Store encoder info
                self.encoders[col] = {
                    'type': 'one_hot',
                    'categories': list(dummies.columns)
                }
                
            else:  # Label encode high cardinality
                le = LabelEncoder()
                X_encoded[f'{col}_encoded'] = le.fit_transform(X_encoded[col].astype(str))
                X_encoded.drop(col, axis=1, inplace=True)
                
                # Store encoder
                self.encoders[col] = {
                    'type': 'label',
                    'encoder': le
                }
        
        return X_encoded
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            
        Returns:
            Dict: Dictionary containing all data splits
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"📊 Data split completed:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        return data_splits
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """
        Train Linear Regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            Dict: Model and performance metrics
        """
        print("🔄 Training Linear Regression...")
        
        # Train model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = lr_model.predict(X_train)
        results = {
            'model': lr_model,
            'model_name': 'Linear Regression',
            'train_predictions': y_train_pred
        }
        
        # Training metrics
        results['train_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = lr_model.predict(X_val)
            results['val_predictions'] = y_val_pred
            results['val_metrics'] = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'r2': r2_score(y_val, y_val_pred)
            }
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': lr_model.coef_,
            'abs_coefficient': np.abs(lr_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['linear_regression'] = results
        print("✅ Linear Regression training completed")
        
        return results
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None,
                           **kwargs) -> Dict:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            **kwargs: Additional parameters for RandomForestRegressor
            
        Returns:
            Dict: Model and performance metrics
        """
        print("🔄 Training Random Forest...")
        
        # Default parameters
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        rf_params.update(kwargs)
        
        # Train model
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = rf_model.predict(X_train)
        results = {
            'model': rf_model,
            'model_name': 'Random Forest',
            'parameters': rf_params,
            'train_predictions': y_train_pred
        }
        
        # Training metrics
        results['train_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = rf_model.predict(X_val)
            results['val_predictions'] = y_val_pred
            results['val_metrics'] = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'r2': r2_score(y_val, y_val_pred)
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['random_forest'] = results
        print("✅ Random Forest training completed")
        
        return results
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame = None, y_val: pd.Series = None,
                     **kwargs) -> Dict:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            **kwargs: Additional parameters for XGBRegressor
            
        Returns:
            Dict: Model and performance metrics
        """
        print("🔄 Training XGBoost...")
        
        # Default parameters
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        xgb_params.update(kwargs)
        
        # Train model
        xgb_model = xgb.XGBRegressor(**xgb_params)
        
        # Use early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = xgb_model.predict(X_train)
        results = {
            'model': xgb_model,
            'model_name': 'XGBoost',
            'parameters': xgb_params,
            'train_predictions': y_train_pred
        }
        
        # Training metrics
        results['train_metrics'] = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = xgb_model.predict(X_val)
            results['val_predictions'] = y_val_pred
            results['val_metrics'] = {
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred),
                'r2': r2_score(y_val, y_val_pred)
            }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.models['xgboost'] = results
        print("✅ XGBoost training completed")
        
        return results
    
    def train_all_models(self, data_splits: Dict) -> Dict:
        """
        Train all models and compare performance.
        
        Args:
            data_splits (Dict): Dictionary containing data splits
            
        Returns:
            Dict: All model results
        """
        print("🚀 Training all models...")
        
        # Extract data
        X_train = data_splits['X_train_scaled']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val_scaled']
        y_val = data_splits['y_val']
        
        # Train models
        self.train_linear_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Compare models
        comparison = self.compare_models()
        
        print("✅ All models trained successfully!")
        return comparison
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.models:
            print("No models trained yet.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model_data in self.models.items():
            row = {
                'Model': model_data['model_name'],
                'Train_RMSE': model_data['train_metrics']['rmse'],
                'Train_MAE': model_data['train_metrics']['mae'],
                'Train_R2': model_data['train_metrics']['r2']
            }
            
            # Add validation metrics if available
            if 'val_metrics' in model_data:
                row.update({
                    'Val_RMSE': model_data['val_metrics']['rmse'],
                    'Val_MAE': model_data['val_metrics']['mae'],
                    'Val_R2': model_data['val_metrics']['r2']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Identify best model (highest validation R2, or training R2 if no validation)
        if 'Val_R2' in comparison_df.columns:
            best_idx = comparison_df['Val_R2'].idxmax()
            best_model_name = comparison_df.loc[best_idx, 'Model']
        else:
            best_idx = comparison_df['Train_R2'].idxmax()
            best_model_name = comparison_df.loc[best_idx, 'Model']
        
        # Store best model
        for model_key, model_data in self.models.items():
            if model_data['model_name'] == best_model_name:
                self.best_model = model_data
                break
        
        print(f"🏆 Best model: {best_model_name}")
        return comparison_df
    
    def explain_model_shap(self, model_data: Dict, X_sample: pd.DataFrame, 
                          max_display: int = 10) -> Dict:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model_data (Dict): Model data dictionary
            X_sample (pd.DataFrame): Sample data for SHAP analysis
            max_display (int): Maximum number of features to display
            
        Returns:
            Dict: SHAP values and explanations
        """
        try:
            print(f"🔍 Generating SHAP explanations for {model_data['model_name']}...")
            
            model = model_data['model']
            
            # Create SHAP explainer
            if model_data['model_name'] == 'Linear Regression':
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Feature importance from SHAP
            feature_importance_shap = pd.DataFrame({
                'feature': X_sample.columns,
                'shap_importance': np.abs(shap_values).mean(0)
            }).sort_values('shap_importance', ascending=False)
            
            shap_results = {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_importance': feature_importance_shap.head(max_display),
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            }
            
            print("✅ SHAP analysis completed")
            return shap_results
            
        except Exception as e:
            print(f"⚠️  SHAP analysis failed: {str(e)}")
            return {}
    
    def save_models(self, save_dir: str = "models") -> None:
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model_data in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.joblib")
            joblib.dump(model_data['model'], model_path)
            print(f"💾 Saved {model_data['model_name']} to {model_path}")
        
        # Save scaler and encoders
        if self.scaler:
            scaler_path = os.path.join(save_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            print(f"💾 Saved scaler to {scaler_path}")
        
        if self.encoders:
            encoders_path = os.path.join(save_dir, "encoders.joblib")
            joblib.dump(self.encoders, encoders_path)
            print(f"💾 Saved encoders to {encoders_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            Any: Loaded model
        """
        return joblib.load(model_path)
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X (pd.DataFrame): Features for prediction
            model_name (str): Name of model to use. If None, use best model.
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model available. Train models first.")
            model = self.best_model['model']
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found.")
            model = self.models[model_name]['model']
        
        # Scale features if scaler is available
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            return model.predict(X_scaled)
        else:
            return model.predict(X)


# Utility functions
def quick_model_training(df: pd.DataFrame, target_column: str, 
                        test_size: float = 0.2) -> InsuranceModelingPipeline:
    """
    Quick model training pipeline for testing.
    
    Args:
        df (pd.DataFrame): Dataset
        target_column (str): Target variable
        test_size (float): Test set proportion
        
    Returns:
        InsuranceModelingPipeline: Trained pipeline
    """
    pipeline = InsuranceModelingPipeline()
    
    # Prepare features
    X, y = pipeline.prepare_features(df, target_column)
    
    # Split data
    data_splits = pipeline.split_data(X, y, test_size=test_size)
    
    # Train models
    pipeline.train_all_models(data_splits)
    
    return pipeline


if __name__ == "__main__":
    # Example usage with sample data
    from data_processing import load_sample_data
    
    sample_df = load_sample_data()
    pipeline = quick_model_training(sample_df, 'TotalPremium')
    
    # Compare models
    comparison = pipeline.compare_models()
    print("\nModel Comparison:")
    print(comparison)
