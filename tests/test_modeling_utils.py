"""
Comprehensive test suite for modeling utilities.
Tests machine learning pipeline, model training, and evaluation functions.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from modeling_utils import InsuranceModelingPipeline


class TestInsuranceModelingPipeline:
    """Test suite for InsuranceModelingPipeline class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample processed data for modeling."""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'VehicleAge': np.random.randint(0, 15, n_samples),
            'SumInsured': np.random.uniform(100000, 500000, n_samples),
            'ExcessSelected': np.random.uniform(500, 2000, n_samples),
            'Cylinders': np.random.randint(4, 8, n_samples),
            'kilowatts': np.random.randint(50, 200, n_samples),
            'PremiumToValue_Ratio': np.random.uniform(0.01, 0.05, n_samples),
            'RiskScore': np.random.uniform(0, 1, n_samples),
            'Gender_Male': np.random.choice([0, 1], n_samples),
            'Province_Gauteng': np.random.choice([0, 1], n_samples),
            'Province_Western_Cape': np.random.choice([0, 1], n_samples),
            'VehicleType_Car': np.random.choice([0, 1], n_samples),
            'CoverType_Motor': np.random.choice([0, 1], n_samples),
            'TotalPremium': np.random.uniform(1000, 6000, n_samples)
        })
    
    @pytest.fixture
    def pipeline(self):
        """Create InsuranceModelingPipeline instance."""
        return InsuranceModelingPipeline(random_state=42)
    
    def test_prepare_features(self, pipeline, sample_data):
        """Test feature preparation for modeling."""
        target_col = 'TotalPremium'
        X, y = pipeline.prepare_features(sample_data, target_col)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert target_col not in X.columns
        assert len(X.columns) == len(sample_data.columns) - 1
    
    def test_split_data(self, pipeline, sample_data):
        """Test data splitting functionality."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        data_splits = pipeline.split_data(X, y, test_size=0.2, val_size=0.2)
        
        assert isinstance(data_splits, dict)
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        for key in required_keys:
            assert key in data_splits
        
        # Check split proportions
        total_samples = len(X)
        assert len(data_splits['X_train']) == int(total_samples * 0.6)
        assert len(data_splits['X_val']) == int(total_samples * 0.2)
        assert len(data_splits['X_test']) == int(total_samples * 0.2)
    
    def test_train_linear_regression(self, pipeline, sample_data):
        """Test linear regression model training."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        data_splits = pipeline.split_data(X, y)
        
        model, metrics = pipeline.train_linear_regression(data_splits)
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        
        # Check if metrics are reasonable
        assert -1 <= metrics['train_r2'] <= 1
        assert -1 <= metrics['val_r2'] <= 1
        assert metrics['train_rmse'] > 0
        assert metrics['val_rmse'] > 0
    
    def test_train_random_forest(self, pipeline, sample_data):
        """Test random forest model training."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        data_splits = pipeline.split_data(X, y)
        
        model, metrics = pipeline.train_random_forest(data_splits, n_estimators=10)
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics
        assert 'feature_importance' in metrics
        
        # Check feature importance
        assert len(metrics['feature_importance']) == len(X.columns)
    
    def test_train_xgboost(self, pipeline, sample_data):
        """Test XGBoost model training."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        data_splits = pipeline.split_data(X, y)
        
        model, metrics = pipeline.train_xgboost(data_splits, n_estimators=10)
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics
        assert 'feature_importance' in metrics
    
    def test_train_all_models(self, pipeline, sample_data):
        """Test training all models together."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        data_splits = pipeline.split_data(X, y)
        
        results = pipeline.train_all_models(data_splits, quick_mode=True)
        
        assert isinstance(results, dict)
        expected_models = ['linear_regression', 'random_forest', 'xgboost']
        
        for model_name in expected_models:
            assert model_name in results
            assert 'model' in results[model_name]
            assert 'metrics' in results[model_name]
    
    def test_evaluate_model(self, pipeline, sample_data):
        """Test model evaluation functionality."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a simple model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        metrics = pipeline.evaluate_model(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
    
    def test_cross_validate_model(self, pipeline, sample_data):
        """Test cross-validation functionality."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        cv_results = pipeline.cross_validate_model(model, X, y, cv=3)
        
        assert isinstance(cv_results, dict)
        assert 'test_r2' in cv_results
        assert 'test_rmse' in cv_results
        assert len(cv_results['test_r2']) == 3
    
    def test_get_feature_importance(self, pipeline, sample_data):
        """Test feature importance extraction."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        importance_df = pipeline.get_feature_importance(model, X.columns)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(X.columns)
    
    def test_save_and_load_model(self, pipeline, sample_data, tmp_path):
        """Test model saving and loading."""
        X, y = pipeline.prepare_features(sample_data, 'TotalPremium')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Test saving
        model_path = tmp_path / "test_model.joblib"
        pipeline.save_model(model, str(model_path))
        assert model_path.exists()
        
        # Test loading
        loaded_model = pipeline.load_model(str(model_path))
        assert loaded_model is not None
        
        # Test predictions are the same
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
