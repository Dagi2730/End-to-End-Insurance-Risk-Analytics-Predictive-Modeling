"""
Comprehensive test suite for data processing module.
Tests all data preprocessing, feature engineering, and validation functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from data_processing import InsuranceDataProcessor


class TestInsuranceDataProcessor:
    """Test suite for InsuranceDataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample insurance data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'UnderwrittenCoverID': range(1000, 1100),
            'PolicyID': [f'POL_{i}' for i in range(100)],
            'TransactionMonth': pd.date_range('2023-01-01', periods=100, freq='D'),
            'IsVATRegistered': np.random.choice([True, False], 100),
            'Citizenship': np.random.choice(['South African', 'Foreign'], 100),
            'LegalType': np.random.choice(['Individual', 'Corporate'], 100),
            'Title': np.random.choice(['Mr', 'Mrs', 'Ms', 'Dr'], 100),
            'Language': np.random.choice(['English', 'Afrikaans', 'Zulu'], 100),
            'Bank': np.random.choice(['ABSA', 'FNB', 'Standard Bank'], 100),
            'AccountType': np.random.choice(['Current', 'Savings'], 100),
            'MaritalStatus': np.random.choice(['Married', 'Single', 'Divorced'], 100),
            'Gender': np.random.choice(['Male', 'Female'], 100),
            'Country': ['South Africa'] * 100,
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], 100),
            'PostalCode': np.random.randint(1000, 9999, 100),
            'MainCrestaZone': np.random.choice(['Zone A', 'Zone B', 'Zone C'], 100),
            'SubCrestaZone': np.random.choice(['Sub1', 'Sub2', 'Sub3'], 100),
            'ItemType': np.random.choice(['Vehicle', 'Property'], 100),
            'mmcode': np.random.randint(100, 999, 100),
            'VehicleType': np.random.choice(['Car', 'Truck', 'Motorcycle'], 100),
            'RegistrationYear': np.random.randint(2010, 2024, 100),
            'make': np.random.choice(['Toyota', 'BMW', 'Mercedes'], 100),
            'Model': np.random.choice(['Corolla', '3 Series', 'C-Class'], 100),
            'Cylinders': np.random.randint(4, 8, 100),
            'cubiccapacity': np.random.randint(1000, 3000, 100),
            'kilowatts': np.random.randint(50, 200, 100),
            'bodytype': np.random.choice(['Sedan', 'SUV', 'Hatchback'], 100),
            'NumberOfDoors': np.random.choice([2, 4, 5], 100),
            'VehicleIntroDate': pd.date_range('2010-01-01', periods=100, freq='30D'),
            'CustomValueEstimate': np.random.uniform(100000, 500000, 100),
            'SumInsured': np.random.uniform(150000, 600000, 100),
            'CalculatedPremiumPerTerm': np.random.uniform(1000, 5000, 100),
            'ExcessSelected': np.random.uniform(500, 2000, 100),
            'CoverCategory': np.random.choice(['Comprehensive', 'Third Party'], 100),
            'CoverType': np.random.choice(['Motor', 'Household'], 100),
            'CoverGroup': np.random.choice(['Group A', 'Group B'], 100),
            'Section': np.random.choice(['Section 1', 'Section 2'], 100),
            'Product': np.random.choice(['Product A', 'Product B'], 100),
            'StatutoryClass': np.random.choice(['Class 1', 'Class 2'], 100),
            'StatutoryRiskType': np.random.choice(['Low', 'Medium', 'High'], 100),
            'TotalPremium': np.random.uniform(1200, 6000, 100),
            'TotalClaims': np.random.uniform(0, 10000, 100)
        })
    
    @pytest.fixture
    def processor(self):
        """Create InsuranceDataProcessor instance."""
        return InsuranceDataProcessor()
    
    def test_load_data_success(self, processor, sample_data):
        """Test successful data loading."""
        with patch('pandas.read_csv', return_value=sample_data):
            result = processor.load_data('dummy_path.csv')
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
    
    def test_load_data_file_not_found(self, processor):
        """Test data loading with file not found."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                processor.load_data('nonexistent.csv')
    
    def test_basic_info(self, processor, sample_data):
        """Test basic data information extraction."""
        info = processor.get_basic_info(sample_data)
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'memory_usage' in info
        assert info['shape'] == (100, 47)
    
    def test_missing_values_analysis(self, processor, sample_data):
        """Test missing values analysis."""
        # Add some missing values
        sample_data.loc[0:5, 'Gender'] = np.nan
        sample_data.loc[10:15, 'Province'] = np.nan
        
        missing_info = processor.analyze_missing_values(sample_data)
        
        assert isinstance(missing_info, pd.DataFrame)
        assert 'missing_count' in missing_info.columns
        assert 'missing_percentage' in missing_info.columns
        assert missing_info.loc['Gender', 'missing_count'] == 6
    
    def test_outlier_detection(self, processor, sample_data):
        """Test outlier detection using IQR method."""
        outliers = processor.detect_outliers(sample_data, ['TotalPremium', 'TotalClaims'])
        
        assert isinstance(outliers, dict)
        assert 'TotalPremium' in outliers
        assert 'TotalClaims' in outliers
        assert isinstance(outliers['TotalPremium'], list)
    
    def test_feature_engineering(self, processor, sample_data):
        """Test feature engineering functions."""
        engineered_data = processor.engineer_features(sample_data)
        
        # Check if new features are created
        expected_features = [
            'VehicleAge', 'PremiumToValue_Ratio', 'ClaimsRatio',
            'IsHighValue', 'RiskScore', 'Month', 'Quarter', 'DayOfWeek'
        ]
        
        for feature in expected_features:
            assert feature in engineered_data.columns
        
        # Check data types and ranges
        assert engineered_data['VehicleAge'].min() >= 0
        assert engineered_data['IsHighValue'].dtype == bool
        assert engineered_data['Month'].between(1, 12).all()
    
    def test_encode_categorical_variables(self, processor, sample_data):
        """Test categorical variable encoding."""
        encoded_data = processor.encode_categorical_variables(sample_data)
        
        # Check if categorical columns are properly encoded
        categorical_cols = ['Gender', 'Province', 'VehicleType', 'CoverType']
        
        for col in categorical_cols:
            if col in sample_data.columns:
                # Check if encoded columns exist
                encoded_cols = [c for c in encoded_data.columns if c.startswith(f'{col}_')]
                assert len(encoded_cols) > 0
    
    def test_data_validation(self, processor, sample_data):
        """Test data validation functions."""
        validation_results = processor.validate_data(sample_data)
        
        assert isinstance(validation_results, dict)
        assert 'is_valid' in validation_results
        assert 'issues' in validation_results
        assert isinstance(validation_results['issues'], list)
    
    def test_preprocessing_pipeline(self, processor, sample_data):
        """Test complete preprocessing pipeline."""
        processed_data = processor.preprocess_pipeline(sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        
        # Check if key engineered features exist
        assert 'VehicleAge' in processed_data.columns
        assert 'RiskScore' in processed_data.columns
        
        # Check if categorical variables are encoded
        assert any(col.startswith('Gender_') for col in processed_data.columns)
    
    def test_data_quality_report(self, processor, sample_data):
        """Test data quality report generation."""
        quality_report = processor.generate_data_quality_report(sample_data)
        
        assert isinstance(quality_report, dict)
        assert 'summary' in quality_report
        assert 'missing_values' in quality_report
        assert 'outliers' in quality_report
        assert 'data_types' in quality_report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
