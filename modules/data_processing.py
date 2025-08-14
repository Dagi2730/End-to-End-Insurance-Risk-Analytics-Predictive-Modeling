"""
AlphaCare Insurance Solutions - Data Processing Module

This module contains all data preprocessing functions for the insurance analytics project.
Handles data loading, cleaning, feature engineering, and preparation for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class InsuranceDataProcessor:
    """
    Comprehensive data processing class for AlphaCare Insurance Analytics.
    
    This class handles all aspects of data preprocessing including:
    - Data loading and validation
    - Missing value treatment
    - Feature engineering
    - Data quality checks
    - Train-test splitting
    """
    
    def __init__(self, data_path: str = "data/insurance_data.csv"):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the insurance dataset
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the insurance dataset with comprehensive error handling.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
                
            # Load data with multiple format support
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path, low_memory=False)
            elif self.data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
                
            print(f"✅ Data loaded successfully: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def validate_data_structure(self) -> Dict[str, any]:
        """
        Validate the structure and quality of the loaded dataset.
        
        Returns:
            Dict: Data validation report
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        validation_report = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'duplicates': self.raw_data.duplicated().sum(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Check for required columns
        required_columns = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        missing_required = [col for col in required_columns if col not in self.raw_data.columns]
        validation_report['missing_required_columns'] = missing_required
        
        print("📊 Data Validation Report:")
        print(f"   Shape: {validation_report['shape']}")
        print(f"   Missing values: {sum(validation_report['missing_values'].values())}")
        print(f"   Duplicates: {validation_report['duplicates']}")
        print(f"   Memory usage: {validation_report['memory_usage']:.2f} MB")
        
        if missing_required:
            print(f"   ⚠️  Missing required columns: {missing_required}")
            
        return validation_report
    
    def handle_missing_values(self, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values using various strategies.
        
        Args:
            strategy (str): Strategy for handling missing values
                          'smart' - Use different strategies per column type
                          'drop' - Drop rows with missing values
                          'fill' - Fill with mean/mode
        
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        df = self.raw_data.copy()
        initial_shape = df.shape
        
        if strategy == 'smart':
            # Numerical columns - fill with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Categorical columns - fill with mode or 'Unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        
        elif strategy == 'drop':
            df = df.dropna()
            
        elif strategy == 'fill':
            df = df.fillna(df.mean(numeric_only=True))
            df = df.fillna(df.mode().iloc[0])
        
        final_shape = df.shape
        print(f"📝 Missing values handled: {initial_shape} → {final_shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df_engineered = df.copy()
        
        # Financial ratios and metrics
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            # Profit margin
            df_engineered['ProfitMargin'] = df_engineered['TotalPremium'] - df_engineered['TotalClaims']
            
            # Claims ratio
            df_engineered['ClaimsRatio'] = np.where(
                df_engineered['TotalPremium'] > 0,
                df_engineered['TotalClaims'] / df_engineered['TotalPremium'],
                0
            )
            
            # Risk categories based on claims ratio
            df_engineered['RiskCategory'] = pd.cut(
                df_engineered['ClaimsRatio'],
                bins=[-np.inf, 0.3, 0.7, np.inf],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
        
        # Vehicle age (if VehicleIntroDate exists)
        if 'VehicleIntroDate' in df.columns:
            try:
                df_engineered['VehicleIntroDate'] = pd.to_datetime(df_engineered['VehicleIntroDate'])
                current_year = datetime.now().year
                df_engineered['VehicleAge'] = current_year - df_engineered['VehicleIntroDate'].dt.year
                
                # Vehicle age categories
                df_engineered['VehicleAgeCategory'] = pd.cut(
                    df_engineered['VehicleAge'],
                    bins=[-np.inf, 3, 7, 15, np.inf],
                    labels=['New', 'Recent', 'Mature', 'Old']
                )
            except:
                print("⚠️  Could not process VehicleIntroDate")
        
        # Customer tenure (if available)
        if 'TransactionMonth' in df.columns:
            try:
                df_engineered['TransactionMonth'] = pd.to_datetime(df_engineered['TransactionMonth'])
                df_engineered['CustomerTenure'] = (
                    df_engineered['TransactionMonth'].max() - df_engineered['TransactionMonth']
                ).dt.days / 365.25
            except:
                print("⚠️  Could not process TransactionMonth")
        
        # Premium per value ratio
        if 'CustomValueEstimate' in df.columns and 'TotalPremium' in df.columns:
            df_engineered['PremiumValueRatio'] = np.where(
                df_engineered['CustomValueEstimate'] > 0,
                df_engineered['TotalPremium'] / df_engineered['CustomValueEstimate'],
                0
            )
        
        new_features = set(df_engineered.columns) - set(df.columns)
        print(f"🔧 Engineered {len(new_features)} new features: {list(new_features)}")
        
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical features for machine learning.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Encoded dataframe and encoding mappings
        """
        df_encoded = df.copy()
        encoding_mappings = {}
        
        # Get categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df_encoded[col].nunique() <= 10:  # One-hot encode low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
                encoding_mappings[col] = {'type': 'one_hot', 'columns': list(dummies.columns)}
                
            else:  # Label encode high cardinality
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoding_mappings[col] = {'type': 'label', 'encoder': le}
        
        print(f"🏷️  Encoded {len(categorical_cols)} categorical features")
        return df_encoded, encoding_mappings
    
    def prepare_modeling_data(self, target_column: str = 'TotalPremium', 
                            test_size: float = 0.3, random_state: int = 42) -> Dict:
        """
        Prepare data for machine learning modeling.
        
        Args:
            target_column (str): Name of target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict: Dictionary containing train/test splits and metadata
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run full preprocessing first.")
            
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        df = self.processed_data.copy()
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Remove non-numeric columns that weren't encoded
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        modeling_data = {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'scaler': scaler,
            'target_column': target_column
        }
        
        print(f"🎯 Modeling data prepared:")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {len(modeling_data['feature_names'])}")
        
        return modeling_data
    
    def run_full_preprocessing(self, missing_strategy: str = 'smart') -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            missing_strategy (str): Strategy for handling missing values
            
        Returns:
            pd.DataFrame: Fully processed dataset
        """
        print("🚀 Starting full preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Validate structure
        validation_report = self.validate_data_structure()
        
        # Step 3: Handle missing values
        df_clean = self.handle_missing_values(strategy=missing_strategy)
        
        # Step 4: Engineer features
        df_engineered = self.engineer_features(df_clean)
        
        # Step 5: Encode categorical features
        df_encoded, encoding_mappings = self.encode_categorical_features(df_engineered)
        
        # Store processed data
        self.processed_data = df_encoded
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        self.processed_data.to_csv('data/processed/insurance_processed.csv', index=False)
        
        print("✅ Full preprocessing completed successfully!")
        print(f"   Final shape: {self.processed_data.shape}")
        print(f"   Processed data saved to: data/processed/insurance_processed.csv")
        
        return self.processed_data


def load_sample_data() -> pd.DataFrame:
    """
    Generate sample insurance data for testing purposes.
    
    Returns:
        pd.DataFrame: Sample insurance dataset
    """
    np.random.seed(42)
    n_samples = 10000
    
    # Generate sample data
    data = {
        'PolicyID': [f'POL_{i:06d}' for i in range(n_samples)],
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape'], n_samples),
        'ZipCode': np.random.choice([f'{i:04d}' for i in range(1000, 9999)], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'VehicleType': np.random.choice(['Sedan', 'Hatchback', 'SUV', 'Truck'], n_samples),
        'VehicleIntroDate': pd.date_range('2000-01-01', '2020-12-31', periods=n_samples),
        'CustomValueEstimate': np.random.normal(200000, 50000, n_samples).clip(50000, 500000),
        'TotalPremium': np.random.normal(15000, 5000, n_samples).clip(5000, 50000),
        'TotalClaims': np.random.normal(8000, 8000, n_samples).clip(0, 100000),
        'TransactionMonth': pd.date_range('2014-02-01', '2015-08-31', periods=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    df['TotalClaims'] = np.where(
        df['VehicleType'] == 'SUV',
        df['TotalClaims'] * 1.2,
        df['TotalClaims']
    )
    
    df['TotalPremium'] = np.where(
        df['Province'] == 'Gauteng',
        df['TotalPremium'] * 1.1,
        df['TotalPremium']
    )
    
    return df


if __name__ == "__main__":
    # Example usage
    processor = InsuranceDataProcessor()
    
    # Generate sample data if no data file exists
    if not os.path.exists("data/insurance_data.csv"):
        print("📁 No data file found. Generating sample data...")
        os.makedirs('data', exist_ok=True)
        sample_data = load_sample_data()
        sample_data.to_csv('data/insurance_data.csv', index=False)
        print("✅ Sample data generated and saved to data/insurance_data.csv")
    
    # Run preprocessing
    processed_data = processor.run_full_preprocessing()
