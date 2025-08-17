"""
Advanced Neural Network Models for Insurance Premium Prediction
Implements deep learning approaches with TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class InsuranceNeuralNetwork:
    """
    Advanced neural network for insurance premium prediction
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the neural network model"""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.history = None
        self.feature_names = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = 'TotalPremium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for neural network training
        
        Args:
            data: Input DataFrame
            target_column: Name of target variable
            
        Returns:
            Tuple of (X, y) preprocessed arrays
        """
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column].values
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def build_model(self, input_dim: int, architecture: str = 'deep') -> keras.Model:
        """
        Build neural network architecture
        
        Args:
            input_dim: Number of input features
            architecture: Type of architecture ('simple', 'deep', 'wide_deep')
            
        Returns:
            Compiled Keras model
        """
        
        if architecture == 'simple':
            model = self._build_simple_model(input_dim)
        elif architecture == 'deep':
            model = self._build_deep_model(input_dim)
        elif architecture == 'wide_deep':
            model = self._build_wide_deep_model(input_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_simple_model(self, input_dim: int) -> keras.Model:
        """Build simple feedforward network"""
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_deep_model(self, input_dim: int) -> keras.Model:
        """Build deep neural network"""
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        return model
    
    def _build_wide_deep_model(self, input_dim: int) -> keras.Model:
        """Build wide & deep architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Wide component (linear)
        wide = layers.Dense(1, activation='linear')(inputs)
        
        # Deep component
        deep = layers.Dense(128, activation='relu')(inputs)
        deep = layers.Dropout(0.3)(deep)
        deep = layers.Dense(64, activation='relu')(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(32, activation='relu')(deep)
        deep = layers.Dense(1, activation='linear')(deep)
        
        # Combine wide and deep
        combined = layers.Add()([wide, deep])
        
        model = keras.Model(inputs=inputs, outputs=combined)
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              architecture: str = 'deep',
              early_stopping: bool = True) -> Dict[str, Any]:
        """
        Train the neural network
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            architecture: Model architecture type
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history and metrics
        """
        
        # Build model
        self.model = self.build_model(X.shape[1], architecture)
        
        # Setup callbacks
        callback_list = []
        
        if early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Learning rate reduction
        lr_reduce = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(lr_reduce)
        
        # Train model
        self.logger.info(f"Training neural network with {architecture} architecture...")
        
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        # Calculate final metrics
        y_pred = self.model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        self.logger.info(f"Training completed. R² Score: {metrics['r2']:.4f}")
        
        return {
            'model': self.model,
            'history': self.history.history,
            'metrics': metrics,
            'architecture': architecture
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X).flatten()
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        # Model architecture summary
        axes[1, 1].text(0.1, 0.5, f"Model Summary:\nTotal Parameters: {self.model.count_params():,}\nTrainable Parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}", 
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Model Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def feature_importance_analysis(self, X: np.ndarray, y: np.ndarray, method: str = 'permutation') -> pd.DataFrame:
        """
        Analyze feature importance using various methods
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Method for importance calculation ('permutation', 'gradient')
            
        Returns:
            DataFrame with feature importance scores
        """
        
        if method == 'permutation':
            return self._permutation_importance(X, y)
        elif method == 'gradient':
            return self._gradient_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Calculate permutation importance"""
        
        # Baseline score
        baseline_score = r2_score(y, self.predict(X))
        
        importance_scores = []
        
        for i in range(X.shape[1]):
            # Create copy and permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate score with permuted feature
            permuted_score = r2_score(y, self.predict(X_permuted))
            
            # Importance is the decrease in performance
            importance = baseline_score - permuted_score
            importance_scores.append(importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _gradient_importance(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Calculate gradient-based importance"""
        
        # Convert to tensor
        X_tensor = tf.constant(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
        
        # Calculate gradients
        gradients = tape.gradient(predictions, X_tensor)
        
        # Calculate importance as mean absolute gradient
        importance_scores = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{filepath}_scaler.joblib")
        joblib.dump(self.label_encoders, f"{filepath}_encoders.joblib")
        joblib.dump(self.feature_names, f"{filepath}_features.joblib")
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and preprocessors"""
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # Load preprocessors
        self.scaler = joblib.load(f"{filepath}_scaler.joblib")
        self.label_encoders = joblib.load(f"{filepath}_encoders.joblib")
        self.feature_names = joblib.load(f"{filepath}_features.joblib")
        
        self.logger.info(f"Model loaded from {filepath}")

def compare_architectures(data: pd.DataFrame, target_column: str = 'TotalPremium') -> pd.DataFrame:
    """
    Compare different neural network architectures
    
    Args:
        data: Input DataFrame
        target_column: Target variable name
        
    Returns:
        Comparison results DataFrame
    """
    
    architectures = ['simple', 'deep', 'wide_deep']
    results = []
    
    for arch in architectures:
        print(f"\nTraining {arch} architecture...")
        
        # Initialize model
        nn = InsuranceNeuralNetwork()
        
        # Preprocess data
        X, y = nn.preprocess_data(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        training_results = nn.train(
            X_train, y_train,
            architecture=arch,
            epochs=50,  # Reduced for comparison
            early_stopping=True
        )
        
        # Evaluate on test set
        y_pred = nn.predict(X_test)
        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        results.append({
            'architecture': arch,
            'train_r2': training_results['metrics']['r2'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'parameters': nn.model.count_params()
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    print("Neural Network Insurance Premium Prediction")
    print("=" * 50)
    
    # Load sample data (replace with actual data path)
    try:
        data = pd.read_csv('data/insurance_data.csv')
        
        # Compare architectures
        comparison_results = compare_architectures(data)
        print("\nArchitecture Comparison Results:")
        print(comparison_results)
        
        # Train best model
        best_arch = comparison_results.loc[comparison_results['test_r2'].idxmax(), 'architecture']
        print(f"\nBest architecture: {best_arch}")
        
        # Train final model
        nn = InsuranceNeuralNetwork()
        X, y = nn.preprocess_data(data)
        
        final_results = nn.train(
            X, y,
            architecture=best_arch,
            epochs=100
        )
        
        # Plot training history
        nn.plot_training_history('results/neural_network_training.png')
        
        # Feature importance
        importance_df = nn.feature_importance_analysis(X, y, method='permutation')
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Save model
        nn.save_model('models/neural_network')
        
        print("\nNeural network training completed successfully!")
        
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/insurance_data.csv' exists.")
    except Exception as e:
        print(f"Error: {e}")
