"""
Performance Benchmarking Suite
Comprehensive performance testing for the entire platform
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from modules.data_processing import InsuranceDataProcessor
from modules.modeling_utils import InsuranceModelingPipeline
from ai_advanced.automl_pipeline import AutoMLPipeline

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self):
        """Setup logging for benchmarking."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def benchmark_data_processing(self, data_size: int = 10000) -> Dict[str, float]:
        """Benchmark data processing performance."""
        self.logger.info(f"🔍 Benchmarking data processing with {data_size} records...")
        
        # Generate synthetic data for benchmarking
        synthetic_data = pd.DataFrame({
            'TotalPremium': np.random.normal(3000, 1000, data_size),
            'TotalClaims': np.random.normal(2000, 800, data_size),
            'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], data_size),
            'Gender': np.random.choice(['Male', 'Female'], data_size),
            'VehicleType': np.random.choice(['Car', 'Truck', 'Motorcycle'], data_size),
            'TransactionMonth': pd.date_range('2020-01-01', periods=data_size, freq='D')
        })
        
        processor = InsuranceDataProcessor()
        
        # Benchmark preprocessing
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processed_data = processor.preprocess_pipeline(synthetic_data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        results = {
            'processing_time': end_time - start_time,
            'memory_usage_mb': end_memory - start_memory,
            'records_per_second': data_size / (end_time - start_time),
            'data_size': data_size,
            'output_features': len(processed_data.columns)
        }
        
        self.logger.info(f"✅ Data processing: {results['processing_time']:.2f}s, {results['records_per_second']:.0f} records/sec")
        return results
    
    def benchmark_model_training(self) -> Dict[str, Dict[str, float]]:
        """Benchmark model training performance."""
        self.logger.info("🔍 Benchmarking model training performance...")
        
        # Generate synthetic training data
        n_samples = 5000
        X = pd.DataFrame(np.random.randn(n_samples, 10), 
                        columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(np.random.normal(3000, 1000, n_samples))
        
        pipeline = InsuranceModelingPipeline()
        data_splits = pipeline.split_data(X, y)
        
        models = ['linear_regression', 'random_forest', 'xgboost']
        results = {}
        
        for model_name in models:
            self.logger.info(f"Training {model_name}...")
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            if model_name == 'linear_regression':
                model, metrics = pipeline.train_linear_regression(data_splits)
            elif model_name == 'random_forest':
                model, metrics = pipeline.train_random_forest(data_splits, n_estimators=50)
            elif model_name == 'xgboost':
                model, metrics = pipeline.train_xgboost(data_splits, n_estimators=50)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results[model_name] = {
                'training_time': end_time - start_time,
                'memory_usage_mb': end_memory - start_memory,
                'r2_score': metrics['r2'],
                'rmse': metrics['rmse']
            }
            
            self.logger.info(f"✅ {model_name}: {results[model_name]['training_time']:.2f}s, R²={results[model_name]['r2_score']:.3f}")
        
        return results
    
    def benchmark_prediction_speed(self) -> Dict[str, float]:
        """Benchmark prediction speed."""
        self.logger.info("🔍 Benchmarking prediction speed...")
        
        # Create a simple model for benchmarking
        from sklearn.ensemble import RandomForestRegressor
        
        X_train = np.random.randn(1000, 10)
        y_train = np.random.randn(1000)
        X_test = np.random.randn(10000, 10)  # Large test set
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Benchmark single predictions
        start_time = time.time()
        for i in range(1000):
            _ = model.predict(X_test[i:i+1])
        single_pred_time = time.time() - start_time
        
        # Benchmark batch predictions
        start_time = time.time()
        _ = model.predict(X_test)
        batch_pred_time = time.time() - start_time
        
        results = {
            'single_prediction_ms': (single_pred_time / 1000) * 1000,
            'batch_prediction_time': batch_pred_time,
            'predictions_per_second': len(X_test) / batch_pred_time,
            'batch_size': len(X_test)
        }
        
        self.logger.info(f"✅ Prediction speed: {results['predictions_per_second']:.0f} predictions/sec")
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        self.logger.info("🚀 Starting comprehensive performance benchmark...")
        
        benchmark_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': sys.version
            },
            'data_processing': self.benchmark_data_processing(),
            'model_training': self.benchmark_model_training(),
            'prediction_speed': self.benchmark_prediction_speed()
        }
        
        # Calculate overall performance score
        processing_score = min(benchmark_results['data_processing']['records_per_second'] / 1000, 10)
        prediction_score = min(benchmark_results['prediction_speed']['predictions_per_second'] / 10000, 10)
        
        benchmark_results['overall_performance_score'] = (processing_score + prediction_score) / 2
        
        # Save results
        Path('results').mkdir(exist_ok=True)
        with open('results/performance_benchmark.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        self.logger.info("🎉 Comprehensive benchmark completed!")
        self.logger.info(f"📊 Overall Performance Score: {benchmark_results['overall_performance_score']:.2f}/10")
        
        return benchmark_results

def main():
    """Main execution function."""
    print("🚀 AlphaCare Insurance Analytics - Performance Benchmark")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n📊 PERFORMANCE BENCHMARK RESULTS")
    print(f"Overall Score: {results['overall_performance_score']:.2f}/10")
    print(f"Data Processing: {results['data_processing']['records_per_second']:.0f} records/sec")
    print(f"Prediction Speed: {results['prediction_speed']['predictions_per_second']:.0f} predictions/sec")
    print(f"System: {results['system_info']['cpu_count']} CPUs, {results['system_info']['memory_gb']:.1f}GB RAM")

if __name__ == "__main__":
    main()
