"""
Real-time data streaming and processing pipeline for insurance analytics.
Handles live data ingestion, processing, and model inference.
"""

import pandas as pd
import numpy as np
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging
from kafka import KafkaProducer, KafkaConsumer
import redis
from sqlalchemy import create_engine
import joblib
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

class RealTimeDataPipeline:
    """
    Real-time data processing pipeline for insurance analytics.
    
    Features:
    - Kafka integration for streaming data
    - Redis for caching and real-time storage
    - Async processing capabilities
    - Real-time model inference
    - Data validation and quality checks
    """
    
    def __init__(self, config: Dict):
        """Initialize the real-time pipeline."""
        self.config = config
        self.logger = self._setup_logging()
        self.redis_client = self._setup_redis()
        self.kafka_producer = self._setup_kafka_producer()
        self.models = {}
        self.processing_queue = queue.Queue()
        self.is_running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger('streaming_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('logs/streaming_pipeline.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection."""
        try:
            client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            client.ping()
            self.logger.info("Redis connection established")
            return client
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            return None
    
    def _setup_kafka_producer(self) -> KafkaProducer:
        """Setup Kafka producer."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            self.logger.info("Kafka producer initialized")
            return producer
        except Exception as e:
            self.logger.error(f"Kafka producer initialization failed: {e}")
            return None
    
    def load_models(self, model_paths: Dict[str, str]):
        """Load ML models for real-time inference."""
        for model_name, model_path in model_paths.items():
            try:
                self.models[model_name] = joblib.load(model_path)
                self.logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
    
    async def process_streaming_data(self, data: Dict) -> Dict:
        """Process incoming streaming data."""
        
        # Add timestamp
        data['processing_timestamp'] = datetime.now().isoformat()
        
        # Data validation
        validation_result = self._validate_data(data)
        if not validation_result['valid']:
            self.logger.warning(f"Data validation failed: {validation_result['errors']}")
            return {'status': 'error', 'message': 'Data validation failed'}
        
        # Data preprocessing
        processed_data = self._preprocess_streaming_data(data)
        
        # Real-time predictions
        predictions = {}
        for model_name, model in self.models.items():
            try:
                prediction = await self._make_prediction(model, processed_data, model_name)
                predictions[model_name] = prediction
            except Exception as e:
                self.logger.error(f"Prediction failed for {model_name}: {e}")
        
        # Store results in Redis
        result = {
            'original_data': data,
            'processed_data': processed_data,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.redis_client:
            cache_key = f"prediction:{data.get('policy_id', 'unknown')}:{int(time.time())}"
            self.redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
        
        # Send to Kafka for downstream processing
        if self.kafka_producer:
            self.kafka_producer.send('insurance_predictions', value=result)
        
        return result
    
    def _validate_data(self, data: Dict) -> Dict:
        """Validate incoming data quality."""
        errors = []
        required_fields = ['vehicle_age', 'sum_insured', 'province', 'vehicle_type']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if 'vehicle_age' in data:
            try:
                age = float(data['vehicle_age'])
                if age < 0 or age > 50:
                    errors.append("Vehicle age out of valid range (0-50)")
            except (ValueError, TypeError):
                errors.append("Invalid vehicle_age format")
        
        if 'sum_insured' in data:
            try:
                amount = float(data['sum_insured'])
                if amount <= 0:
                    errors.append("Sum insured must be positive")
            except (ValueError, TypeError):
                errors.append("Invalid sum_insured format")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _preprocess_streaming_data(self, data: Dict) -> pd.DataFrame:
        """Preprocess streaming data for model inference."""
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Feature engineering (matching training pipeline)
        if 'vehicle_age' in df.columns and 'sum_insured' in df.columns:
            df['premium_to_value_ratio'] = df.get('total_premium', 0) / df['sum_insured']
        
        # Handle categorical variables
        categorical_columns = ['province', 'vehicle_type', 'gender']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    async def _make_prediction(self, model, data: pd.DataFrame, model_name: str) -> Dict:
        """Make async prediction using loaded model."""
        
        loop = asyncio.get_event_loop()
        
        # Run prediction in thread pool to avoid blocking
        with ThreadPoolExecutor() as executor:
            try:
                prediction = await loop.run_in_executor(
                    executor, 
                    model.predict, 
                    data.select_dtypes(include=[np.number])
                )
                
                # Calculate confidence interval (simplified)
                confidence_interval = self._calculate_confidence_interval(prediction[0])
                
                return {
                    'prediction': float(prediction[0]),
                    'confidence_interval': confidence_interval,
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Prediction error in {model_name}: {e}")
                return {
                    'error': str(e),
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
    
    def _calculate_confidence_interval(self, prediction: float, confidence: float = 0.95) -> List[float]:
        """Calculate prediction confidence interval."""
        # Simplified confidence interval calculation
        # In production, use proper prediction intervals from the model
        margin = prediction * 0.1  # 10% margin as example
        return [prediction - margin, prediction + margin]
    
    def start_kafka_consumer(self, topic: str = 'insurance_data'):
        """Start Kafka consumer for real-time data processing."""
        
        def consume_messages():
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            self.logger.info(f"Started Kafka consumer for topic: {topic}")
            
            for message in consumer:
                if not self.is_running:
                    break
                
                try:
                    # Process message asynchronously
                    asyncio.run(self.process_streaming_data(message.value))
                except Exception as e:
                    self.logger.error(f"Error processing Kafka message: {e}")
        
        # Start consumer in separate thread
        consumer_thread = threading.Thread(target=consume_messages)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        return consumer_thread
    
    def start_pipeline(self):
        """Start the real-time processing pipeline."""
        self.is_running = True
        self.logger.info("Real-time pipeline started")
        
        # Start Kafka consumer
        consumer_thread = self.start_kafka_consumer()
        
        return consumer_thread
    
    def stop_pipeline(self):
        """Stop the real-time processing pipeline."""
        self.is_running = False
        self.logger.info("Real-time pipeline stopped")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent predictions from Redis cache."""
        if not self.redis_client:
            return []
        
        try:
            keys = self.redis_client.keys('prediction:*')
            keys.sort(reverse=True)  # Most recent first
            
            predictions = []
            for key in keys[:limit]:
                data = self.redis_client.get(key)
                if data:
                    predictions.append(json.loads(data))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        if not self.redis_client:
            return {}
        
        try:
            # Get basic stats from Redis
            total_predictions = len(self.redis_client.keys('prediction:*'))
            
            # Calculate processing rate (simplified)
            recent_predictions = self.get_recent_predictions(limit=10)
            if len(recent_predictions) >= 2:
                time_diff = (
                    datetime.fromisoformat(recent_predictions[0]['timestamp']) - 
                    datetime.fromisoformat(recent_predictions[-1]['timestamp'])
                ).total_seconds()
                processing_rate = len(recent_predictions) / max(time_diff, 1)
            else:
                processing_rate = 0
            
            return {
                'total_predictions': total_predictions,
                'processing_rate_per_second': processing_rate,
                'pipeline_status': 'running' if self.is_running else 'stopped',
                'models_loaded': len(self.models),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {e}")
            return {}

# Configuration and usage example
def create_pipeline_config() -> Dict:
    """Create default pipeline configuration."""
    return {
        'kafka_servers': ['localhost:9092'],
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0,
        'model_paths': {
            'xgboost': 'models/xgboost_model.joblib',
            'random_forest': 'models/random_forest_model.joblib'
        }
    }

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    config = create_pipeline_config()
    pipeline = RealTimeDataPipeline(config)
    
    # Load models
    pipeline.load_models(config['model_paths'])
    
    # Start pipeline
    pipeline.start_pipeline()
    
    print("Real-time pipeline started. Press Ctrl+C to stop.")
    
    try:
        while True:
            stats = pipeline.get_pipeline_stats()
            print(f"Pipeline stats: {stats}")
            time.sleep(30)
    except KeyboardInterrupt:
        pipeline.stop_pipeline()
        print("Pipeline stopped.")
