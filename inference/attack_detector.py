#!/usr/bin/env python3
"""
Inference Module for SCADA Attack Detection

This module provides functionality for real-time detection of attacks
on a SCADA system using the trained neural network model.
"""

# Import compatibility fix for Python 3.13
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compatibility_fix

import os
import json
import numpy as np
import pandas as pd
import time
import joblib
import tensorflow as tf
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttackDetector:
    """Real-time detector for SCADA attacks"""
    
    def __init__(self, model_dir=None, transformer_path=None):
        """
        Initialize the attack detector
        
        Args:
            model_dir: Directory containing the trained model
            transformer_path: Path to preprocessor transformers
        """
        # Set default paths if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(base_dir, 'training', 'models')
        
        if transformer_path is None:
            self.transformer_path = os.path.join(
                base_dir, 'training', 'data', 'processed', 'processed_transformers.pkl'
            )
        else:
            self.transformer_path = transformer_path
            
        # Attributes that will be set during loading
        self.model = None
        self.transformers = None
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        
        # Detection history
        self.detection_history = []
        
        # Thresholds
        self.threshold = 0.5  # Default threshold for binary classification
        
        # Performance tracking
        self.inference_times = []
        
    def load_model_and_transformers(self):
        """Load the trained model and preprocessing transformers"""
        try:
            # Find the best model path
            if os.path.isfile(os.path.join(self.model_dir, 'best_model.h5')):
                model_path = os.path.join(self.model_dir, 'best_model.h5')
            else:
                # The model_dir might be the directory above best_model
                # Read best_model.txt to find the actual model
                best_model_file = os.path.join(self.model_dir, 'best_model.txt')
                if os.path.isfile(best_model_file):
                    with open(best_model_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith('Best model:'):
                                model_name = line.split(':')[1].strip()
                                model_path = os.path.join(
                                    self.model_dir, model_name, 'best_model.h5'
                                )
                                break
                else:
                    # Try to find any .h5 file
                    h5_files = [f for f in os.listdir(self.model_dir) 
                                if f.endswith('.h5')]
                    if h5_files:
                        model_path = os.path.join(self.model_dir, h5_files[0])
                    else:
                        raise FileNotFoundError("Could not find model file")
            
            # Load the model
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load transformers
            logger.info(f"Loading transformers from {self.transformer_path}")
            transformers = joblib.load(self.transformer_path)
            
            self.transformers = transformers
            self.feature_columns = transformers['feature_columns']
            self.categorical_columns = transformers['categorical_columns']
            self.numerical_columns = transformers['numerical_columns']
            
            logger.info("Model and transformers loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model and transformers: {e}")
            return False
    
    def preprocess_data(self, data):
        """
        Preprocess SCADA data for inference
        
        Args:
            data: Dictionary or pandas DataFrame with SCADA measurements
        
        Returns:
            Preprocessed data ready for model input
        """
        # Convert dictionary to DataFrame if necessary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a dictionary or DataFrame")
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df.columns:
                # Set default values for missing columns
                if col == 'rtu_id':
                    df[col] = 'unknown'
                elif col == 'bus_number':
                    df[col] = 0
                elif col == 'status':
                    df[col] = 'UNKNOWN'
                elif 'voltage' in col:
                    df[col] = 1.0
                elif 'frequency' in col:
                    df[col] = 50.0
                else:
                    df[col] = 0.0
                    
        # Add derived features if they don't exist
        if 'voltage_deviation' not in df.columns and 'voltage_magnitude' in df.columns:
            df['voltage_deviation'] = abs(df['voltage_magnitude'] - 1.0)
            
        if 'frequency_deviation' not in df.columns and 'frequency' in df.columns:
            df['frequency_deviation'] = abs(df['frequency'] - 50.0)
            
        if 'voltage_in_normal_range' not in df.columns and 'voltage_magnitude' in df.columns:
            df['voltage_in_normal_range'] = ((df['voltage_magnitude'] >= 0.95) & 
                                          (df['voltage_magnitude'] <= 1.05)).astype(int)
            
        if 'frequency_in_normal_range' not in df.columns and 'frequency' in df.columns:
            df['frequency_in_normal_range'] = ((df['frequency'] >= 49.5) & 
                                            (df['frequency'] <= 50.5)).astype(int)
        
        # Select only the required features
        df = df[self.feature_columns].copy()
        
        # Process categorical features
        for col in self.categorical_columns:
            # Get encoder for this column
            encoder = self.transformers['encoders'][col]
            
            # Transform
            col_encoded = encoder.transform(df[[col]])
            
            # Get feature names
            feature_names = [f"{col}_{int(i)}" for i in range(col_encoded.shape[1])]
            
            # Add encoded features
            for i, name in enumerate(feature_names):
                df[name] = col_encoded[:, i]
            
            # Drop original column
            df.drop(col, axis=1, inplace=True)
        
        # Process numerical features
        scaler = self.transformers['scalers']['numerical']
        df[self.numerical_columns] = scaler.transform(df[self.numerical_columns])
        
        return df
    
    def detect(self, data):
        """
        Detect if the given SCADA data represents an attack
        
        Args:
            data: Dictionary or pandas DataFrame with SCADA measurements
        
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Preprocess the data
        preprocessed_data = self.preprocess_data(data)
        
        # Make prediction
        prediction_prob = self.model.predict(preprocessed_data, verbose=0)[0][0]
        prediction = int(prediction_prob > self.threshold)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Determine attack type based on measurements
        attack_type = "None"
        confidence = 1 - prediction_prob if prediction == 0 else prediction_prob
        
        if prediction == 1:
            # This is a very simple heuristic - in real applications this would be more sophisticated
            if isinstance(data, dict):
                voltage = data.get('voltage_magnitude', 1.0)
                frequency = data.get('frequency', 50.0)
                
                if abs(voltage - 1.0) > 0.05 and abs(frequency - 50.0) > 0.5:
                    attack_type = "Combined Attack"
                elif abs(voltage - 1.0) > 0.05:
                    attack_type = "Voltage Attack"
                elif abs(frequency - 50.0) > 0.5:
                    attack_type = "Frequency Attack"
                else:
                    attack_type = "Subtle Attack"
            else:
                attack_type = "Unknown Attack"
        
        # Create result
        result = {
            'timestamp': datetime.now().isoformat(),
            'is_attack': bool(prediction),
            'attack_type': attack_type,
            'confidence': float(confidence),
            'probability': float(prediction_prob),
            'inference_time_ms': inference_time * 1000
        }
        
        # Add to detection history
        self.detection_history.append(result)
        if len(self.detection_history) > 1000:  # Limit history size
            self.detection_history.pop(0)
            
        return result
    
    def get_detection_stats(self):
        """Get statistics about recent detections"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'attack_detected_count': 0,
                'attack_percentage': 0,
                'avg_inference_time_ms': 0
            }
        
        attacks = [d for d in self.detection_history if d['is_attack']]
        
        return {
            'total_detections': len(self.detection_history),
            'attack_detected_count': len(attacks),
            'attack_percentage': len(attacks) / len(self.detection_history) * 100,
            'avg_inference_time_ms': np.mean([d['inference_time_ms'] for d in self.detection_history]),
            'attack_types': {
                attack_type: len([a for a in attacks if a['attack_type'] == attack_type])
                for attack_type in set(a['attack_type'] for a in attacks)
            } if attacks else {}
        }
    
    def update_threshold(self, new_threshold):
        """Update the detection threshold"""
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            return True
        return False
    
    def get_performance_stats(self):
        """Get model performance statistics"""
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'min_inference_time_ms': 0,
            }
            
        return {
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
        }
    
    def read_system_status(self, status_file):
        """
        Read system status data from a JSON file and detect attacks
        
        Args:
            status_file: Path to system status JSON file
        
        Returns:
            Detection results or None if file cannot be read
        """
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
                
            # Extract relevant SCADA measurements
            scada_data = {}
            
            # Example extraction - adjust based on your status file structure
            if 'measurements' in status_data:
                measurements = status_data['measurements']
                
                # Get the first RTU data
                for rtu_id, rtu_data in measurements.items():
                    scada_data['rtu_id'] = rtu_id
                    
                    # Bus measurements
                    for bus_id, bus_data in rtu_data.get('buses', {}).items():
                        scada_data['bus_number'] = int(bus_id)
                        scada_data['voltage_magnitude'] = bus_data.get('voltage_magnitude', 1.0)
                        scada_data['voltage_angle'] = bus_data.get('voltage_angle', 0.0)
                        break  # Just use the first bus for now
                    
                    # Frequency and status
                    scada_data['frequency'] = rtu_data.get('frequency', 50.0)
                    scada_data['status'] = rtu_data.get('status', 'UNKNOWN')
                    
                    # Power
                    scada_data['active_power'] = rtu_data.get('active_power', 0.0)
                    scada_data['reactive_power'] = rtu_data.get('reactive_power', 0.0)
                    
                    # Derived features
                    scada_data['voltage_deviation'] = abs(scada_data['voltage_magnitude'] - 1.0)
                    scada_data['frequency_deviation'] = abs(scada_data['frequency'] - 50.0)
                    
                    # Range checks
                    scada_data['voltage_in_normal_range'] = int(
                        0.95 <= scada_data['voltage_magnitude'] <= 1.05
                    )
                    scada_data['frequency_in_normal_range'] = int(
                        49.5 <= scada_data['frequency'] <= 50.5
                    )
                    
                    break  # Just use the first RTU for now
            
            # Run detection on the extracted data
            if scada_data:
                return self.detect(scada_data)
            else:
                logger.warning("No valid SCADA data found in status file")
                return None
                
        except Exception as e:
            logger.error(f"Error reading system status: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Basic test
    detector = AttackDetector()
    success = detector.load_model_and_transformers()
    
    if success:
        # Example data
        test_data = {
            'rtu_id': 'RTU1',
            'bus_number': 1,
            'voltage_magnitude': 1.12,  # Abnormal voltage
            'voltage_angle': 0.05,
            'frequency': 50.8,  # Slightly abnormal frequency
            'active_power': 100.0,
            'reactive_power': 20.0,
            'status': 'GOOD'
        }
        
        # Detect attack
        result = detector.detect(test_data)
        
        print("🔍 ATTACK DETECTION TEST")
        print("=" * 50)
        print(f"Is attack: {result['is_attack']}")
        print(f"Attack type: {result['attack_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    else:
        print("❌ Failed to load model and transformers")