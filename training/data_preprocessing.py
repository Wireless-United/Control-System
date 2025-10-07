#!/usr/bin/env python3
"""
Data Preprocessing for Attack Detection

This module handles data preprocessing for the SCADA attack detection model, including:
- Data cleaning
- Feature engineering
- Normalization
- Train/test splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocessor for SCADA attack detection data"""
    
    def __init__(self, data_dir='./data', output_dir=None):
        self.data_dir = data_dir
        self.output_dir = output_dir or data_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize transformers
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, filename='attack_dataset.csv'):
        """Load dataset from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Dataset loaded: {len(df)} samples from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def explore_data(self, df):
        """Explore and visualize dataset"""
        # Print basic information
        print("\n📊 DATASET EXPLORATION")
        print("=" * 50)
        print(f"Dataset shape: {df.shape}")
        print("\nData types:")
        print(df.dtypes)
        
        print("\nBasic statistics:")
        print(df.describe())
        
        print("\nMissing values:")
        print(df.isnull().sum())
        
        print("\nClass distribution:")
        print(df['attack'].value_counts())
        print(f"Attack percentage: {df['attack'].mean()*100:.2f}%")
        
        # Create exploratory visualizations
        output_path = os.path.join(self.output_dir, 'exploratory')
        os.makedirs(output_path, exist_ok=True)
        
        # 1. Plot attack distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='attack_type')
        plt.title('Attack Type Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'attack_distribution.png'))
        
        # 2. Voltage vs Frequency scatter plot colored by attack
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='voltage_magnitude', y='frequency', hue='attack')
        plt.title('Voltage vs Frequency by Attack Status')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'voltage_frequency_scatter.png'))
        
        # 3. Distribution of voltage by attack type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='attack_type', y='voltage_magnitude')
        plt.title('Voltage Magnitude Distribution by Attack Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'voltage_by_attack_boxplot.png'))
        
        # 4. Distribution of frequency by attack type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='attack_type', y='frequency')
        plt.title('Frequency Distribution by Attack Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'frequency_by_attack_boxplot.png'))
        
        # 5. Correlation heatmap
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=['number'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'correlation_heatmap.png'))
        
        logger.info(f"Exploratory visualizations saved to {output_path}")
    
    def preprocess_data(self, df):
        """Preprocess data for model training"""
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # 1. Handle missing values
        logger.info(f"Missing values before imputation: {processed_df.isnull().sum().sum()}")
        processed_df.fillna({
            'voltage_magnitude': 1.0,
            'frequency': 50.0,
            'status': 'GOOD'
        }, inplace=True)
        
        # 2. Feature selection
        self.feature_columns = [
            'rtu_id', 'bus_number', 
            'voltage_magnitude', 'voltage_angle', 
            'active_power', 'reactive_power', 'frequency',
            'status',  # Quality indicator
            'voltage_deviation', 'frequency_deviation',
            'voltage_in_normal_range', 'frequency_in_normal_range'
        ]
        
        # Keep only the features we want
        X = processed_df[self.feature_columns].copy()
        
        # Labels
        y = processed_df['attack'].astype(int)
        
        # 3. Split categorical and numerical features
        self.categorical_columns = ['rtu_id', 'bus_number', 'status']
        self.numerical_columns = [col for col in self.feature_columns 
                                  if col not in self.categorical_columns]
        
        # 4. Create feature encoders and scalers
        X_transformed = self._transform_features(X, fit=True)
        
        # Return preprocessed features and labels
        return X_transformed, y
    
    def _transform_features(self, X, fit=False):
        """Transform features using fitted or new transformers"""
        X_processed = X.copy()
        
        # Process categorical features
        for col in self.categorical_columns:
            if fit or col not in self.encoders:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
            
            # Transform
            col_encoded = self.encoders[col].transform(X[[col]])
            
            # Get feature names
            feature_names = [f"{col}_{int(i)}" for i in range(col_encoded.shape[1])]
            
            # Add encoded features
            for i, name in enumerate(feature_names):
                X_processed[name] = col_encoded[:, i]
            
            # Drop original column
            X_processed.drop(col, axis=1, inplace=True)
        
        # Process numerical features
        if fit:
            scaler = StandardScaler()
            scaler.fit(X_processed[self.numerical_columns])
            self.scalers['numerical'] = scaler
        
        # Scale numerical features
        X_processed[self.numerical_columns] = self.scalers['numerical'].transform(
            X_processed[self.numerical_columns]
        )
        
        return X_processed
    
    def prepare_train_test_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        # Save data splits
        train_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        return train_data
    
    def save_preprocessed_data(self, train_data, filename_prefix='processed'):
        """Save preprocessed data and transformers"""
        # Save data splits
        for name, data in train_data.items():
            filepath = os.path.join(self.output_dir, f"{filename_prefix}_{name}.pkl")
            joblib.dump(data, filepath)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save transformers
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }
        
        transformer_path = os.path.join(self.output_dir, f"{filename_prefix}_transformers.pkl")
        joblib.dump(transformers, transformer_path)
        logger.info(f"Saved transformers to {transformer_path}")
        
        # Save feature names for reference
        feature_names = list(train_data['X_train'].columns)
        with open(os.path.join(self.output_dir, f"{filename_prefix}_feature_names.txt"), 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        
        # Generate preprocessing report
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'X_train': train_data['X_train'].shape,
                'X_test': train_data['X_test'].shape
            },
            'class_distribution': {
                'train': {'normal': int((train_data['y_train'] == 0).sum()), 
                          'attack': int((train_data['y_train'] == 1).sum())},
                'test': {'normal': int((train_data['y_test'] == 0).sum()), 
                         'attack': int((train_data['y_test'] == 1).sum())}
            },
            'feature_count': len(feature_names),
            'categorical_features': len(self.categorical_columns),
            'numerical_features': len(self.numerical_columns)
        }
        
        with open(os.path.join(self.output_dir, f"{filename_prefix}_report.txt"), 'w') as f:
            f.write("PREPROCESSING REPORT\n")
            f.write("===================\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            f.write("Data Shapes:\n")
            for name, shape in report['data_shape'].items():
                f.write(f"  {name}: {shape}\n")
            
            f.write("\nClass Distribution:\n")
            for split, dist in report['class_distribution'].items():
                total = dist['normal'] + dist['attack']
                f.write(f"  {split}: {dist['normal']} normal ({dist['normal']/total*100:.1f}%), ")
                f.write(f"{dist['attack']} attack ({dist['attack']/total*100:.1f}%)\n")
            
            f.write(f"\nFeature Count: {report['feature_count']}\n")
            f.write(f"  Categorical features: {report['categorical_features']}\n")
            f.write(f"  Numerical features: {report['numerical_features']}\n")
        
        logger.info(f"Preprocessing completed and saved to {self.output_dir}")

def main():
    """Main preprocessing function"""
    print("🔍 SCADA ATTACK DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Set directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_dir=data_dir, output_dir=processed_dir)
    
    # Load data
    print("\n1️⃣ Loading dataset...")
    df = preprocessor.load_data()
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Explore data
    print("\n2️⃣ Exploring and visualizing data...")
    preprocessor.explore_data(df)
    
    # Preprocess data
    print("\n3️⃣ Preprocessing data...")
    X, y = preprocessor.preprocess_data(df)
    
    # Split data
    print("\n4️⃣ Splitting into training and testing sets...")
    train_data = preprocessor.prepare_train_test_data(X, y, test_size=0.2)
    
    # Save preprocessed data
    print("\n5️⃣ Saving preprocessed data...")
    preprocessor.save_preprocessed_data(train_data)
    
    print("\n✅ PREPROCESSING COMPLETE")
    print(f"📂 Processed data saved to: {processed_dir}")

if __name__ == "__main__":
    main()