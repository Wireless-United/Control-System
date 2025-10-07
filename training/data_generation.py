#!/usr/bin/env python3
"""
Data Generator for SCADA Attack Detection

This module generates labeled datasets for training machine learning models
to detect attacks on SCADA systems. It simulates normal operation and
different attack types, creating a balanced dataset for model training.
"""

import os
import numpy as np
import pandas as pd
import random
import json
import logging
import time
try:
    from tqdm import tqdm  # For progress bars
except ImportError:
    # Create a simple replacement if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AttackDataGenerator:
    """Generator for SCADA attack detection training data"""
    
    def __init__(self, output_dir=None):
        """
        Initialize the data generator
        
        Args:
            output_dir: Directory to save generated data
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir or os.path.join(base_dir, "data")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Normal operation parameters
        self.voltage_normal_mean = 1.0  # per unit
        self.voltage_normal_std = 0.01  # standard deviation
        self.frequency_normal_mean = 50.0  # Hz
        self.frequency_normal_std = 0.1  # standard deviation
        
        # RTU and bus configurations
        self.rtu_ids = ["RTU1", "RTU2", "RTU3", "RTU4"]
        self.bus_numbers = list(range(1, 11))  # 10 buses
        
        # Status values
        self.status_values = ["GOOD", "SUSPECT", "INVALID"]
        self.status_probabilities = [0.95, 0.03, 0.02]
        
        # Attack parameters
        self.attack_types = [
            "None",  # No attack
            "Voltage Attack",  # Voltage manipulation
            "Frequency Attack",  # Frequency manipulation
            "Combined Attack",  # Both voltage and frequency
            "Status Spoofing",  # Falsifying status values
            "Measurement Noise"  # Adding noise to measurements
        ]
        
        # Dataset storage
        self.dataset = None
    
    def generate_normal_data(self, n_samples):
        """Generate data representing normal operation"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Generate normal voltage and frequency
            voltage = np.random.normal(
                self.voltage_normal_mean,
                self.voltage_normal_std
            )
            frequency = np.random.normal(
                self.frequency_normal_mean,
                self.frequency_normal_std
            )
            
            # Generate random voltage angle
            voltage_angle = np.random.uniform(-0.1, 0.1)
            
            # Generate random power values
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges
            active_power_range = (50, 500) if is_gen else (-400, -50)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges
            reactive_power_range = (0, 200) if is_gen else (-200, 0) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Generate status based on probabilities
            status = np.random.choice(
                self.status_values,
                p=self.status_probabilities
            )
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 0,  # No attack
                "attack_type": "None",
                "attack_severity": "none"
            }
            
            data.append(data_point)
            
        return data
    
    def generate_voltage_attack_data(self, n_samples):
        """Generate data representing voltage manipulation attacks"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Determine severity
            severity = random.choice(["low", "medium", "high"])
            effect_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0}[severity]
            
            # Generate manipulated voltage
            # Either too high or too low
            if random.random() > 0.5:
                voltage = np.random.uniform(1.06, 1.06 + 0.15 * effect_multiplier)  # High voltage
            else:
                voltage = np.random.uniform(0.94 - 0.15 * effect_multiplier, 0.94)  # Low voltage
            
            # Normal frequency
            frequency = np.random.normal(
                self.frequency_normal_mean,
                self.frequency_normal_std
            )
            
            # Generate random voltage angle
            voltage_angle = np.random.uniform(-0.1, 0.1)
            
            # Generate random power values
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges
            active_power_range = (50, 500) if is_gen else (-400, -50)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges
            reactive_power_range = (0, 200) if is_gen else (-200, 0) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Calculate voltage offset from normal
            voltage_offset = voltage - self.voltage_normal_mean
            frequency_offset = 0  # No frequency change in voltage attack
            
            # Status more likely to be SUSPECT or INVALID
            status_probs = [0.6, 0.3, 0.1]  # More suspicious for attacks
            status = np.random.choice(
                self.status_values,
                p=status_probs
            )
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 1,  # Attack
                "attack_type": "Voltage Attack",
                "attack_severity": severity,
                "voltage_offset": round(voltage_offset, 3),
                "frequency_offset": frequency_offset
            }
            
            data.append(data_point)
            
        return data
    
    def generate_frequency_attack_data(self, n_samples):
        """Generate data representing frequency manipulation attacks"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Determine severity
            severity = random.choice(["low", "medium", "high"])
            effect_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0}[severity]
            
            # Normal voltage
            voltage = np.random.normal(
                self.voltage_normal_mean,
                self.voltage_normal_std
            )
            
            # Generate manipulated frequency
            # Either too high or too low
            if random.random() > 0.5:
                frequency = np.random.uniform(50.5, 50.5 + 1.5 * effect_multiplier)  # High frequency
            else:
                frequency = np.random.uniform(49.5 - 1.5 * effect_multiplier, 49.5)  # Low frequency
            
            # Generate random voltage angle
            voltage_angle = np.random.uniform(-0.1, 0.1)
            
            # Generate random power values
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges
            active_power_range = (50, 500) if is_gen else (-400, -50)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges
            reactive_power_range = (0, 200) if is_gen else (-200, 0) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Calculate offsets
            voltage_offset = 0  # No voltage change in frequency attack
            frequency_offset = frequency - self.frequency_normal_mean
            
            # Status more likely to be SUSPECT or INVALID
            status_probs = [0.6, 0.3, 0.1]  # More suspicious for attacks
            status = np.random.choice(
                self.status_values,
                p=status_probs
            )
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 1,  # Attack
                "attack_type": "Frequency Attack",
                "attack_severity": severity,
                "voltage_offset": voltage_offset,
                "frequency_offset": round(frequency_offset, 2)
            }
            
            data.append(data_point)
            
        return data
    
    def generate_combined_attack_data(self, n_samples):
        """Generate data representing combined voltage and frequency attacks"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Determine severity
            severity = random.choice(["low", "medium", "high"])
            effect_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0}[severity]
            
            # Generate manipulated voltage
            if random.random() > 0.5:
                voltage = np.random.uniform(1.06, 1.06 + 0.15 * effect_multiplier)  # High voltage
            else:
                voltage = np.random.uniform(0.94 - 0.15 * effect_multiplier, 0.94)  # Low voltage
            
            # Generate manipulated frequency
            if random.random() > 0.5:
                frequency = np.random.uniform(50.5, 50.5 + 1.5 * effect_multiplier)  # High frequency
            else:
                frequency = np.random.uniform(49.5 - 1.5 * effect_multiplier, 49.5)  # Low frequency
            
            # Generate random voltage angle
            voltage_angle = np.random.uniform(-0.1, 0.1)
            
            # Generate random power values
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges
            active_power_range = (50, 500) if is_gen else (-400, -50)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges
            reactive_power_range = (0, 200) if is_gen else (-200, 0) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Calculate offsets
            voltage_offset = voltage - self.voltage_normal_mean
            frequency_offset = frequency - self.frequency_normal_mean
            
            # Status more likely to be SUSPECT or INVALID
            status_probs = [0.5, 0.3, 0.2]  # Even more suspicious for combined attacks
            status = np.random.choice(
                self.status_values,
                p=status_probs
            )
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 1,  # Attack
                "attack_type": "Combined Attack",
                "attack_severity": severity,
                "voltage_offset": round(voltage_offset, 3),
                "frequency_offset": round(frequency_offset, 2)
            }
            
            data.append(data_point)
            
        return data
    
    def generate_status_spoofing_data(self, n_samples):
        """Generate data representing status spoofing attacks"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Determine severity
            severity = random.choice(["low", "medium", "high"])
            effect_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.5}[severity]
            
            # Generate slightly abnormal voltage and frequency
            # but mark status as GOOD
            if random.random() > 0.5:
                voltage = np.random.uniform(1.04, 1.04 + 0.06 * effect_multiplier)  # Slightly high voltage
            else:
                voltage = np.random.uniform(0.96 - 0.06 * effect_multiplier, 0.96)  # Slightly low voltage
            
            if random.random() > 0.5:
                frequency = np.random.uniform(50.4, 50.4 + 0.6 * effect_multiplier)  # Slightly high frequency
            else:
                frequency = np.random.uniform(49.6 - 0.6 * effect_multiplier, 49.6)  # Slightly low frequency
            
            # Generate random voltage angle
            voltage_angle = np.random.uniform(-0.1, 0.1)
            
            # Generate random power values
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges
            active_power_range = (50, 500) if is_gen else (-400, -50)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges
            reactive_power_range = (0, 200) if is_gen else (-200, 0) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Always good status despite abnormal readings
            status = "GOOD"
            
            # Calculate offsets
            voltage_offset = voltage - self.voltage_normal_mean
            frequency_offset = frequency - self.frequency_normal_mean
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 1,  # Attack
                "attack_type": "Status Spoofing",
                "attack_severity": severity,
                "voltage_offset": round(voltage_offset, 3),
                "frequency_offset": round(frequency_offset, 2)
            }
            
            data.append(data_point)
            
        return data
    
    def generate_measurement_noise_data(self, n_samples):
        """Generate data representing measurement noise attacks"""
        data = []
        
        for _ in range(n_samples):
            # Select a random RTU and bus
            rtu_id = random.choice(self.rtu_ids)
            bus_number = random.choice(self.bus_numbers)
            
            # Determine severity
            severity = random.choice(["low", "medium", "high"])
            effect_multiplier = {"low": 1.0, "medium": 2.0, "high": 3.0}[severity]
            
            # Start with normal values
            base_voltage = np.random.normal(
                self.voltage_normal_mean,
                self.voltage_normal_std
            )
            
            base_frequency = np.random.normal(
                self.frequency_normal_mean,
                self.frequency_normal_std
            )
            
            # Add random noise
            noise_level = np.random.uniform(0.02, 0.05) * effect_multiplier
            voltage = base_voltage + np.random.uniform(-noise_level, noise_level)
            frequency = base_frequency + np.random.uniform(-noise_level*10, noise_level*10)
            
            # Generate random voltage angle with noise
            voltage_angle = np.random.uniform(-0.15, 0.15)
            
            # Generate random power values with noise
            # Determine if it is a generator or load
            is_gen = random.choice([True, False])
            
            # Active power ranges with more variance
            active_power_range = (30, 550) if is_gen else (-450, -30)
            active_power = round(random.uniform(*active_power_range), 1)
            
            # Reactive power ranges with more variance
            reactive_power_range = (-50, 250) if is_gen else (-250, 50) 
            reactive_power = round(random.uniform(*reactive_power_range), 1)
            
            # Generate status with higher probability of non-GOOD
            status = np.random.choice(
                self.status_values,
                p=[0.7, 0.15, 0.15]  # Higher probability of issues
            )
            
            # Calculate offsets
            voltage_offset = voltage - self.voltage_normal_mean
            frequency_offset = frequency - self.frequency_normal_mean
            
            # Calculate derived features
            voltage_deviation = abs(voltage - self.voltage_normal_mean)
            frequency_deviation = abs(frequency - self.frequency_normal_mean)
            
            voltage_in_normal_range = int(0.95 <= voltage <= 1.05)
            frequency_in_normal_range = int(49.5 <= frequency <= 50.5)
            
            # Create data point
            data_point = {
                "timestamp": time.time(),
                "rtu_id": rtu_id,
                "bus_number": bus_number,
                "voltage_magnitude": round(voltage, 3),
                "voltage_angle": round(voltage_angle, 3),
                "frequency": round(frequency, 2),
                "active_power": active_power,
                "reactive_power": reactive_power,
                "status": status,
                "voltage_deviation": round(voltage_deviation, 4),
                "frequency_deviation": round(frequency_deviation, 4),
                "voltage_in_normal_range": voltage_in_normal_range,
                "frequency_in_normal_range": frequency_in_normal_range,
                "attack": 1,  # Attack
                "attack_type": "Measurement Noise",
                "attack_severity": severity,
                "voltage_offset": round(voltage_offset, 3),
                "frequency_offset": round(frequency_offset, 2)
            }
            
            data.append(data_point)
            
        return data
    
    def generate_dataset(self, n_samples=2000, balanced=True, verbose=True):
        """
        Generate a complete dataset with normal and attack data
        
        Args:
            n_samples: Total number of samples to generate
            balanced: Whether to balance classes
            verbose: Whether to print progress
            
        Returns:
            Pandas DataFrame with generated data
        """
        if balanced:
            # Calculate samples per class
            n_classes = len(self.attack_types)
            samples_per_class = n_samples // n_classes
            
            if verbose:
                print(f"Generating {samples_per_class} samples per class")
            
            # Generate normal data (class 0)
            normal_data = self.generate_normal_data(samples_per_class)
            
            if verbose:
                print(f"Generated {len(normal_data)} normal samples")
            
            # Generate attack data for each type
            voltage_attack_data = self.generate_voltage_attack_data(samples_per_class)
            if verbose:
                print(f"Generated {len(voltage_attack_data)} voltage attack samples")
            
            frequency_attack_data = self.generate_frequency_attack_data(samples_per_class)
            if verbose:
                print(f"Generated {len(frequency_attack_data)} frequency attack samples")
            
            combined_attack_data = self.generate_combined_attack_data(samples_per_class)
            if verbose:
                print(f"Generated {len(combined_attack_data)} combined attack samples")
            
            status_attack_data = self.generate_status_spoofing_data(samples_per_class)
            if verbose:
                print(f"Generated {len(status_attack_data)} status spoofing samples")
            
            noise_attack_data = self.generate_measurement_noise_data(samples_per_class)
            if verbose:
                print(f"Generated {len(noise_attack_data)} measurement noise samples")
            
            # Combine all data
            all_data = normal_data + voltage_attack_data + frequency_attack_data + \
                      combined_attack_data + status_attack_data + noise_attack_data
        else:
            # 70% normal, 30% attacks
            n_normal = int(n_samples * 0.7)
            n_attacks = n_samples - n_normal
            
            # Equal distribution among attack types
            n_attack_types = len(self.attack_types) - 1  # Excluding "None"
            samples_per_attack = n_attacks // n_attack_types
            
            if verbose:
                print(f"Generating {n_normal} normal samples and {samples_per_attack} samples per attack type")
            
            # Generate data
            normal_data = self.generate_normal_data(n_normal)
            voltage_attack_data = self.generate_voltage_attack_data(samples_per_attack)
            frequency_attack_data = self.generate_frequency_attack_data(samples_per_attack)
            combined_attack_data = self.generate_combined_attack_data(samples_per_attack)
            status_attack_data = self.generate_status_spoofing_data(samples_per_attack)
            noise_attack_data = self.generate_measurement_noise_data(samples_per_attack)
            
            # Combine all data
            all_data = normal_data + voltage_attack_data + frequency_attack_data + \
                      combined_attack_data + status_attack_data + noise_attack_data
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Convert timestamp to datetime format
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        
        # Store dataset
        self.dataset = df
        
        if verbose:
            print(f"Final dataset shape: {df.shape}")
            print(f"Attack distribution:\n{df['attack_type'].value_counts()}")
        
        return df
    
    def save_dataset(self, df=None, filename="attack_dataset.csv"):
        """Save the dataset to CSV"""
        if df is None:
            if self.dataset is None:
                logger.error("No dataset to save")
                return None
            df = self.dataset
            
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        
        # Also save a sample as JSON for reference
        sample_path = os.path.join(self.output_dir, "sample_data.json")
        sample_df = df.sample(10).copy()
        
        # Convert any non-serializable objects
        for col in sample_df.select_dtypes(include=['datetime64']).columns:
            sample_df[col] = sample_df[col].astype(str)
            
        sample_data = sample_df.to_dict(orient="records")
        
        with open(sample_path, "w") as f:
            json.dump(sample_data, f, indent=2)
            
        logger.info(f"Sample data saved to {sample_path}")
        
        # Save dataset stats as JSON
        stats = {
            "total_samples": len(df),
            "attack_count": int(df["attack"].sum()),
            "attack_percentage": float(df["attack"].mean() * 100),
            "attack_types": {k: int(v) for k, v in df["attack_type"].value_counts().to_dict().items()},
            "generated_at": pd.Timestamp.now().isoformat()
        }
        
        stats_file = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {stats_file}")
        
        return output_path

def main():
    """Main function to generate training data"""
    print("âš¡ SCADA ATTACK DETECTION - DATA GENERATION")
    print("=" * 60)
    
    # Set directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize generator
    generator = AttackDataGenerator(output_dir=data_dir)
    
    # Generate dataset
    print("\nGenerating dataset...")
    df = generator.generate_dataset(n_samples=2000, balanced=True)
    
    # Save dataset
    output_path = generator.save_dataset(df)
    
    print(f"\nâœ… DATA GENERATION COMPLETE")
    print(f"ðŸ“Š Dataset shape: {df.shape}")
    print(f"ðŸ“‚ Dataset saved to: {output_path}")
    
    # Print class distribution
    print("\nClass distribution:")
    print(df["attack_type"].value_counts())
    
    # Print sample data
    print("\nSample data:")
    print(df.sample(5)[["rtu_id", "bus_number", "voltage_magnitude", 
                        "frequency", "attack", "attack_type"]].to_string())

if __name__ == "__main__":
    main()
