# SCADA Attack Detection System

This system provides machine learning-based detection of attacks on SCADA systems. It monitors real-time measurements from power system components and identifies potential attacks based on patterns learned from training data.

## System Components

The attack detection system consists of the following components:

1. **Data Generation**: Creates synthetic labeled data for training the detection models
2. **Data Preprocessing**: Cleans and prepares the raw data for model training
3. **Model Training**: Trains and evaluates multiple neural network architectures
4. **Inference Engine**: Real-time attack detection using the trained model
5. **Detection UI**: Visual interface for monitoring system status and attack alerts

## Directory Structure

```
training/
  ├── data_generation.py     # Generates synthetic training data
  ├── data_preprocessing.py  # Preprocesses data for model training
  ├── model_training.py      # Trains and evaluates multiple models
  ├── data/                  # Raw and processed training data
  └── models/                # Trained model files

inference/
  ├── attack_detector.py     # Core attack detection module
  └── detection_ui.py        # Streamlit-based detection interface

launch_attack_detection.py   # Launcher script for the whole system
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (automatically installed by launcher):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - streamlit
  - joblib

### Quick Start

1. Run the launcher script with the desired action:

```bash
python launch_attack_detection.py --action full
```

This will:

- Generate training data
- Preprocess the data
- Train multiple detection models
- Optionally launch the detection UI

### Individual Components

You can run individual components using the launcher:

- Generate training data:

  ```bash
  python launch_attack_detection.py --action generate
  ```

- Preprocess data:

  ```bash
  python launch_attack_detection.py --action preprocess
  ```

- Train detection models:

  ```bash
  python launch_attack_detection.py --action train
  ```

- Launch detection UI:
  ```bash
  python launch_attack_detection.py --action detect
  ```

## Detection System Details

### Attack Types Detected

The system can detect the following types of attacks:

1. **Voltage Attacks**: Manipulation of voltage measurements
2. **Frequency Attacks**: Manipulation of frequency measurements
3. **Combined Attacks**: Simultaneous manipulation of multiple measurements
4. **Status Spoofing**: False reporting of system status
5. **Measurement Noise**: Introduction of noise to confuse operators

### Model Architecture

The system trains multiple neural network architectures:

- Simple feed-forward neural network
- Medium-depth neural network
- Deep neural network
- Wide neural network

The best-performing model is automatically selected based on F1 score.

### Real-time Detection

The detection UI provides:

- Real-time monitoring of system measurements
- Visual alerts for detected attacks
- Historical trends of measurements and detections
- Attack type classification

## Integration with Existing System

The attack detection system integrates with the existing SCADA monitoring system by reading measurements from the shared system status file (`system_status.json`). This allows it to monitor the same data that operators see in the SCADA interface.

## Performance Metrics

The detection models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Development

### Adding New Attack Types

To add new attack types:

1. Update `AttackDataGenerator` in `data_generation.py`
2. Add a new method to generate the specific attack pattern
3. Include the new attack type in the `generate_dataset` method
4. Retrain the models

### Customizing Detection Thresholds

The detection threshold can be adjusted in the UI to balance between sensitivity (detecting more potential attacks) and specificity (reducing false alarms).

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that the system status file exists and has the expected format
3. Verify that models have been trained and saved correctly
4. Check the console output for error messages

## License

This project is proprietary and confidential. Unauthorized copying, transferring, or reproduction of the contents is strictly prohibited.

## Acknowledgements

This system was developed as part of the Control Systems Security project.
