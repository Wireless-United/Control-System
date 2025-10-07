#!/usr/bin/env python3
"""
Model Training for SCADA Attack Detection

This module handles training different neural network architectures for
SCADA attack detection and evaluates their performance.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import logging

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AttackDetectionModelTrainer:
    """Trainer for SCADA attack detection models"""
    
    def __init__(self, data_dir='./data/processed', output_dir=None):
        """
        Initialize the model trainer
        
        Args:
            data_dir: Directory with preprocessed data
            output_dir: Directory to save models and results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(data_dir), 'models')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For storing model evaluation results
        self.model_results = {}
        
    def load_data(self, prefix='processed'):
        """Load preprocessed training and testing data"""
        try:
            # Load data splits
            X_train = joblib.load(os.path.join(self.data_dir, f"{prefix}_X_train.pkl"))
            y_train = joblib.load(os.path.join(self.data_dir, f"{prefix}_y_train.pkl"))
            X_test = joblib.load(os.path.join(self.data_dir, f"{prefix}_X_test.pkl"))
            y_test = joblib.load(os.path.join(self.data_dir, f"{prefix}_y_test.pkl"))
            
            # Load transformers
            transformers = joblib.load(os.path.join(self.data_dir, f"{prefix}_transformers.pkl"))
            
            logger.info(f"Data loaded successfully. X_train shape: {X_train.shape}")
            
            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'transformers': transformers
            }
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            return None
    
    def build_model_fcnn(self, input_dim, layers=None):
        """Build a fully connected neural network model"""
        if layers is None:
            layers = [128, 64, 32]
            
        inputs = Input(shape=(input_dim,))
        x = Dense(layers[0], activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Hidden layers
        for units in layers[1:]:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
        
    def build_model_deep_fcnn(self, input_dim):
        """Build a deeper fully connected neural network model"""
        inputs = Input(shape=(input_dim,))
        
        # Input layer
        x = Dense(256, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Hidden layers
        for units in [128, 128, 64, 64, 32]:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model_wide(self, input_dim):
        """Build a wide neural network model"""
        inputs = Input(shape=(input_dim,))
        
        # Input layer
        x = Dense(512, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Hidden layer
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, 
                   model_name, batch_size=32, epochs=100):
        """Train and evaluate a model"""
        # Create directory for this model
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Stop training if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Save best model
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            # Reduce learning rate when plateauing
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Plot and save model architecture
        try:
            plot_model(
                model,
                to_file=os.path.join(model_dir, 'model_architecture.png'),
                show_shapes=True,
                show_layer_names=True
            )
        except Exception as e:
            logger.warning(f"Could not plot model architecture: {e}")
        
        # Save model summary to text file
        with open(os.path.join(model_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
            # Redirect summary output to file
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Start training timer
        start_time = time.time()
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save training history
        joblib.dump(history.history, os.path.join(model_dir, 'training_history.pkl'))
        
        # Plot training history
        self.plot_training_history(history.history, model_dir)
        
        # Evaluate model on test data
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'training_time': training_time,
            'model_parameters': model.count_params()
        }
        
        # Save metrics
        self.model_results[model_name] = metrics
        
        # Generate evaluation plots
        self.plot_evaluation_metrics(y_test, y_pred, y_pred_prob, model_dir)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open(os.path.join(model_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
            
        # Save model configuration
        config = {
            'name': model_name,
            'input_shape': X_train.shape[1],
            'batch_size': batch_size,
            'epochs': epochs,
            'training_time_seconds': training_time,
            'metrics': metrics
        }
        
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Model {model_name} trained and evaluated.")
        logger.info(f"Metrics: {metrics}")
        
        return {
            'model': model,
            'metrics': metrics,
            'history': history.history
        }
    
    def plot_training_history(self, history, model_dir):
        """Plot and save training history graphs"""
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_history.png'))
        plt.close()
    
    def plot_evaluation_metrics(self, y_true, y_pred, y_prob, model_dir):
        """Plot and save evaluation metric visualizations"""
        # Create figures directory
        figures_dir = os.path.join(model_dir, 'evaluation_figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use a better color map for the confusion matrix
        cax = ax.matshow(cm, cmap='Blues', alpha=0.8)
        plt.colorbar(cax, ax=ax)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', 
                        size='xx-large', color='black' if cm[i, j] < cm.max()/1.5 else 'white')
                
        ax.set_xlabel('Predicted', fontsize=14)
        ax.set_ylabel('Actual', fontsize=14)
        ax.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax.xaxis.set_ticks([0, 1])
        ax.xaxis.set_ticklabels(['Normal', 'Attack'], fontsize=12)
        ax.yaxis.set_ticks([0, 1])
        ax.yaxis.set_ticklabels(['Normal', 'Attack'], fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)  # Random model line
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=16)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        # Find the optimal threshold - closest to top-left (0,1)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                label=f'Optimal threshold = {optimal_threshold:.3f}')
        ax.legend(loc="lower right", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save ROC data for later comparison
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': float(optimal_threshold)
        }
        
        with open(os.path.join(figures_dir, 'roc_data.json'), 'w') as f:
            json.dump(roc_data, f, indent=4)
        
        # 3. Precision-Recall Threshold Curve
        fig, ax = plt.subplots(figsize=(10, 8))
        
        precision = []
        recall = []
        f1_scores = []
        thresholds_to_plot = np.arange(0.1, 1.0, 0.05)
        
        for threshold in thresholds_to_plot:
            y_pred_t = (y_prob > threshold).astype(int)
            precision.append(precision_score(y_true, y_pred_t))
            recall.append(recall_score(y_true, y_pred_t))
            f1_scores.append(f1_score(y_true, y_pred_t))
            
        ax.plot(thresholds_to_plot, precision, 'b-', label='Precision')
        ax.plot(thresholds_to_plot, recall, 'g-', label='Recall')
        ax.plot(thresholds_to_plot, f1_scores, 'r-', label='F1 Score')
        
        # Add the default threshold (0.5)
        ax.axvline(x=0.5, color='k', linestyle='--', label='Default threshold (0.5)')
        
        # Mark the threshold with highest F1 score
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds_to_plot[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]
        
        ax.plot(best_threshold, best_f1, 'ro', markersize=8,
                label=f'Best F1 threshold = {best_threshold:.2f}')
        
        ax.set_xlabel('Classification Threshold', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Precision, Recall and F1 Score vs Threshold', fontsize=16)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'precision_recall_threshold.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create a histogram of prediction probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate normal and attack samples
        normal_probs = y_prob[y_true == 0].flatten()
        attack_probs = y_prob[y_true == 1].flatten()
        
        bins = np.linspace(0, 1, 20)
        ax.hist(normal_probs, bins=bins, alpha=0.5, color='green', label='Normal')
        ax.hist(attack_probs, bins=bins, alpha=0.5, color='red', label='Attack')
        
        ax.axvline(x=0.5, color='k', linestyle='--', label='Default threshold (0.5)')
        ax.set_xlabel('Prediction Probability', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Distribution of Prediction Probabilities', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def train_all_models(self, data_dict):
        """Train all model architectures"""
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")
        
        # Model configurations to train
        model_configs = [
            {
                'name': 'simple_fcnn',
                'build_fn': lambda: self.build_model_fcnn(
                    input_dim, 
                    layers=[64, 32]
                ),
                'batch_size': 32,
                'epochs': 100
            },
            {
                'name': 'medium_fcnn',
                'build_fn': lambda: self.build_model_fcnn(
                    input_dim, 
                    layers=[128, 64, 32]
                ),
                'batch_size': 32,
                'epochs': 100
            },
            {
                'name': 'deep_fcnn',
                'build_fn': lambda: self.build_model_deep_fcnn(input_dim),
                'batch_size': 32,
                'epochs': 100
            },
            {
                'name': 'wide_nn',
                'build_fn': lambda: self.build_model_wide(input_dim),
                'batch_size': 32,
                'epochs': 100
            }
        ]
        
        # Train all models
        for config in model_configs:
            logger.info(f"Training model: {config['name']}")
            
            model = config['build_fn']()
            
            self.train_model(
                model=model,
                X_train=X_train, 
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_name=config['name'],
                batch_size=config['batch_size'],
                epochs=config['epochs']
            )
    
    def create_results_dashboard(self, comparison_df, best_model_name):
        """Create a comprehensive results dashboard"""
        try:
            import seaborn as sns
            
            # Set style
            sns.set(style="whitegrid")
            
            # Create a large figure for the dashboard
            fig = plt.figure(figsize=(20, 16))
            
            # Use GridSpec to organize the layout
            gs = fig.add_gridspec(3, 3)
            
            # Title for the dashboard
            fig.suptitle("SCADA Attack Detection Model Training Results", fontsize=24, y=0.98)
            
            # 1. Model Performance Comparison (Top left)
            ax1 = fig.add_subplot(gs[0, 0])
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = sns.color_palette("husl", len(metrics))
            
            bar_width = 0.2
            index = np.arange(len(comparison_df.index))
            
            for i, metric in enumerate(metrics):
                ax1.bar(
                    index + i * bar_width,
                    comparison_df[metric],
                    bar_width,
                    label=metric.replace('_', ' ').title(),
                    color=colors[i],
                    edgecolor='white'
                )
            
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_title('Model Performance Metrics', fontsize=14)
            ax1.set_xticks(index + bar_width * 1.5)
            ax1.set_xticklabels(comparison_df.index, rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim(0, 1.05)
            
            # Highlight best model
            for i, model in enumerate(comparison_df.index):
                if model == best_model_name:
                    ax1.annotate(
                        '★ BEST MODEL',
                        xy=(i, 1.03),
                        xytext=(i, 1.03),
                        ha='center',
                        va='center',
                        fontsize=12,
                        fontweight='bold',
                        color='green'
                    )
            
            # 2. Training Time vs Complexity (Top middle)
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Get normalized model parameters for sizing the points
            max_params = comparison_df['model_parameters'].max()
            sizes = comparison_df['model_parameters'] / max_params * 500
            
            scatter = ax2.scatter(
                comparison_df['training_time'],
                comparison_df['f1_score'],
                s=sizes,
                c=range(len(comparison_df)),
                cmap='viridis',
                alpha=0.7,
                edgecolors='black'
            )
            
            # Add annotations for each model
            for i, model in enumerate(comparison_df.index):
                ax2.annotate(
                    model,
                    (comparison_df.loc[model, 'training_time'], comparison_df.loc[model, 'f1_score']),
                    fontsize=10,
                    ha='center',
                    va='bottom',
                    xytext=(0, 7),
                    textcoords='offset points'
                )
                
                # Highlight best model
                if model == best_model_name:
                    ax2.annotate(
                        '★',
                        (comparison_df.loc[model, 'training_time'], comparison_df.loc[model, 'f1_score']),
                        fontsize=20,
                        ha='center',
                        va='bottom',
                        xytext=(0, -20),
                        textcoords='offset points',
                        color='green'
                    )
            
            ax2.set_xlabel('Training Time (seconds)', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Time vs Performance', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add legend for bubble size
            ax2.annotate(
                f"Bubble size = parameter count",
                xy=(0.05, 0.02),
                xycoords='axes fraction',
                fontsize=10
            )
            
            # 3. Best Model Metrics (Top right)
            ax3 = fig.add_subplot(gs[0, 2])
            
            best_model_data = comparison_df.loc[best_model_name].to_dict()
            
            # Create bar chart for best model metrics
            metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [best_model_data[m] for m in metrics_to_show]
            
            bars = ax3.bar(
                metrics_to_show,
                values,
                color=sns.color_palette("husl", len(metrics_to_show)),
                edgecolor='white'
            )
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=10)
            
            ax3.set_ylim(0, 1.1)
            ax3.set_title(f'Best Model: {best_model_name}', fontsize=14)
            ax3.set_ylabel('Score', fontsize=12)
            ax3.grid(True, axis='y', alpha=0.3)
            
            # Add info text
            ax3.annotate(
                f"Parameters: {int(best_model_data['model_parameters']):,}\n"
                f"Training time: {best_model_data['training_time']:.2f} seconds",
                xy=(0.5, 0.02),
                xycoords='axes fraction',
                fontsize=10,
                ha='center'
            )
            
            # 4. Model Complexity Comparison (Middle left)
            ax4 = fig.add_subplot(gs[1, 0])
            
            # Create horizontal bar chart for model complexity
            bars = ax4.barh(
                comparison_df.index[::-1],
                comparison_df['model_parameters'][::-1],
                color='purple',
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax4.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{int(width):,}',
                        va='center', fontsize=10)
            
            ax4.set_xlabel('Parameter Count', fontsize=12)
            ax4.set_title('Model Complexity', fontsize=14)
            ax4.grid(True, axis='x', alpha=0.3)
            
            # 5. Heatmap of metrics (Middle)
            ax5 = fig.add_subplot(gs[1, 1])
            
            # Create a heatmap for all metrics
            metrics_for_heatmap = ['accuracy', 'precision', 'recall', 'f1_score']
            heatmap_data = comparison_df[metrics_for_heatmap]
            
            sns.heatmap(
                heatmap_data.T,
                annot=True,
                fmt='.4f',
                cmap='YlGnBu',
                cbar_kws={'label': 'Score'},
                ax=ax5
            )
            
            ax5.set_title('Performance Metrics Heatmap', fontsize=14)
            ax5.set_ylabel('Metric', fontsize=12)
            ax5.set_xlabel('Model', fontsize=12)
            
            # 6. Training Time Comparison (Middle right)
            ax6 = fig.add_subplot(gs[1, 2])
            
            bars = ax6.barh(
                comparison_df.index[::-1],
                comparison_df['training_time'][::-1],
                color='teal',
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax6.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}s',
                        va='center', fontsize=10)
            
            ax6.set_xlabel('Time (seconds)', fontsize=12)
            ax6.set_title('Training Time', fontsize=14)
            ax6.grid(True, axis='x', alpha=0.3)
            
            # 7. Summary text (Bottom left)
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.axis('off')
            
            summary_text = (
                f"Model Training Summary\n"
                f"======================\n\n"
                f"• Models Trained: {len(comparison_df)}\n"
                f"• Best Model: {best_model_name}\n"
                f"• Best F1 Score: {comparison_df.loc[best_model_name, 'f1_score']:.4f}\n"
                f"• Best Accuracy: {comparison_df.loc[best_model_name, 'accuracy']:.4f}\n"
                f"• Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"The model parameters and training scripts can be \n"
                f"found in the 'models' directory."
            )
            
            ax7.text(0, 1.0, summary_text, fontsize=12, va='top', fontfamily='monospace')
            
            # 8. ROC Comparison (Bottom middle and right)
            ax8 = fig.add_subplot(gs[2, 1:])
            
            for model_name in comparison_df.index:
                model_dir = os.path.join(self.output_dir, model_name)
                roc_data_path = os.path.join(model_dir, 'evaluation_figures', 'roc_data.json')
                
                if os.path.exists(roc_data_path):
                    with open(roc_data_path, 'r') as f:
                        roc_data = json.load(f)
                        
                        line_style = '-'
                        line_width = 2
                        
                        # Highlight best model with thicker line
                        if model_name == best_model_name:
                            line_style = '-'
                            line_width = 3
                            
                        ax8.plot(roc_data['fpr'], roc_data['tpr'], 
                                line_style,
                                lw=line_width, 
                                label=f"{model_name} (AUC = {roc_data['auc']:.3f})")
            
            ax8.plot([0, 1], [0, 1], 'k--', lw=2)
            ax8.set_xlim([0.0, 1.0])
            ax8.set_ylim([0.0, 1.05])
            ax8.set_xlabel('False Positive Rate', fontsize=12)
            ax8.set_ylabel('True Positive Rate', fontsize=12)
            ax8.set_title('ROC Curves Comparison', fontsize=14)
            ax8.legend(loc="lower right", fontsize=10)
            ax8.grid(True, alpha=0.3)
            
            # Layout adjustment
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save the dashboard
            plt.savefig(os.path.join(self.output_dir, 'training_results_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Created results dashboard visualization")
            
        except Exception as e:
            logger.error(f"Error creating results dashboard: {e}")
    
    def compare_models(self):
        """Compare trained models and identify the best one"""
        if not self.model_results:
            logger.warning("No model results to compare")
            return None
        
        # Create comparison table
        comparison_df = pd.DataFrame.from_dict(self.model_results, orient='index')
        
        # Sort by F1 score descending
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Save comparison to CSV
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path)
        
        # Save comparison to formatted text
        with open(os.path.join(self.output_dir, 'model_comparison.txt'), 'w') as f:
            f.write("MODEL COMPARISON\n")
            f.write("===============\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(comparison_df.to_string())
        
        # Create comparison charts
        self.plot_model_comparison(comparison_df)
        
        # Identify best model based on F1 score
        best_model_name = comparison_df.index[0]
        best_model_metrics = comparison_df.iloc[0].to_dict()
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Metrics: {best_model_metrics}")
        
        # Create symlink to best model for easy reference
        best_model_dir = os.path.join(self.output_dir, best_model_name)
        best_model_link = os.path.join(self.output_dir, 'best_model')
        
        # On Windows, symlinks require admin rights, so we'll create a text file pointer instead
        with open(os.path.join(self.output_dir, 'best_model.txt'), 'w') as f:
            f.write(f"Best model: {best_model_name}\n")
            f.write(f"F1 Score: {best_model_metrics['f1_score']:.4f}\n")
            f.write(f"Accuracy: {best_model_metrics['accuracy']:.4f}\n")
            f.write(f"Directory: {best_model_dir}")
        
        # Create comprehensive results dashboard
        self.create_results_dashboard(comparison_df, best_model_name)
        
        return {
            'name': best_model_name,
            'metrics': best_model_metrics,
            'model_path': os.path.join(best_model_dir, 'best_model.h5')
        }
    
    def plot_model_comparison(self, comparison_df):
        """Create visualizations comparing model performance"""
        # Set style for better visualizations
        plt.style.use('ggplot')
        
        # Metrics to compare
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        # 1. Create radar chart comparing all metrics
        fig = plt.figure(figsize=(12, 10))
        
        # Number of metrics
        N = len(metrics)
        
        # Create angle for each metric (evenly spaced)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        
        for i, model in enumerate(comparison_df.index):
            # Get model metrics
            model_metrics = comparison_df.loc[model, metrics].tolist()
            model_metrics += model_metrics[:1]  # Close the polygon
            
            # Plot the model metrics
            ax.plot(angles, model_metrics, 'o-', linewidth=2, 
                    label=model, alpha=0.8)
            ax.fill(angles, model_metrics, alpha=0.1)
        
        # Set labels for each angle
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics])
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Metrics Comparison', size=16, y=1.08)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create grouped bar chart comparing metrics
        plt.figure(figsize=(14, 8))
        
        bar_width = 0.2
        index = np.arange(len(comparison_df.index))
        
        for i, metric in enumerate(metrics):
            plt.bar(
                index + i * bar_width,
                comparison_df[metric],
                bar_width,
                label=metric.replace('_', ' ').title(),
                color=colors[i],
                edgecolor='white'
            )
        
        plt.xlabel('Model Architecture', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xticks(index + bar_width * 1.5, comparison_df.index, rotation=45, ha='right', fontsize=12)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Create time vs accuracy scatter plot with size based on model complexity
        plt.figure(figsize=(12, 8))
        
        # Get normalized model parameters for sizing the points
        max_params = comparison_df['model_parameters'].max()
        sizes = comparison_df['model_parameters'] / max_params * 1000
        
        # Create scatter plot
        scatter = plt.scatter(
            comparison_df['training_time'],
            comparison_df['f1_score'],
            s=sizes,
            c=comparison_df['accuracy'],
            cmap='viridis',
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Accuracy', fontsize=12)
        
        # Add annotations for each model
        for i, model in enumerate(comparison_df.index):
            plt.annotate(
                model,
                (comparison_df.loc[model, 'training_time'], comparison_df.loc[model, 'f1_score']),
                fontsize=10,
                ha='center',
                va='bottom',
                xytext=(0, 7),
                textcoords='offset points'
            )
        
        plt.xlabel('Training Time (seconds)', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.title('Training Time vs F1 Score', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Add an explanation for the bubble size
        plt.figtext(0.15, 0.02, f"Bubble size represents model parameter count (max = {int(max_params):,} parameters)",
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_time_vs_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create training time comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df.index, comparison_df['training_time'], color='teal', alpha=0.7)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}s',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model Architecture', fontsize=14)
        plt.ylabel('Training Time (seconds)', fontsize=14)
        plt.title('Training Time Comparison', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Create model complexity comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df.index, comparison_df['model_parameters'], color='purple', alpha=0.7)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model Architecture', fontsize=14)
        plt.ylabel('Parameter Count', fontsize=14)
        plt.title('Model Complexity Comparison', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_complexity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Create a ROC comparison chart with all models
        # We need to reload each model and evaluate it
        try:
            plt.figure(figsize=(10, 8))
            
            for model_name in comparison_df.index:
                model_dir = os.path.join(self.output_dir, model_name)
                roc_data_path = os.path.join(model_dir, 'evaluation_figures', 'roc_data.json')
                
                if os.path.exists(roc_data_path):
                    with open(roc_data_path, 'r') as f:
                        roc_data = json.load(f)
                        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                                lw=2, label=f"{model_name} (AUC = {roc_data['auc']:.3f})")
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves for All Models', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison_roc.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create ROC comparison: {e}")

def generate_html_report(model_dir, best_model):
    """Generate an HTML report summarizing model training results"""
    try:
        # Import base64 for embedding images directly in HTML
        import base64
        
        # Path for the HTML report
        report_path = os.path.join(model_dir, 'training_results.html')
        
        # Function to encode images to base64
        def image_to_base64(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        
        # Images to include
        image_paths = {
            'dashboard': os.path.join(model_dir, 'training_results_dashboard.png'),
            'metrics': os.path.join(model_dir, 'model_comparison_metrics.png'),
            'roc': os.path.join(model_dir, 'model_comparison_roc.png'),
            'time_vs_performance': os.path.join(model_dir, 'model_comparison_time_vs_performance.png'),
            'radar': os.path.join(model_dir, 'model_comparison_radar.png')
        }
        
        # Read comparison CSV
        comparison_path = os.path.join(model_dir, 'model_comparison.csv')
        comparison_df = pd.read_csv(comparison_path)
        
        # Convert the dataframe to HTML table
        comparison_html = comparison_df.to_html(classes='data-table', index=True)
        
        # HTML template
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SCADA Attack Detection - Model Training Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .best-model {{
                    background-color: #e8f8f5;
                    padding: 20px;
                    border-radius: 5px;
                    border-left: 5px solid #2ecc71;
                }}
                .best-model h2 {{
                    color: #27ae60;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #f2f2f2;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .data-table tr:hover {{
                    background-color: #f1f1f1;
                }}
                .code {{
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: monospace;
                    overflow-x: auto;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    border-top: 1px solid #ddd;
                    padding-top: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>🧠 SCADA Attack Detection - Model Training Results</h1>
            <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            
            <div class="section best-model">
                <h2>⭐ Best Performing Model</h2>
                <p><strong>Model Name:</strong> {best_model['name']}</p>
                <p><strong>F1 Score:</strong> {best_model['metrics']['f1_score']:.4f}</p>
                <p><strong>Accuracy:</strong> {best_model['metrics']['accuracy']:.4f}</p>
                <p><strong>Precision:</strong> {best_model['metrics']['precision']:.4f}</p>
                <p><strong>Recall:</strong> {best_model['metrics']['recall']:.4f}</p>
                <p><strong>Parameters:</strong> {best_model['metrics']['model_parameters']:,}</p>
                <p><strong>Training Time:</strong> {best_model['metrics']['training_time']:.2f} seconds</p>
                <p><strong>Model Path:</strong> <code>{best_model['model_path']}</code></p>
            </div>
            
            <div class="section">
                <h2>📊 Results Dashboard</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(image_paths['dashboard'])}" alt="Results Dashboard">
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Detailed Model Comparison</h2>
                {comparison_html}
            </div>
            
            <div class="section">
                <h2>📈 Performance Metrics</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(image_paths['metrics'])}" alt="Performance Metrics Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 ROC Curve Comparison</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(image_paths['roc'])}" alt="ROC Curves">
                </div>
            </div>
            
            <div class="section">
                <h2>⚖️ Training Time vs. Performance</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(image_paths['time_vs_performance'])}" alt="Time vs Performance">
                </div>
            </div>
            
            <div class="section">
                <h2>🕸️ Radar Chart Comparison</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(image_paths['radar'])}" alt="Radar Comparison">
                </div>
            </div>
            
            <div class="section">
                <h2>💻 Using the Model for Inference</h2>
                <p>To use this model for inference, use the following code template:</p>
                <pre class="code">
from inference.attack_detector import AttackDetector

# Initialize the detector
detector = AttackDetector()

# Load the model
detector.load_model_and_transformers()

# Detect attacks on system data
result = detector.detect()

if result["is_attack"]:
    print("🚨 Attack detected: " + result["attack_type"])
    print("Confidence: " + str(round(result["confidence"], 2)))
else:
    print("✅ System operating normally")
                </pre>
            </div>
            
            <div class="footer">
                <p>SCADA Attack Detection System | Training Results Report</p>
                <p>This report was automatically generated after model training</p>
            </div>
        </body>
        </html>
        '''
        
        # Write the HTML content to a file
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        print(f"📄 HTML report generated: {report_path}")
        return report_path
    
    except Exception as e:
        print(f"⚠️ Error generating HTML report: {e}")
        return None

def main():
    """Main training function"""
    print("🧠 SCADA ATTACK DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Set directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    model_dir = os.path.join(base_dir, "models")
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = AttackDetectionModelTrainer(data_dir=data_dir, output_dir=model_dir)
    
    # Load data
    print("\n1️⃣ Loading preprocessed data...")
    data_dict = trainer.load_data()
    if data_dict is None:
        print("❌ Failed to load preprocessed data. Exiting.")
        return
    
    # Train models
    print("\n2️⃣ Training models...")
    trainer.train_all_models(data_dict)
    
    # Compare models
    print("\n3️⃣ Comparing models and selecting best...")
    best_model = trainer.compare_models()
    
    if best_model:
        print(f"\n✅ TRAINING COMPLETE")
        print(f"📊 Best model: {best_model['name']}")
        print(f"   F1 Score: {best_model['metrics']['f1_score']:.4f}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
        print(f"📂 Models saved to: {model_dir}")
        
        # Generate HTML report
        print("\n4️⃣ Generating comprehensive results report...")
        report_path = generate_html_report(model_dir, best_model)
        if report_path:
            print(f"📄 Results report saved to: {report_path}")
    else:
        print("❌ No models were successfully trained and compared.")

if __name__ == "__main__":
    main()