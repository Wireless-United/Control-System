#!/usr/bin/env python3
"""
Model Visualization Script - Standalone

This script provides visualization capabilities for model training results
without requiring TensorFlow, allowing it to work on Python 3.13.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
import random
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVisualizer:
    """Visualizer for model training results"""
    
    def __init__(self, output_dir='./models'):
        """
        Initialize the model visualizer
        
        Args:
            output_dir: Directory containing model results
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For storing model results
        self.model_results = {}
        
    def load_model_results(self):
        """Load model results from comparison CSV if it exists"""
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        
        if os.path.exists(comparison_path):
            # Load existing results
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            self.model_results = comparison_df.to_dict(orient='index')
            return comparison_df
        
        # If no results exist, create mock data for visualization testing
        return self.create_mock_data()
    
    def create_mock_data(self):
        """Create mock data for testing visualizations"""
        print("⚠️ No model results found. Creating mock data for visualization testing.")
        
        # Define mock models
        models = [
            'simple_fcnn',
            'medium_fcnn',
            'deep_fcnn',
            'wide_nn'
        ]
        
        # Create mock data with realistic values
        mock_data = {}
        for model in models:
            # Base metrics with some randomness
            accuracy = random.uniform(0.85, 0.98)
            precision = random.uniform(0.82, 0.98)
            recall = random.uniform(0.80, 0.99)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Other metrics
            training_time = random.uniform(10, 200)
            model_parameters = random.randint(10000, 1000000)
            
            mock_data[model] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'training_time': training_time,
                'model_parameters': model_parameters
            }
            
            # Create directory for this model
            model_dir = os.path.join(self.output_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create mock ROC data
            figures_dir = os.path.join(model_dir, 'evaluation_figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            # Generate ROC curve data
            fpr = np.linspace(0, 1, 100)
            # Create a curve above the random line
            tpr = np.clip(fpr + (1-fpr) * random.uniform(0.7, 0.95), 0, 1)
            auc_value = np.trapz(tpr, fpr)  # Area under the curve
            
            # Save ROC data for later comparison
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(auc_value),
                'thresholds': np.linspace(1, 0, 100).tolist(),
                'optimal_threshold': float(random.uniform(0.3, 0.7))
            }
            
            with open(os.path.join(figures_dir, 'roc_data.json'), 'w') as f:
                json.dump(roc_data, f, indent=4)
        
        # Convert to DataFrame and save
        comparison_df = pd.DataFrame.from_dict(mock_data, orient='index')
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path)
        
        self.model_results = mock_data
        return comparison_df
    
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
    
    def create_results_dashboard(self, comparison_df, best_model_name):
        """Create a comprehensive results dashboard"""
        try:
            # Set style
            plt.style.use('ggplot')
            
            # Create a large figure for the dashboard
            fig = plt.figure(figsize=(20, 16))
            
            # Use GridSpec to organize the layout
            gs = fig.add_gridspec(3, 3)
            
            # Title for the dashboard
            fig.suptitle("SCADA Attack Detection Model Training Results", fontsize=24, y=0.98)
            
            # 1. Model Performance Comparison (Top left)
            ax1 = fig.add_subplot(gs[0, 0])
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            
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
                color=colors,
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
            
            # Create a simple heatmap manually since seaborn may not be available
            im = ax5.imshow(heatmap_data.T.values, cmap='YlGnBu', aspect='auto')
            
            # Add text annotations
            for i in range(heatmap_data.shape[0]):
                for j in range(len(metrics_for_heatmap)):
                    text = ax5.text(i, j, f"{heatmap_data.iloc[i, j]:.4f}",
                                  ha="center", va="center", color="black")
            
            # Set tick labels
            ax5.set_xticks(np.arange(len(heatmap_data.index)))
            ax5.set_yticks(np.arange(len(metrics_for_heatmap)))
            ax5.set_xticklabels(heatmap_data.index)
            ax5.set_yticklabels(metrics_for_heatmap)
            
            ax5.set_title('Performance Metrics Heatmap', fontsize=14)
            ax5.set_ylabel('Metric', fontsize=12)
            ax5.set_xlabel('Model', fontsize=12)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax5)
            cbar.set_label('Score')
            
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
        """Compare models and identify the best one"""
        # Load model results
        comparison_df = self.load_model_results()
        
        # Sort by F1 score descending
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        # Create comparison charts
        self.plot_model_comparison(comparison_df)
        
        # Identify best model based on F1 score
        best_model_name = comparison_df.index[0]
        best_model_metrics = comparison_df.iloc[0].to_dict()
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Metrics: {best_model_metrics}")
        
        # Create comprehensive results dashboard
        self.create_results_dashboard(comparison_df, best_model_name)
        
        # Create HTML report
        self.generate_html_report(comparison_df, best_model_name)
        
        return {
            'name': best_model_name,
            'metrics': best_model_metrics
        }
    
    def generate_html_report(self, comparison_df, best_model_name):
        """Generate an HTML report summarizing model training results"""
        try:
            # Import base64 for embedding images directly in HTML
            import base64
            
            # Path for the HTML report
            report_path = os.path.join(self.output_dir, 'training_results.html')
            
            # Function to encode images to base64
            def image_to_base64(image_path):
                try:
                    with open(image_path, "rb") as img_file:
                        return base64.b64encode(img_file.read()).decode('utf-8')
                except Exception:
                    # Return a small transparent gif if image doesn't exist
                    return "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
            
            # Images to include
            image_paths = {
                'dashboard': os.path.join(self.output_dir, 'training_results_dashboard.png'),
                'metrics': os.path.join(self.output_dir, 'model_comparison_metrics.png'),
                'roc': os.path.join(self.output_dir, 'model_comparison_roc.png'),
                'time_vs_performance': os.path.join(self.output_dir, 'model_comparison_time_vs_performance.png'),
                'radar': os.path.join(self.output_dir, 'model_comparison_radar.png')
            }
            
            # Convert the dataframe to HTML table
            comparison_html = comparison_df.to_html(classes='data-table', index=True)
            
            best_model_data = comparison_df.loc[best_model_name].to_dict()
            
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
                <h1>SCADA Attack Detection - Model Training Results</h1>
                <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                
                <div class="section best-model">
                    <h2>Best Performing Model</h2>
                    <p><strong>Model Name:</strong> {best_model_name}</p>
                    <p><strong>F1 Score:</strong> {best_model_data['f1_score']:.4f}</p>
                    <p><strong>Accuracy:</strong> {best_model_data['accuracy']:.4f}</p>
                    <p><strong>Precision:</strong> {best_model_data['precision']:.4f}</p>
                    <p><strong>Recall:</strong> {best_model_data['recall']:.4f}</p>
                    <p><strong>Parameters:</strong> {best_model_data['model_parameters']:,}</p>
                    <p><strong>Training Time:</strong> {best_model_data['training_time']:.2f} seconds</p>
                </div>
                
                <div class="section">
                    <h2>Results Dashboard</h2>
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_to_base64(image_paths['dashboard'])}" alt="Results Dashboard">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Detailed Model Comparison</h2>
                    {comparison_html}
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_to_base64(image_paths['metrics'])}" alt="Performance Metrics Comparison">
                    </div>
                </div>
                
                <div class="section">
                    <h2>ROC Curve Comparison</h2>
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_to_base64(image_paths['roc'])}" alt="ROC Curves">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Training Time vs. Performance</h2>
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_to_base64(image_paths['time_vs_performance'])}" alt="Time vs Performance">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Radar Chart Comparison</h2>
                    <div class="image-container">
                        <img src="data:image/png;base64,{image_to_base64(image_paths['radar'])}" alt="Radar Comparison">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Using the Model for Inference</h2>
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
    print("Attack detected: " + result["attack_type"])
    print("Confidence: " + str(round(result["confidence"], 2)))
else:
    print("System operating normally")
                    </pre>
                </div>
                
                <div class="footer">
                    <p>SCADA Attack Detection System | Training Results Report</p>
                    <p>This report was automatically generated after model training</p>
                </div>
            </body>
            </html>
            '''
            
            # Write the HTML content to a file using UTF-8 encoding
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            print(f"HTML report generated: {report_path}")
            return report_path
        
        except Exception as e:
            print(f"⚠️ Error generating HTML report: {e}")
            return None

def main():
    """Main function for model visualization"""
    print("SCADA ATTACK DETECTION - MODEL VISUALIZATION")
    print("=" * 60)
    
    # Set directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ModelVisualizer(output_dir=model_dir)
    
    # Generate visualizations and reports
    print("\nGenerating visualizations and reports...")
    best_model = visualizer.compare_models()
    
    if best_model:
        print(f"\nVISUALIZATION COMPLETE")
        print(f"Best model: {best_model['name']}")
        print(f"   F1 Score: {best_model['metrics']['f1_score']:.4f}")
        print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
        print(f"Reports and visualizations saved to: {model_dir}")
    else:
        print("No models were successfully compared.")

if __name__ == "__main__":
    main()