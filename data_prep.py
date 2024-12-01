# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.interpolate import interp1d
import joblib
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import logging
import warnings
import argparse
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SENSOR_RANGES = {
    'heart_rate': {'sensors': [2], 'range': (30, 220), 'unit': 'bpm'},
    'temperature': {'sensors': [3, 13, 23], 'range': (20, 40), 'unit': '°C'},
    'acceleration': {'sensors': [4,5,6, 14,15,16, 24,25,26], 'range': (-50, 50), 'unit': 'm/s²'},
    'gyroscope': {'sensors': [7,8,9, 17,18,19, 27,28,29], 'range': (-20, 20), 'unit': 'rad/s'},
    'magnetometer': {'sensors': [10,11,12, 20,21,22, 30,31,32], 'range': (-100, 100), 'unit': 'µT'}
}

ACTIVITY_NAMES = {
    1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking very slow',
    5: 'Normal walking', 6: 'Nordic walking', 7: 'Running',
    8: 'Ascending stairs', 9: 'Descending stairs', 10: 'Cycling',
    11: 'Ironing', 12: 'Vacuum cleaning', 13: 'Rope jumping',
    14: 'Playing soccer'
}

class SensorDataAnalyzer:
    """Main class for analyzing and preprocessing sensor data."""
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.learning_path = os.path.join(root_path, 'LS')
        self.test_path = os.path.join(root_path, 'TS')
        self.sensors = None
        self.activities = None
        self.subjects = None
        
    def load_data(self, dataset_type: str = 'learning') -> None:
        """Load sensor data, activities, and subject IDs."""
        logger.info(f"Loading {dataset_type} dataset...")
        path = self.learning_path if dataset_type == 'learning' else self.test_path
        
        # Load sensors
        self.sensors = {}
        for i in tqdm(range(2, 33), desc="Loading sensors"):
            file_prefix = 'LS' if dataset_type == 'learning' else 'TS'
            sensor_file = os.path.join(path, f'{file_prefix}_sensor_{i}.txt')
            self.sensors[i] = np.loadtxt(sensor_file)
            
        # Load activities and subjects
        self.subjects = np.loadtxt(os.path.join(path, 'subject_Id.txt'))
        if dataset_type == 'learning':
            self.activities = np.loadtxt(os.path.join(path, 'activity_Id.txt'))
            
    def analyze_missing_values(self) -> Dict:
        """Analyze missing values in sensor data."""
        logger.info("Analyzing missing values...")
        missing_stats = {}
        
        for sensor_id, data in self.sensors.items():
            missing_samples = np.sum(np.all(data == -999999.99, axis=1))
            missing_points = np.sum(data == -999999.99)
            
            missing_stats[sensor_id] = {
                'total_samples': data.shape[0],
                'missing_samples': missing_samples,
                'missing_points': missing_points,
                'missing_percentage': (missing_points / data.size) * 100
            }
            
        return missing_stats
    
    def analyze_outliers(self) -> Dict:
        """Analyze outliers based on physical constraints and statistical methods."""
        logger.info("Analyzing outliers...")
        outlier_stats = {}
        
        for sensor_type, info in SENSOR_RANGES.items():
            for sensor_id in info['sensors']:
                data = self.sensors[sensor_id]
                valid_data = data[data != -999999.99]
                
                # Physical constraints
                physical_outliers = valid_data[
                    (valid_data < info['range'][0]) | 
                    (valid_data > info['range'][1])
                ]
                
                # Statistical outliers (IQR method)
                q1, q3 = np.percentile(valid_data, [25, 75])
                iqr = q3 - q1
                stat_outliers = valid_data[
                    (valid_data < q1 - 1.5*iqr) | 
                    (valid_data > q3 + 1.5*iqr)
                ]
                
                outlier_stats[sensor_id] = {
                    'sensor_type': sensor_type,
                    'physical_outliers': len(physical_outliers),
                    'statistical_outliers': len(stat_outliers),
                    'physical_outlier_pct': len(physical_outliers) / len(valid_data) * 100,
                    'statistical_outlier_pct': len(stat_outliers) / len(valid_data) * 100
                }
                
        return outlier_stats
    
    def visualize_distributions(self) -> None:
        """Visualize sensor data distributions and activity distribution."""
        logger.info("Creating visualizations...")
        
        # Activity distribution
        plt.figure(figsize=(15, 6))
        activity_counts = pd.Series(self.activities).value_counts().sort_index()
        sns.barplot(x=activity_counts.index, y=activity_counts.values)
        plt.title('Activity Distribution')
        plt.xticks(range(len(ACTIVITY_NAMES)), 
                  [ACTIVITY_NAMES[i] for i in range(1, 15)], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Sensor distributions by type
        fig, axes = plt.subplots(len(SENSOR_RANGES), 1, 
                                figsize=(15, 5*len(SENSOR_RANGES)))
        
        for idx, (sensor_type, info) in enumerate(SENSOR_RANGES.items()):
            all_values = []
            for sensor_id in info['sensors']:
                valid_data = self.sensors[sensor_id][self.sensors[sensor_id] != -999999.99]
                all_values.extend(valid_data)
                
            sns.histplot(all_values, bins=100, ax=axes[idx])
            axes[idx].axvline(info['range'][0], color='r', linestyle='--')
            axes[idx].axvline(info['range'][1], color='r', linestyle='--')
            axes[idx].set_title(f'{sensor_type.capitalize()} Distribution')
            
        plt.tight_layout()
        plt.show()

class SensorDataPreprocessor:
    """Class for preprocessing sensor data with various imputation strategies."""
    
    def __init__(self, analyzer: SensorDataAnalyzer):
        self.analyzer = analyzer
        self.scalers = {}
        
    def preprocess(self, imputation_method: str = 'knn', 
                  remove_outliers: bool = True) -> Dict:
        """
        Preprocess sensor data with specified imputation method.
        
        Args:
            imputation_method: One of ['mean', 'mode', 'knn', 'interpolation']
            remove_outliers: Whether to remove statistical outliers
        """
        logger.info(f"Preprocessing data using {imputation_method} imputation...")
        
        processed_sensors = {}
        for sensor_id, data in tqdm(self.analyzer.sensors.items(), 
                                  desc="Processing sensors"):
            # Handle missing values
            if imputation_method == 'knn':
                processed_data = self._knn_impute(data)
            elif imputation_method == 'interpolation':
                processed_data = self._interpolate(data)
            else:
                processed_data = self._simple_impute(data, imputation_method)
                
            # Remove outliers if specified
            if remove_outliers:
                processed_data = self._remove_outliers(processed_data, sensor_id)
                
            # Standardize
            self.scalers[sensor_id] = StandardScaler()
            processed_sensors[sensor_id] = self.scalers[sensor_id].fit_transform(processed_data)
            
        return processed_sensors
    
    def _knn_impute(self, data: np.ndarray) -> np.ndarray:
        """KNN imputation for missing values."""
        imputer = KNNImputer(n_neighbors=5)
        return imputer.fit_transform(data)
    
    def _interpolate(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation for missing values."""
        processed = data.copy()
        for i in range(processed.shape[1]):
            mask = processed[:, i] != -999999.99
            if np.any(mask):
                f = interp1d(np.where(mask)[0], processed[mask, i], 
                           bounds_error=False, fill_value="extrapolate")
                processed[~mask, i] = f(np.where(~mask)[0])
        return processed
    
    def _simple_impute(self, data: np.ndarray, 
                      strategy: str) -> np.ndarray:
        """Simple imputation using mean or mode."""
        imputer = SimpleImputer(strategy=strategy)
        return imputer.fit_transform(data)
    
    def _remove_outliers(self, data: np.ndarray, 
                        sensor_id: int) -> np.ndarray:
        """Remove statistical outliers using IQR method."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        mask = (data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)
        return np.where(mask, data, np.nan)
    
    def save_preprocessed_data(self, processed_sensors: Dict, 
                             method: str) -> None:
        """Save preprocessed data and scalers."""
        logger.info("Saving preprocessed data...")
        output_dir = os.path.join(self.analyzer.root_path, 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(processed_sensors, 
                   os.path.join(output_dir, f'{method}_sensors.pkl'))
        joblib.dump(self.scalers, 
                   os.path.join(output_dir, f'{method}_scalers.pkl'))

def explore_data(analyzer):
    """Run full data exploration workflow"""
    missing_stats = analyzer.analyze_missing_values()
    outlier_stats = analyzer.analyze_outliers()
    analyzer.visualize_distributions()
    return missing_stats, outlier_stats

def process_data(analyzer, method='all'):
    """Run data preprocessing workflow"""
    preprocessor = SensorDataPreprocessor(analyzer)
    methods = ['mean', 'mode', 'knn', 'interpolation'] if method == 'all' else [method]
    
    for m in methods:
        processed_data = preprocessor.preprocess(m)
        preprocessor.save_preprocessed_data(processed_data, m)

def save_analysis_results(sensors, activities, output_file='output/analysis_results.txt'):
    """Save analysis results to a text file"""
    with open(output_file, 'w') as f:
        # Activity Distribution
        f.write("=== ACTIVITY DISTRIBUTION ===\n\n")
        activity_counts = pd.Series(activities).value_counts().sort_index()
        for act_id, count in activity_counts.items():
            percentage = (count/len(activities))*100
            f.write(f"{ACTIVITY_NAMES[act_id]}: {count} samples ({percentage:.2f}%)\n")
        
        # Missing Values Analysis
        f.write("\n=== MISSING VALUES ANALYSIS ===\n\n")
        for sensor_id, data in sensors.items():
            missing_mask = data == -999999.99
            missing_count = np.sum(missing_mask)
            missing_rate = (missing_count / data.size) * 100
            f.write(f"Sensor {sensor_id}:\n")
            f.write(f"  Total missing values: {missing_count:,}\n")
            f.write(f"  Missing rate: {missing_rate:.2f}%\n")
        
        # Outlier Analysis
        f.write("\n=== OUTLIER ANALYSIS ===\n\n")
        for sensor_type, info in SENSOR_RANGES.items():
            f.write(f"{sensor_type.upper()}:\n")
            for sensor_id in info['sensors']:
                data = analyzer.sensors[sensor_id]
                valid_data = data[data != -999999.99]  # Exclude missing values
                
                # Range-based outliers
                range_outliers = valid_data[(valid_data < info['range'][0]) | 
                                         (valid_data > info['range'][1])]
                range_rate = (len(range_outliers) / len(valid_data)) * 100
                
                # Statistical outliers (IQR method)
                Q1 = np.percentile(valid_data, 25)
                Q3 = np.percentile(valid_data, 75)
                IQR = Q3 - Q1
                stat_outliers = valid_data[(valid_data < Q1 - 1.5*IQR) | 
                                         (valid_data > Q3 + 1.5*IQR)]
                stat_rate = (len(stat_outliers) / len(valid_data)) * 100
                
                f.write(f"  Sensor {sensor_id}:\n")
                f.write(f"    Range outliers: {len(range_outliers):,} ({range_rate:.2f}%)\n")
                f.write(f"    Statistical outliers: {len(stat_outliers):,} ({stat_rate:.2f}%)\n")

def save_visualizations(activities):
    """Save all visualizations"""
    # Activity Distribution
    plt.figure(figsize=(15, 6))
    activity_counts = pd.Series(activities).value_counts().sort_index()
    plt.bar(range(1, 15), activity_counts)
    plt.title('Activity Distribution')
    plt.xticks(range(1, 15), [ACTIVITY_NAMES[i] for i in range(1, 15)], 
               rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/activity_distribution.png')
    plt.close()
    
    # Sensor distributions by type
    for sensor_type, info in SENSOR_RANGES.items():
        plt.figure(figsize=(15, 5))
        all_values = []
        for sensor_id in info['sensors']:
            data = analyzer.sensors[sensor_id]
            valid_data = data[data != -999999.99]
            all_values.extend(valid_data)
        
        sns.histplot(all_values, bins=100)
        plt.axvline(info['range'][0], color='r', linestyle='--', alpha=0.5)
        plt.axvline(info['range'][1], color='r', linestyle='--', alpha=0.5)
        plt.title(f'{sensor_type.capitalize()} Distribution')
        plt.xlabel(f'Value ({info["unit"]})')
        plt.tight_layout()
        plt.savefig(f'output/{sensor_type}_distribution.png')
        plt.close()

def save_enhanced_analysis(sensors, activities, output_dir='output'):
    """Save comprehensive analysis results including descriptive stats and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'analysis_results.txt'), 'w') as f:
        # Activity Distribution section remains the same
        f.write("=== ACTIVITY DISTRIBUTION ===\n\n")
        activity_counts = pd.Series(activities).value_counts().sort_index()
        for act_id, count in activity_counts.items():
            percentage = (count/len(activities))*100
            f.write(f"{ACTIVITY_NAMES[act_id]}: {count} samples ({percentage:.2f}%)\n")
        
        # Sensor Analysis by Type
        for sensor_type, info in SENSOR_RANGES.items():
            f.write(f"\n=== {sensor_type.upper()} SENSORS ===\n\n")
            
            for sensor_id in info['sensors']:
                data = sensors[sensor_id]  # Shape: (3500, 512)
                valid_mask = data != -999999.99
                valid_data = data[valid_mask]
                
                # Descriptive Statistics
                f.write(f"Sensor {sensor_id}:\n")
                f.write("  Descriptive Statistics:\n")
                f.write(f"    Mean: {np.mean(valid_data):.2f} {info['unit']}\n")
                f.write(f"    Std: {np.std(valid_data):.2f} {info['unit']}\n")
                f.write(f"    Min: {np.min(valid_data):.2f} {info['unit']}\n")
                f.write(f"    25%: {np.percentile(valid_data, 25):.2f} {info['unit']}\n")
                f.write(f"    Median: {np.median(valid_data):.2f} {info['unit']}\n")
                f.write(f"    75%: {np.percentile(valid_data, 75):.2f} {info['unit']}\n")
                f.write(f"    Max: {np.max(valid_data):.2f} {info['unit']}\n")
                
                # Missing Values
                missing_count = np.sum(~valid_mask)
                missing_rate = (missing_count / data.size) * 100
                f.write("  Missing Values:\n")
                f.write(f"    Count: {missing_count:,}\n")
                f.write(f"    Rate: {missing_rate:.2f}%\n")
                
                # Outliers (excluding missing values)
                range_outliers = valid_data[(valid_data < info['range'][0]) | 
                                         (valid_data > info['range'][1])]
                f.write("  Outliers:\n")
                f.write(f"    Count: {len(range_outliers):,}\n")
                f.write(f"    Rate: {(len(range_outliers)/len(valid_data))*100:.2f}%\n")
                
                # Activity-specific statistics
                f.write("  Activity-specific Statistics:\n")
                for activity in range(1, 15):
                    # Calculate mean for each time series
                    activity_mask = activities == activity
                    activity_data = data[activity_mask]
                    activity_means = np.mean(activity_data, axis=1)
                    valid_means = activity_means[activity_means != -999999.99]
                    
                    if len(valid_means) > 0:
                        f.write(f"    {ACTIVITY_NAMES[activity]}:\n")
                        f.write(f"      Mean: {np.mean(valid_means):.2f} {info['unit']}\n")
                        f.write(f"      Std: {np.std(valid_means):.2f} {info['unit']}\n")
                f.write("\n")

    # Sensor characteristics visualization
    selected_sensors = {
        'Heart Rate': 2,
        'Hand Acceleration': 4,
        'Chest Temperature': 13,
        'Foot Acceleration': 24,
        'Foot Magnetometer': 30
    }
    
    fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(15, 20))
    fig.suptitle('Sensor Data Across Different Activities', fontsize=16)
    
    for i, (sensor_name, sensor_id) in enumerate(selected_sensors.items()):
        sensor_data = sensors[sensor_id]
        
        box_data = []
        for activity in range(1, 15):
            activity_mask = activities == activity
            activity_data = sensor_data[activity_mask]
            # Calculate mean for each time series, excluding missing values
            series_means = np.ma.masked_equal(activity_data, -999999.99).mean(axis=1)
            box_data.append(series_means.compressed())
        
        axes[i].boxplot(box_data, labels=[ACTIVITY_NAMES[j] for j in range(1, 15)])
        axes[i].set_title(f'{sensor_name} (Sensor {sensor_id})')
        axes[i].set_xlabel('Activities')
        axes[i].set_ylabel('Average Sensor Value')
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_characteristics.png'))
    plt.close()

    # Distribution plots
    for sensor_type, info in SENSOR_RANGES.items():
        plt.figure(figsize=(15, 5))
        for sensor_id in info['sensors']:
            data = sensors[sensor_id]
            valid_data = data[data != -999999.99]
            sns.kdeplot(valid_data, label=f'Sensor {sensor_id}')
        
        plt.axvline(info['range'][0], color='r', linestyle='--', alpha=0.5)
        plt.axvline(info['range'][1], color='r', linestyle='--', alpha=0.5)
        plt.title(f'{sensor_type.capitalize()} Distribution')
        plt.xlabel(f'Value ({info["unit"]})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sensor_type}_distribution.png'))
        plt.close()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Sensor Data Analysis and Processing Tool')
    parser.add_argument('mode', choices=['explore', 'process'], 
                      help='Mode of operation: explore data or process data')
    parser.add_argument('--method', choices=['all', 'mean', 'mode', 'knn', 'interpolation'],
                      default='all', help='Preprocessing method (only for process mode)')
    parser.add_argument('--data_path', default='.', 
                      help='Path to data directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SensorDataAnalyzer(args.data_path)
    analyzer.load_data('learning')
    
    # Run selected workflow
    if args.mode == 'explore':
        missing_stats, outlier_stats = explore_data(analyzer)
        save_analysis_results(analyzer.sensors, analyzer.activities)
        save_visualizations(analyzer.activities)
        save_enhanced_analysis(analyzer.sensors, analyzer.activities)
    else:  # process mode
        process_data(analyzer, args.method)