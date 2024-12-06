"""
This script provides functionality for exploring and preprocessing sensor data from a dataset. The script can operate in two modes: 'explore' and 'process'.

- In 'explore' mode, the script performs a comprehensive analysis of the sensor data, including:
    - Loading sensor data, activities, and subject IDs.
    - Analyzing missing values and outliers.
    - Visualizing distributions of sensor data and activity distribution.
    - Saving analysis results and visualizations to the 'explorer' directory.

- In 'process' mode, the script preprocesses the sensor data using various imputation methods and outlier removal strategies:
    - Imputing missing values using one of the following methods: 'mean', 'mode', 'knn', or 'interpolation'.
    - Replacing outliers based on physical constraints and statistical methods with median value.
    - Standardizing the sensor data.
    - Saving the preprocessed data and scalers to the 'processed' directory.

Classes:
- SensorDataAnalyzer: Main class for analyzing and preprocessing sensor data.
- SensorDataPreprocessor: Class for preprocessing sensor data with various imputation strategies.

Functions:
- explore_data(analyzer, action='both'): Run full data exploration workflow.
- process_data(analyzer, method='all'): Run data preprocessing workflow.
- explorer_analysis(sensors, activities, output_dir='explorer'): Save comprehensive analysis results including descriptive stats and visualizations.

Usage:
- To explore data: python script.py explore --data_path <path_to_data>
- To process data: python script.py process --method <imputation_method> --data_path <path_to_data>
"""

import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.interpolate import interp1d, UnivariateSpline
from stat_outliers import StatisticalOutlierHandler
from tabulate import tabulate
from tqdm import tqdm
from typing import Dict
warnings.filterwarnings('ignore')

# Create explorer directory if it doesn't exist
os.makedirs('explorer', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SENSOR_RANGES = {
    'heart_rate': {'sensors': [2], 'range': (30, 220), 'unit': 'bpm'},
    'temperature': {'sensors': [3, 13, 23], 'range': (20, 40), 'unit': '°C'},
    'acceleration': {'sensors': [4, 5, 6, 14, 15, 16, 24, 25, 26], 'range': (-50, 50), 'unit': 'm/s^2'},
    'gyroscope': {'sensors': [7, 8, 9, 17, 18, 19, 27, 28, 29], 'range': (-10, 10), 'unit': 'rad/s'},
    'magnetometer': {'sensors': [10, 11, 12, 20, 21, 22, 30, 31, 32], 'range': (-100, 100), 'unit': 'µT'}
}

ACTIVITY_NAMES = {
    1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking very slow',
    5: 'Normal walking', 6: 'Nordic walking', 7: 'Running',
    8: 'Ascending stairs', 9: 'Descending stairs', 10: 'Cycling',
    11: 'Ironing', 12: 'Vacuum cleaning', 13: 'Rope jumping',
    14: 'Playing soccer'
}

MISSING_VALUE = -999999.99


# ANALYZER CLASSES
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
            self.sensors[i] = np.loadtxt(sensor_file)   # Shape: (3500, 512)

        # Load activities and subjects
        self.subjects = np.loadtxt(os.path.join(
            path, 'subject_Id.txt')).astype(int)
        if dataset_type == 'learning':
            self.activities = np.loadtxt(os.path.join(
                path, 'activity_Id.txt')).astype(int)

    def analyze_missing_values(self) -> Dict:
        """Analyze missing values in sensor data."""
        logger.info("Analyzing missing values...")
        missing_stats = {}

        for sensor_id, data in self.sensors.items():
            # number of complete samples (rows) that are entirely missing
            missing_samples = np.all(data == MISSING_VALUE, axis=1).sum()
            # number of missing points (values) in the entire dataset
            missing_points = np.sum(data == MISSING_VALUE)
            # number of samples with 20%+ missing values
            missing_pct = np.mean(data == MISSING_VALUE, axis=1)
            missing_samples_20pct = np.sum(missing_pct > 0.20)

            missing_stats[sensor_id] = {
                'total_samples': data.shape[0],
                'missing_time_series': missing_samples,
                'time_seriess_with_20%+_missing': missing_samples_20pct,
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
                valid_data = data[data != MISSING_VALUE]

                # Physical constraints
                physical_outliers = valid_data[
                    (valid_data < info['range'][0]) |
                    (valid_data > info['range'][1])
                ]

                # Statistical outliers (Interquartile Range - IQR method)
                q1, q3 = np.percentile(valid_data, [25, 75])
                iqr = q3 - q1
                TUKEY_CONSTANT = 1.5
                stat_outliers = valid_data[  # Tukey's fences method
                    (valid_data < q1 - TUKEY_CONSTANT * iqr) |   # lower bound
                    (valid_data > q3 + TUKEY_CONSTANT * iqr)     # upper bound
                ]

                outlier_stats[sensor_id] = {
                    'sensor_type': sensor_type,
                    'physical_outliers': len(physical_outliers),
                    'statistical_outliers': len(stat_outliers),
                    'physical_outlier_pct': len(physical_outliers) / len(valid_data) * 100,
                    'statistical_outlier_pct': len(stat_outliers) / len(valid_data) * 100
                }

        return outlier_stats

    def visualize_distributions(self, action: str = 'visualize') -> None:
        """Visualize or save sensor data distributions and activity distribution."""
        logger.info("Creating visualizations...")

        # Activity distribution (bar plot)
        if self.activities is not None:
            plt.figure(figsize=(15, 6))
            # Count the number of samples for each activity and sort by activity ID
            activity_counts = pd.Series(
                self.activities).value_counts().sort_index()
            sns.barplot(x=activity_counts.index, y=activity_counts.values)
            plt.title('Activity Distribution')
            plt.xlabel('Activity')
            plt.ylabel('Count')
            plt.xticks(range(len(ACTIVITY_NAMES)),
                       [ACTIVITY_NAMES[i] for i in range(1, 15)],
                       rotation=45, ha='right')
            plt.tight_layout()
            if action in ['save', 'both']:
                plt.savefig('explorer/activity_distribution.png')
            if action in ['visualize', 'both']:
                plt.show()
            plt.close()

        # Sensor distributions by type (KDE plot)
        for sensor_type, info in SENSOR_RANGES.items():
            plt.figure(figsize=(15, 5))
            for sensor_id in info['sensors']:
                data = self.sensors[sensor_id]
                valid_data = data[data != MISSING_VALUE].flatten()
                sns.kdeplot(valid_data, label=f'Sensor {sensor_id}')

            plt.axvline(info['range'][0], color='r', linestyle='--', alpha=0.5)
            plt.axvline(info['range'][1], color='r', linestyle='--', alpha=0.5)
            plt.title(f'{sensor_type.capitalize()} Distribution')
            plt.xlabel(f'Value ({info["unit"]})')
            plt.legend()
            plt.tight_layout()

            # Set x-axis limits for specific sensor types
            if sensor_type in ['acceleration', 'gyroscope', 'magnetometer']:
                plt.xlim(info['range'][0] - 1, info['range'][1] + 1)

            if action in ['save', 'both']:
                plt.savefig(f'explorer/{sensor_type}_distribution.png')
            if action in ['visualize', 'both']:
                plt.show()
            plt.close()


class SensorDataPreprocessor:
    """Class for preprocessing sensor data with various imputation strategies."""

    def __init__(self, analyzer: SensorDataAnalyzer):
        self.analyzer = analyzer
        self.scalers = {}

    def preprocess(self, imputation_method: str = 'knn', remove_outliers: bool = True) -> Dict:
        """
        Preprocess sensor data with specified imputation method.

        Args:
            imputation_method: One of ['knn', 'interpolation', 'spline']
            remove_outliers: Whether to remove statistical outliers

        Returns:
            Processed sensor data as a dictionary of sensor ID to processed data.
        """
        processed_sensors = {}
        outlier_handler = StatisticalOutlierHandler()
        for sensor_id, data in tqdm(self.analyzer.sensors.items(), desc="Processing sensors"):
            # Remove rows with excessive missing values
            PERCENTAGE_THRESHOLD = 0.20
            missing_proportion = np.mean(data == MISSING_VALUE, axis=1)
            data = data[missing_proportion <= PERCENTAGE_THRESHOLD]

            # Handle missing values
            if imputation_method == 'knn':
                processed_data = self._knn_impute(data)
            elif imputation_method == 'interpolation':
                processed_data = self._interpolate(data)
            elif imputation_method == 'spline':
                processed_data = self._spline_impute(data)
            else:
                raise ValueError(
                    f"Invalid imputation method: {imputation_method}")

            # Remove outliers if specified
            if remove_outliers:
                processed_data = self._remove_outliers(
                    outlier_handler, processed_data, sensor_id, imputation_method)

            # Normalize
            normalizer = MinMaxScaler()
            self.scalers[sensor_id] = normalizer
            # processed_sensors[sensor_id] = normalizer.fit_transform(processed_data)
            processed_sensors[sensor_id] = processed_data

        return processed_sensors

    def _knn_impute(self, data: np.ndarray) -> np.ndarray:
        """KNN imputation for missing values."""
        imputer = KNNImputer(n_neighbors=5)
        return imputer.fit_transform(data)

    def _interpolate(self, data: np.ndarray) -> np.ndarray:
        """Linear interpolation for missing values."""
        processed = data.copy()
        for i in range(processed.shape[1]):
            mask = processed[:, i] != MISSING_VALUE
            if np.any(mask):
                f = interp1d(np.where(mask)[
                             0], processed[mask, i], bounds_error=False, fill_value="extrapolate")
                processed[~mask, i] = f(np.where(~mask)[0])
        return processed

    def _spline_impute(self, data: np.ndarray) -> np.ndarray:
        """Spline interpolation for missing values."""
        processed = data.copy()
        for i in range(processed.shape[1]):
            mask = processed[:, i] != MISSING_VALUE
            if np.any(mask):
                spline = UnivariateSpline(
                    np.where(mask)[0], processed[mask, i], s=0)
                processed[~mask, i] = spline(np.where(~mask)[0])
        return processed

    def _remove_outliers(self, handler: StatisticalOutlierHandler, data: np.ndarray, sensor_id: int, imputation_method: str) -> np.ndarray:
        """Remove outliers using physical constraints and statistical methods, then impute."""
        sensor_type = next(
            key for key, value in SENSOR_RANGES.items() if sensor_id in value['sensors'])
        valid_range = SENSOR_RANGES[sensor_type]['range']

        # Physically impossible values replaced with NaN
        def _remove_physical_outliers(row, valid_range):    # preserves data shape
            lower_bound, upper_bound = valid_range
            return np.where((row >= lower_bound) & (row <= upper_bound), row, np.nan)

        data = np.apply_along_axis(
            _remove_physical_outliers, 1, data, valid_range)

        # Statistical outliers are imputed
        data = handler.handle_outliers(data, sensor_id)

        return data

    # def _winsorize_outliers(self, data: np.ndarray, sensor_id: int) -> np.ndarray:
    #     """Winsorization: Cap outliers at percentiles"""
    #     processed_data = data.copy()
    #     for i in range(data.shape[1]):
    #         column = data[:, i]
    #         lower, upper = np.percentile(column, [2.5, 97.5])
    #         processed_data[:, i] = np.clip(column, lower, upper)
    #     return processed_data

    def save_preprocessed_data(self, processed_sensors: Dict, method: str) -> None:
        """Save preprocessed data and scalers."""
        output_dir = os.path.join(self.analyzer.root_path, 'processed')
        os.makedirs(output_dir, exist_ok=True)

        # Serialize processed sensor data and scalers to files
        for sensor_id, sensor_data in processed_sensors.items():
            directory = os.path.join(output_dir, method)
            os.makedirs(directory, exist_ok=True)

            filename = os.path.join(directory, f'sensor_{sensor_id}.pkl')
            joblib.dump(sensor_data, filename)
            joblib.dump(self.scalers[sensor_id],
                        filename.replace('.pkl', '_scaler.pkl'))

        # Generate and save summary statistics
        summary_file = os.path.join(output_dir, f'{method}_summary.txt')
        summary_data = []

        for sensor_id, data in processed_sensors.items():
            num_time_series = data.shape[0]
            num_values = np.prod(data.shape)

            row_means = np.nanmean(data, axis=1)
            mean_of_means = np.nanmean(row_means)

            row_stds = np.nanstd(data, axis=1)
            mean_of_stds = np.nanmean(row_stds)

            summary_data.append(
                [sensor_id, num_time_series, num_values, mean_of_means, mean_of_stds])

        headers = ["Sensor ID", "Number of time series",
                   "Number of values", "Mean", "Std"]
        table = tabulate(summary_data, headers,
                         tablefmt="fancy_grid", floatfmt=".2f")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(table)
            f.close()


# EXPLORER ANALYSIS
def explore_data(analyzer, action='both'):
    """
    Run full data exploration workflow.

    This function performs a comprehensive data exploration by analyzing missing values,
    detecting outliers, and visualizing data distributions.

    Parameters:
        analyzer (object): An object that provides methods to analyze missing values, outliers,
                        and visualize data distributions.
        action (str): Specifies what to do with distributions plotting. Default is 'both'.
                    Possible values are:
                    - 'visualize': (Only) Visualize the distributions.
                    - 'save': (Only) Save the distributions to files.
                    - 'both': Both visualize and save the distributions.

    Returns:
        tuple: A tuple containing:
            - missing_stats (dict): Statistics about missing values.
            - outlier_stats (dict): Statistics about outliers.
    """
    missing_stats = analyzer.analyze_missing_values()
    outlier_stats = analyzer.analyze_outliers()
    analyzer.visualize_distributions(action)
    return missing_stats, outlier_stats


def explorer_analysis(analyzer, missing_stats=None, outlier_stats=None, output_dir='explorer'):
    """
    Save comprehensive analysis results including descriptive stats and visualizations.
    Parameters:
        analyzer (object): An object containing sensor and activity data.
        output_dir (str, optional): Directory where analysis results will be saved. Defaults to 'explorer'.
        missing_stats (dict, optional): Precomputed missing value statistics. Defaults to None.
        outlier_stats (dict, optional): Precomputed outlier statistics. Defaults to None.
    Returns:
        None
    This function performs a full analysis of the provided data, including:
    - Saving activity distribution statistics.
    - Analyzing and saving missing values statistics for each sensor.
    - Writing detailed sensor analysis by type.
    - Generating and saving visualizations of sensor data across different activities.
    """
    os.makedirs(output_dir, exist_ok=True)

    sensors = analyzer.sensors
    activities = analyzer.activities
    if missing_stats is None or outlier_stats is None:
        missing_stats, outlier_stats = explore_data(analyzer, 'save')

    with open(os.path.join(output_dir, 'analysis_results.txt'), 'w', encoding='utf-8') as f:
        # Activity Distribution section remains the same
        f.write("=== ACTIVITY DISTRIBUTION ===\n\n")
        activity_counts = pd.Series(activities).value_counts().sort_index()
        for act_id, count in activity_counts.items():
            percentage = (count/len(activities))*100
            f.write(
                f"{ACTIVITY_NAMES[act_id]}: {count} samples ({percentage:.2f}%)\n")

        # Overall Missing Values Analysis
        f.write("\n=== MISSING VALUES ANALYSIS ===\n\n")
        for sensor_id, data in sensors.items():
            missing_mask = data == MISSING_VALUE
            missing_count = np.sum(missing_mask)
            missing_rate = (missing_count / data.size) * 100
            f.write(f"Sensor {sensor_id}:\n")
            f.write(f"  Total missing values: {missing_count:,}\n")
            f.write(f"  Missing rate: {missing_rate:.2f}%\n\n")
        f.close()

        # Sensor Analysis by Type
        _write_sensor_analysis(output_dir, sensors,
                               activities, missing_stats, outlier_stats)

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
            series_means = np.ma.masked_equal(
                activity_data, MISSING_VALUE).mean(axis=1)
            box_data.append(series_means.compressed())

        axes[i].boxplot(box_data, labels=[ACTIVITY_NAMES[j]
                        for j in range(1, 15)])
        axes[i].set_title(f'{sensor_name} (Sensor {sensor_id})')
        axes[i].set_xlabel('Activities')
        axes[i].set_ylabel('Average Sensor Value')
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_characteristics.png'))
    plt.close()


def _write_sensor_analysis(output_dir, sensors, activities, missing_stats, outlier_stats):
    """
    (Private) Writes a detailed analysis of sensor data to a text file.
    Parameters:
        output_dir (str): The directory where the analysis results will be saved.
        sensors (dict): A dictionary containing sensor data, where keys are sensor IDs and values are numpy arrays of sensor readings.
        activities (numpy array): An array indicating the activity type for each data point.
        missing_stats (dict): A dictionary containing statistics about missing values for each sensor.
        outlier_stats (dict): A dictionary containing statistics about outliers for each sensor.
    The function performs the following analyses for each sensor:
    - Descriptive statistics (mean, standard deviation, min, 25th percentile, median, 75th percentile, max).
    - Missing values (count and rate).
    - Outliers (physical and statistical, count and rate).
    - Activity-specific statistics (mean, standard deviation, validity rate, and sample count for each activity).
    The results are written to a file named 'analysis_results.txt' in the specified output directory.
    """
    with open(os.path.join(output_dir, 'analysis_results.txt'), 'a', encoding='utf-8') as f:
        for sensor_type, info in SENSOR_RANGES.items():
            f.write(f"\n=== {sensor_type.upper()} SENSORS ===\n\n")

            for sensor_id in info['sensors']:
                data = sensors[sensor_id]  # Shape: (3500, 512)
                valid_mask = data != MISSING_VALUE
                valid_data = data[valid_mask]

                # Descriptive Statistics
                descriptive_stats = [
                    ["Mean", f"{np.mean(valid_data):.2f} {info['unit']}"],
                    ["Std", f"{np.std(valid_data):.2f} {info['unit']}"],
                    ["Min", f"{np.min(valid_data):.2f} {info['unit']}"],
                    ["25%",
                        f"{np.percentile(valid_data, 25):.2f} {info['unit']}"],
                    ["Median", f"{np.median(valid_data):.2f} {info['unit']}"],
                    ["75%",
                        f"{np.percentile(valid_data, 75):.2f} {info['unit']}"],
                    ["Max", f"{np.max(valid_data):.2f} {info['unit']}"]
                ]
                f.write(f"Sensor {sensor_id}:\n")
                f.write("  Descriptive Statistics:\n")
                f.write(tabulate(descriptive_stats, headers=[
                        "Statistic", "Value"], tablefmt="fancy_grid"))
                f.write("\n")

                # Missing Values
                current_missing = missing_stats[sensor_id]
                f.write(
                    f"  Missing Values: {current_missing['missing_points']:,} ({current_missing['missing_percentage']:.2f}%)\n")
                f.write("\n")

                # Outliers (excluding missing values)
                current_outliers = outlier_stats[sensor_id]
                f.write("  Outliers:\n")
                f.write(
                    f"    Physical: {current_outliers['physical_outliers']:,} ({current_outliers['physical_outlier_pct']:.2f}%)\n")
                f.write(
                    f"    Statistical: {current_outliers['statistical_outliers']:,} ({current_outliers['statistical_outlier_pct']:.2f}%)\n")
                f.write("\n\n")

                # Activity-specific statistics
                f.write("  Activity-specific Statistics:\n")
                activity_stats = []
                for activity in range(1, 15):
                    activity_mask = activities == activity
                    activity_data = data[activity_mask]
                    valid_mask = activity_data != MISSING_VALUE

                    if np.any(valid_mask):
                        valid_data = activity_data[valid_mask]
                        validity_rate = (np.sum(valid_mask) /
                                         valid_mask.size) * 100
                        activity_stats.append([
                            ACTIVITY_NAMES[activity],
                            f"{np.mean(valid_data):.2f} {info['unit']}",
                            f"{np.std(valid_data):.2f} {info['unit']}",
                            f"{validity_rate:.1f}%",
                            f"{np.sum(valid_mask):,} / {valid_mask.size:,}"
                        ])
                f.write(tabulate(activity_stats, headers=[
                        "Activity", "Mean", "Std", "Valid data", "Samples"], tablefmt="fancy_grid"))
                f.write("\n\n")
        f.close()


# PROCESSOR ANALYSIS
def process_data(analyzer, method='all'):
    """Run data preprocessing workflow"""
    preprocessor = SensorDataPreprocessor(analyzer)
    methods = ['knn', 'interpolation',
               'spline'] if method == 'all' else [method]

    for m in methods:
        logger.info(f"Processing data using {m} imputation...")
        processed_data = preprocessor.preprocess(m)
        logger.info(f"Saving preprocessed data for {m} imputation...")
        preprocessor.save_preprocessed_data(processed_data, m)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Sensor Data Analysis and Processing Tool')
    parser.add_argument('mode', choices=[
                        'explore', 'process'], help='Mode of operation: explore data or process data', default='explore')
    parser.add_argument('--method', '--m', choices=['all', 'knn', 'interpolation',
                        'spline'], default='all', help='Preprocessing method (only for process mode)')
    parser.add_argument('--visualization', '--v', choices=['visualize', 'save', 'both'],
                        default='both', help='Action for visualization plots (only for explore mode)')
    parser.add_argument('--dataset_type', '--dt', choices=[
                        'learning', 'test'], default='learning', help='Type of dataset to load (only for explore mode)')
    parser.add_argument('--data_path', default='.',
                        help='Path to data directory (default: current directory)')
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SensorDataAnalyzer(args.data_path)
    analyzer.load_data(args.dataset_type)

    # Run selected workflow
    if args.mode == 'explore':
        missing_stats, outlier_stats = explore_data(
            analyzer, args.visualization)   # check explore_data doc for args
        explorer_analysis(analyzer, missing_stats=missing_stats,
                          outlier_stats=outlier_stats)
    else:  # process mode
        process_data(analyzer, args.method)
