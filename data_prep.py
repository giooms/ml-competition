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
    'acceleration': {'sensors': [4, 5, 6, 14, 15, 16, 24, 25, 26], 'range': (-25, 25), 'unit': 'm/s^2'},
    'gyroscope': {'sensors': [7, 8, 9, 17, 18, 19, 27, 28, 29], 'range': (-2.5, 2.5), 'unit': 'rad/s'},
    'magnetometer': {'sensors': [10, 11, 12, 20, 21, 22, 30, 31, 32], 'range': (-150, 150), 'unit': 'µT'}
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
        self.sensors: Dict[int, pd.DataFrame] = {}
        self.activities = pd.Series(dtype=int)
        self.subjects = pd.Series(dtype=int)

    def load_data(self, dataset_type: str = 'learning') -> None:
        """Load sensor data, activities, and subject IDs."""
        logger.info(f"Loading {dataset_type} dataset...")
        path = self.learning_path if dataset_type == 'learning' else self.test_path

        # Load sensors
        self.sensors = {}
        for i in tqdm(range(2, 33), desc="Loading sensors"):
            file_prefix = 'LS' if dataset_type == 'learning' else 'TS'
            sensor_file = os.path.join(path, f'{file_prefix}_sensor_{i}.txt')
            data = pd.read_csv(sensor_file, delimiter=' ', header=None)
            data.replace(MISSING_VALUE, np.nan, inplace=True)
            self.sensors[i] = data

        # Load activities and subjects
        self.subjects = pd.read_csv(os.path.join(
            path, 'subject_Id.txt'), header=None).squeeze()
        if dataset_type == 'learning':
            self.activities = pd.read_csv(os.path.join(
                path, 'activity_Id.txt'), delimiter=' ', header=None).squeeze()

        # Debug
        # print(f"Loaded {len(self.sensors)} sensors")
        # print(f"Loaded {len(self.activities)} activities")
        # print(f"Loaded {len(self.subjects)} subjects")

    def analyze_missing_values(self) -> Dict:
        """Analyze missing values in sensor data."""
        logger.info("Analyzing missing values...")
        missing_stats = {}

        for sensor_id, data in self.sensors.items():
            assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
            results = self._analyze_df_missing_patterns(data)

            #  total_missing  missing_percentage  longest_sequence  num_sequences  avg_sequence_length
            missing_stats[sensor_id] = {
                'total_samples': results.shape[0],
                'missing_time_series': (results['missing_percentage'] == 100.0).sum(),
                'missing_points': results['total_missing'].sum(),
                'missing_percentage': results['missing_percentage'].mean(),
                'longest_sequence': results['longest_sequence'].max(),
                'avg_sequence_length': results['avg_sequence_length'].mean()
            }

        return missing_stats

    def _analyze_df_missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for idx, row in df.iterrows():
            total_missing, max_seq, num_seq = self._analyze_missing_pattern(
                row)
            results.append({
                'total_missing': total_missing,
                'missing_percentage': (total_missing / len(row)) * 100,
                'longest_sequence': max_seq,
                'num_sequences': num_seq,
                'avg_sequence_length': total_missing / num_seq if num_seq > 0 else 0
            })
        return pd.DataFrame(results, index=df.index)

    def _analyze_missing_pattern(self, df_row: pd.Series) -> tuple:
        """
        Analyzes the pattern of missing values in a row.
        Returns:
            - Number of missing values
            - Length of longest continuous sequence of missing values
            - Number of separate missing sequences
        """
        missing_mask = df_row.isna()
        total_missing = missing_mask.sum()

        # If no missing values, return early
        if total_missing == 0:
            return 0, 0, 0

        # Convert to numeric (True -> 1, False -> 0)
        numeric_mask = missing_mask.astype(int)

        # Find sequences of missing values
        # When diff() == 1, it's the start of a sequence
        # When diff() == -1, it's the end of a sequence
        diff_mask = numeric_mask.diff()
        sequence_starts = diff_mask[diff_mask == 1].index
        sequence_ends = diff_mask[diff_mask == -1].index

        # Handle edge cases (missing values at start/end)
        if numeric_mask.iloc[0] == 1:
            sequence_starts = pd.Index(
                [numeric_mask.index[0]]).append(sequence_starts)
        if numeric_mask.iloc[-1] == 1:
            sequence_ends = sequence_ends.append(
                pd.Index([numeric_mask.index[-1]]))

        # Calculate lengths of sequences
        sequence_lengths = [end - start + 1 for start,
                            end in zip(sequence_starts, sequence_ends)]
        max_sequence = max(sequence_lengths) if sequence_lengths else 0
        num_sequences = len(sequence_lengths)

        return total_missing, max_sequence, num_sequences

    def analyze_outliers(self) -> Dict:
        """Analyze outliers based on physical constraints and statistical methods."""
        logger.info("Analyzing outliers...")
        outlier_stats = {}

        for sensor_id in range(2, 33):
            sensor_type = get_sensor_type(sensor_id)
            lower_bound, upper_bound = get_sensor_range(sensor_id)

            data = self.sensors[sensor_id]

            # Physical constraints
            physical_outliers_count = (
                (data < lower_bound) | (data > upper_bound)).sum().sum()

            # Statistical outliers (Interquartile Range - IQR method)
            quantiles = data.quantile([0.25, 0.75])
            q1 = quantiles.loc[0.25]
            q3 = quantiles.loc[0.75]
            iqr = q3 - q1
            TUKEY_CONSTANT = 1.5
            stat_outliers_count = (
                (data < q1 - TUKEY_CONSTANT * iqr) |
                (data > q3 + TUKEY_CONSTANT * iqr)).sum().sum()

            outlier_stats[sensor_id] = {
                'sensor_type': sensor_type,
                'physical_outliers': physical_outliers_count,
                'statistical_outliers': stat_outliers_count,
                'physical_outlier_pct': physical_outliers_count / data.size * 100,
                'statistical_outlier_pct': stat_outliers_count / data.size * 100
            }

        return outlier_stats

    def visualize_distributions(self, action: str = 'visualize') -> None:
        """Visualize or save sensor data distributions and activity distribution."""
        logger.info("Creating visualizations...")

        # Activity distribution (bar plot)
        if self.activities is not None:
            plt.figure(figsize=(15, 6))
            # Count the number of samples for each activity and sort by activity ID
            activity_counts = self.activities.value_counts().sort_index()
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
                valid_data = data.dropna().values.flatten()
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
        self.scalers = {}   # Store (min, max) values for each sensor

    def preprocess(self, imputation_method: str = 'spline', remove_outliers: bool = True) -> tuple:
        """Preprocess sensor data with specified imputation method."""
        processed_sensors = {}
        PERCENTAGE_THRESHOLD = 0.25
        series_to_drop = pd.Series([False] * 3500)
        outlier_handler = StatisticalOutlierHandler()

        # Identify series to drop
        for sensor_id, data in tqdm(self.analyzer.sensors.items(), desc="Identifying time series with excessive NaNs"):
            missing_patterns = self.analyzer._analyze_df_missing_patterns(data)

            # Drop time series with >25% missing
            series_to_drop |= (missing_patterns['missing_percentage'] > PERCENTAGE_THRESHOLD)

            # Drop time series with too many missing sequences
            series_to_drop |= (missing_patterns['num_sequences'] > 50)

        # Now process each sensor with chosen imputation and normalization
        for sensor_id, data in tqdm(self.analyzer.sensors.items(), desc="Cleaning sensors"):
            assert len(series_to_drop) == len(data), "Length mismatch."
            processed_data = data[~series_to_drop].reset_index(drop=True)

            # Impute missing values if any
            still_nans = processed_data.isna().values.any()
            if still_nans:
                if imputation_method == 'knn':
                    processed_data = self._knn_impute(processed_data)
                elif imputation_method == 'interpolation':
                    processed_data = self._interpolate(processed_data)
                elif imputation_method == 'spline':
                    processed_data = self._spline_impute(processed_data)
                else:
                    raise ValueError(f"Invalid imputation method: {imputation_method}")

                if processed_data.isna().values.any():
                    # If still missing values after imputation, log a warning
                    logger.warning(f"Sensor {sensor_id} still has missing values after imputation")

            # Remove outliers if needed
            if remove_outliers:
                processed_data = self._remove_outliers(outlier_handler, processed_data, sensor_id)

            # Compute median and IQR across columns (time steps)
            sensor_median = processed_data.median(axis=0)
            q1_vals = processed_data.quantile(0.25, axis=0)
            q3_vals = processed_data.quantile(0.75, axis=0)
            iqr_vals = q3_vals - q1_vals

            # Avoid division by zero
            iqr_vals[iqr_vals == 0] = 1e-9

            # Apply robust scaling
            normalized_data = (processed_data - sensor_median) / iqr_vals

            # Store scalers for test set normalization
            self.scalers[sensor_id] = (sensor_median, iqr_vals)

            processed_sensors[sensor_id] = normalized_data

        # Only after processing all sensors do we select the final activities and subjects
        activities = self.analyzer.activities[~series_to_drop].reset_index(drop=True).squeeze()
        subjects = self.analyzer.subjects[~series_to_drop].reset_index(drop=True).squeeze()

        return processed_sensors, activities, subjects

    def _knn_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """KNN imputation for missing values treating rows independently."""
        imputer = KNNImputer(n_neighbors=5)
        # Transpose the data to treat rows independently (imputation along columns)
        transposed_data = data.T
        imputed_data = imputer.fit_transform(transposed_data)
        # Transpose back to the original shape and convert to DataFrame
        imputed_df = pd.DataFrame(imputed_data.T)
        return imputed_df

    def _interpolate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Linear interpolation for missing values in each row independently."""
        processed = data.copy()
        for i in range(processed.shape[0]):
            row = processed.iloc[i]
            mask = row.notna()
            if mask.any():
                f = interp1d(row.index[mask], row[mask],
                             bounds_error=False, fill_value="extrapolate")
                processed.iloc[i, ~mask] = f(row.index[~mask])
        return processed

    def _spline_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Spline interpolation for missing values for each unique row independently."""
        processed = data.copy()
        for i in range(processed.shape[0]):
            row = processed.iloc[i, :]
            mask_notna = row.notna()
            if mask_notna.any():
                spline = UnivariateSpline(
                    np.where(mask_notna)[0], row[mask_notna], s=0)
                row[~mask_notna] = spline(np.where(~mask_notna)[0])
            processed.iloc[i, :] = row
        return processed

    def _remove_outliers(self, handler: StatisticalOutlierHandler, data: pd.DataFrame, sensor_id: int) -> pd.DataFrame:
        """Remove outliers using physical constraints and statistical methods, then impute."""
        valid_range = get_sensor_range(sensor_id)

        # Physically impossible values replaced with the value at time t-1
        data = data.apply(lambda row: self._remove_physical_outliers(
            row, valid_range), axis=1)

        # Statistical outliers are imputed
        data = handler.handle_outliers(data, sensor_id)

        return data

    def _remove_physical_outliers(self, row: pd.Series, valid_range: tuple) -> pd.Series:
        """Remove physically impossible values from a row based on range."""
        assert isinstance(row, pd.Series), "Row must be a pandas Series"
        assert len(valid_range) == 2, "Valid range must be a tuple of two values"

        lower_bound, upper_bound = valid_range
        previous_value = lower_bound
        for i in range(len(row)):
            if lower_bound <= row.iloc[i] <= upper_bound:
                previous_value = row.iloc[i]
            else:
                row.iloc[i] = previous_value
        return row

    def save_preprocessed_data(self, processed_sensors: Dict, processed_activities: pd.Series, processed_subjects: pd.Series, method: str) -> None:
        """Save preprocessed data and scalers."""
        output_dir = os.path.join(self.analyzer.root_path, 'processed')
        os.makedirs(output_dir, exist_ok=True)

        method_dir = os.path.join(output_dir, method)
        scaler_dir = os.path.join(method_dir, 'scalers')
        os.makedirs(method_dir, exist_ok=True)
        os.makedirs(scaler_dir, exist_ok=True)

        # Serialize processed sensor data and scalers to files
        for sensor_id, sensor_data in processed_sensors.items():
            sensor_path = os.path.join(method_dir, f'sensor_{sensor_id}.pkl')
            scaler_path = os.path.join(scaler_dir, f'scaler_{sensor_id}.pkl')

            sensor_data.to_pickle(sensor_path)
            joblib.dump(self.scalers[sensor_id], scaler_path)

        # Serialize activities and subjects to files
        activity_path = os.path.join(output_dir, f'activities.pkl')
        subjects_path = os.path.join(output_dir, f'subjects.pkl')

        # Debug: Print shapes before serialization
        # print(f"Activities shape before serialization: {processed_activities.size}")
        # print(f"Subjects shape before serialization: {processed_subjects.size}")

        processed_activities.to_pickle(activity_path)
        processed_subjects.to_pickle(subjects_path)

        # Generate and save summary statistics
        summary_file = os.path.join(output_dir, f'{method}_summary.txt')
        summary_data = []

        for sensor_id, sensor_data in processed_sensors.items():
            num_time_series = sensor_data.shape[0]
            num_values = sensor_data.size

            row_means = sensor_data.mean(axis=1, skipna=True)
            mean_of_means = row_means.mean()

            row_stds = sensor_data.std(axis=1, skipna=True)
            mean_of_stds = row_stds.mean()

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
def explore_data(analyzer: SensorDataAnalyzer, action: str = 'both') -> tuple:
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
    if action == 'none':
        return missing_stats, outlier_stats

    analyzer.visualize_distributions(action)
    return missing_stats, outlier_stats


def explorer_analysis(analyzer: SensorDataAnalyzer, missing_stats: dict = None, outlier_stats: dict = None, output_dir: str = 'explorer') -> None:
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
        missing_stats, outlier_stats = explore_data(analyzer, 'none')

    with open(os.path.join(output_dir, 'analysis_results.txt'), 'w', encoding='utf-8') as f:
        # Activity Distribution section remains the same
        f.write("=== ACTIVITY DISTRIBUTION ===\n\n")
        activity_counts = activities.value_counts().sort_index()
        for act_id, count in activity_counts.items():
            percentage = (count/len(activities))*100
            f.write(
                f"{ACTIVITY_NAMES[act_id]}: {count} samples ({percentage:.2f}%)\n")

        # Overall Missing Values Analysis
        f.write("\n=== MISSING VALUES ANALYSIS ===\n\n")
        for sensor_id in sensors.keys():
            current_missing = missing_stats[sensor_id]
            f.write(f"Sensor {sensor_id}:\n")
            f.write(
                f"  Total missing values: {current_missing['missing_points']:,}\n")
            f.write(
                f"  Missing rate: {current_missing['missing_percentage']:.2f}%\n")
            f.write(
                f"  Empty time series: {current_missing['missing_time_series']}\n\n")
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
            activity_data = sensor_data.loc[activity_mask]
            # Calculate mean for each time series, excluding missing values
            series_means = activity_data.mean(axis=1)
            box_data.append(series_means.dropna().values)

        axes[i].boxplot(box_data, labels=[ACTIVITY_NAMES[j]
                        for j in range(1, 15)])
        axes[i].set_title(f'{sensor_name} (Sensor {sensor_id})')
        axes[i].set_xlabel('Activities')
        axes[i].set_ylabel('Average Sensor Value')
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_characteristics.png'))
    plt.close()


def _write_sensor_analysis(output_dir: str, sensors: dict, activities: pd.Series, missing_stats: dict, outlier_stats: dict) -> None:
    """(Private) Writes a detailed analysis of sensor data to a text file."""
    with open(os.path.join(output_dir, 'analysis_results.txt'), 'a', encoding='utf-8') as f:
        for sensor_type, info in SENSOR_RANGES.items():
            f.write(f"\n=== {sensor_type.upper()} SENSORS ===\n\n")

            for sensor_id in info['sensors']:
                data = sensors[sensor_id]  # DataFrame
                data_as_series = data.stack()

                # Descriptive Statistics
                descriptive_stats = [
                    ["Mean", f"{data_as_series.mean():.2f} {info['unit']}"],
                    ["Std", f"{data_as_series.std():.2f} {info['unit']}"],
                    ["Min", f"{data_as_series.min():.2f} {info['unit']}"],
                    ["25%",
                        f"{data_as_series.quantile(0.25):.2f} {info['unit']}"],
                    ["Median", f"{data_as_series.median():.2f} {info['unit']}"],
                    ["75%",
                        f"{data_as_series.quantile(0.75):.2f} {info['unit']}"],
                    ["Max", f"{data_as_series.max():.2f} {info['unit']}"]
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
                f.write(
                    f"  Missing Time Series: {current_missing['missing_time_series']:,} / {current_missing['total_samples']:.2f}\n")
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
                    activity_data = data.loc[activity_mask]
                    data_as_series = activity_data.stack()

                    if not activity_data.empty:
                        validity_rate = (
                            1 - data_as_series.isna().mean())

                        activity_stats.append([
                            ACTIVITY_NAMES[activity],
                            f"{data_as_series.mean():.2f} {info['unit']}",
                            f"{data_as_series.std():.2f} {info['unit']}",
                            f"{(validity_rate*100):.1f}%",
                            f"{(validity_rate * data_as_series.size):,.0f} / {data_as_series.size:,}"
                        ])
                f.write(tabulate(activity_stats, headers=[
                        "Activity", "Mean", "Std", "Non-NaN data", "Samples"], tablefmt="fancy_grid"))
                f.write("\n\n")
        f.close()


# PROCESSOR ANALYSIS
def process_data(analyzer: SensorDataAnalyzer, method: str = 'spline') -> None:
    """Run data preprocessing workflow"""
    preprocessor = SensorDataPreprocessor(analyzer)
    methods = ['knn', 'interpolation',
               'spline'] if method == 'all' else [method]

    for m in methods:
        logger.info(f"Processing data using {m} imputation...")
        new_data, new_activities, new_subjects = preprocessor.preprocess(m)
        logger.info(f"Saving preprocessed data for {m} imputation...")
        preprocessor.save_preprocessed_data(new_data, new_activities, new_subjects, m)


# HELPER FUNCTIONS
def get_sensor_type(sensor_id: int) -> str:
    for sensor_type, properties in SENSOR_RANGES.items():
        if sensor_id in properties['sensors']:
            return sensor_type
    raise ValueError(f"Sensor ID {sensor_id} not found in SENSOR_RANGES")


def get_sensor_range(sensor_id: int) -> tuple:
    for _, properties in SENSOR_RANGES.items():
        if sensor_id in properties['sensors']:
            return properties['range']
    raise ValueError(f"Sensor ID {sensor_id} not found in SENSOR_RANGES")


def get_sensor_unit(sensor_id: int) -> str:
    for _, properties in SENSOR_RANGES.items():
        if sensor_id in properties['sensors']:
            return properties['unit']
    raise ValueError(f"Sensor ID {sensor_id} not found in SENSOR_RANGES")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Sensor Data Analysis and Processing Tool')
    parser.add_argument('mode', choices=[
                        'explore', 'process'], help='Mode of operation: explore data or process data', default='explore')
    parser.add_argument('--method', '-m', choices=['all', 'knn', 'interpolation',
                        'spline'], default='spline', help='Preprocessing method (only for process mode)')
    parser.add_argument('--visualization', '-v', choices=['visualize', 'save', 'both'],
                        default='both', help='Action for visualization plots (only for explore mode)')
    parser.add_argument('--dataset_type', '-dt', choices=[
                        'learning', 'test'], default='learning', help='Type of dataset to load (only for explore mode)')
    parser.add_argument('--data_path', default='.',
                        help='Path to data directory (default: current directory)')
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SensorDataAnalyzer(args.data_path)
    analyzer.load_data(args.dataset_type)

    # Run selected workflow
    if args.mode == 'explore':
        # missing_stats, outlier_stats = explore_data(
        #     analyzer, args.visualization)   # check explore_data doc for args
        explorer_analysis(analyzer, missing_stats=None,
                          outlier_stats=None)
    else:  # process mode
        process_data(analyzer, args.method)
