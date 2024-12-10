import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

"""
Feature Extraction Script - Sensor-Specific

This script loads processed sensor time series data and computes engineered features 
based on the sensor type:

Sensor Mapping:
2: Heart rate (bpm) [1D, slow signal]
3, 13, 23: Temperature (°C) [1D, slow signal]
4-6, 14-16, 24-26: Acceleration (3D)
7-9, 17-19, 27-29: Gyroscope (3D)
10-12, 20-22, 30-32: Magnetometer (3D)

Strategy:
- For 1D sensors (heart rate, temperature):
  * Compute time-domain statistics (mean, median, std, min, max, skew, kurtosis, range).
  * Possibly compute energy (sum of squares) to gauge variability.
  * Frequency-domain analysis might be less critical, but we’ll still include dominant frequency 
    and spectral entropy for completeness, or at least show how to do it.

- For 3D sensors (acc, gyro, mag):
  * Compute axis-wise features (same stats as 1D).
  * Compute vector magnitude = sqrt(x^2 + y^2 + z^2) and extract features from that magnitude signal.
  * Include frequency-domain features as these signals can have distinct periodic patterns (e.g., walking).

After running:
- Produces X_train_features.pkl and X_test_features.pkl containing aggregated features for all sensors.

Next steps after evaluating model results:
- Possibly refine features (remove frequency domain for slow sensors, add more domain-specific transformations, or do feature selection).
"""

METHOD = 'spline'  
DATA_PATH = '.'     
LS_PATH = os.path.join(DATA_PATH, 'processed', METHOD)
TS_PATH = os.path.join(DATA_PATH, 'TS')
FEATURES = range(2, 33)
N_SAMPLES = 3500
N_TIMEPOINTS = 512

# Sampling frequency assumption (adjust if known)
SAMPLING_FREQ = 100.0  # Hz (Example; if unknown, use 1.0 and interpret frequency features relatively)

########################
# Sensor Type Definitions
########################
SENSOR_TYPE_MAP = {
    # Single-dimension sensors
    2: 'heart_rate',
    3: 'temperature', 13: 'temperature', 23: 'temperature',
    
    # 3D sensors grouped by modality
    # Hand
    4: 'acceleration', 5: 'acceleration', 6: 'acceleration',
    7: 'gyroscope', 8: 'gyroscope', 9: 'gyroscope',
    10: 'magnetometer', 11: 'magnetometer', 12: 'magnetometer',
    
    # Chest
    14: 'acceleration', 15: 'acceleration', 16: 'acceleration',
    17: 'gyroscope', 18: 'gyroscope', 19: 'gyroscope',
    20: 'magnetometer', 21: 'magnetometer', 22: 'magnetometer',
    
    # Foot
    24: 'acceleration', 25: 'acceleration', 26: 'acceleration',
    27: 'gyroscope', 28: 'gyroscope', 29: 'gyroscope',
    30: 'magnetometer', 31: 'magnetometer', 32: 'magnetometer'
}

def zero_crossing_rate(signal):
    zero_crossings = np.nonzero(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)

def spectral_entropy(signal):
    fft_vals = np.abs(rfft(signal))
    psd = fft_vals**2
    psd_norm = psd / (np.sum(psd) + 1e-12)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def extract_1d_features(series):
    """Extract features for a single-dimension sensor signal."""
    feats = {}
    # Time domain
    feats['mean'] = np.mean(series)
    feats['median'] = np.median(series)
    feats['std'] = np.std(series)
    feats['min'] = np.min(series)
    feats['max'] = np.max(series)
    feats['range'] = np.max(series) - np.min(series)
    feats['skew'] = stats.skew(series)
    feats['kurtosis'] = stats.kurtosis(series)
    feats['zcr'] = zero_crossing_rate(series)
    feats['energy'] = np.sum(series**2)

    # Frequency domain
    fft_vals = rfft(series)
    fft_freq = rfftfreq(len(series), d=1/SAMPLING_FREQ)
    fft_power = np.abs(fft_vals)**2

    idx_peak = np.argmax(fft_power)
    feats['dominant_freq'] = fft_freq[idx_peak]

    total_power = np.sum(fft_power) + 1e-12
    feats['spectral_centroid'] = np.sum(fft_freq * fft_power) / total_power
    feats['spectral_entropy'] = spectral_entropy(series)
    return feats

def extract_3d_features(x, y, z):
    """Extract features for a 3D sensor. Computes per-axis features and magnitude features."""
    # Per-axis features
    x_feats = extract_1d_features(x)
    y_feats = extract_1d_features(y)
    z_feats = extract_1d_features(z)

    # Compute magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_feats = extract_1d_features(mag)

    # Prefix features
    feats = {}
    for k, v in x_feats.items():
        feats[f'x_{k}'] = v
    for k, v in y_feats.items():
        feats[f'y_{k}'] = v
    for k, v in z_feats.items():
        feats[f'z_{k}'] = v
    for k, v in mag_feats.items():
        feats[f'mag_{k}'] = v
    return feats

# def extract_features_for_sample(sensor_id, series):
#     """
#     Extract features for a single sample from a given sensor_id.
#     If it's a single-dimension sensor (heart_rate or temperature), we use extract_1d_features.
#     If it's a 3D sensor (acceleration, gyroscope, magnetometer), we group their axes together.
#     """
#     sensor_type = SENSOR_TYPE_MAP.get(sensor_id, None)
#     # Identify if it's a standalone (1D) or part of a triplet (3D)
#     if sensor_type in ['heart_rate', 'temperature']:
#         # Single dimension
#         return extract_1d_features(series)
#     else:
#         # Let's handle that logic outside this function. This function will just do 1D extraction.
#         # For 3D sensors, we won't call this function directly. Instead, we will call a separate routine.
#         return extract_1d_features(series)  # Placeholder if accidentally called.
#         # In the main loop, we will handle 3D grouping.

def get_3d_sensor_groups():
    """
    Return a list of tuples, each containing the three sensor IDs for the 3D sensors.
    Example:
    Hand acceleration: (4, 5, 6)
    Hand gyroscope: (7, 8, 9)
    ...
    """
    # We know from the mapping that each triplet is consecutive.
    # We can hardcode or dynamically find them:
    groups = [
        (4,5,6), (7,8,9), (10,11,12),
        (14,15,16), (17,18,19), (20,21,22),
        (24,25,26), (27,28,29), (30,31,32)
    ]
    return groups

def get_1d_sensors():
    return [2, 3, 13, 23]

def extract_features():
    # Load activity labels for sanity check
    y = pd.read_pickle(os.path.join(os.path.dirname(LS_PATH), 'activities.pkl'))

    # Load 1D sensors
    X_train_1d = []
    X_test_1d = []

    print("Extracting features for 1D sensors (heart_rate, temperature)...")
    for sensor_id in get_1d_sensors():
        train_data = pd.read_pickle(os.path.join(LS_PATH, f'sensor_{sensor_id}.pkl')).values
        test_data = pd.read_csv(os.path.join(TS_PATH, f'TS_sensor_{sensor_id}.txt'), delimiter=' ', header=None).values

        assert train_data.shape[0] == y.shape[0]
        assert train_data.shape[1] == N_TIMEPOINTS
        assert test_data.shape[0] == N_SAMPLES
        assert test_data.shape[1] == N_TIMEPOINTS

        # Extract features per sample
        train_feats = [extract_1d_features(train_data[i,:]) for i in range(train_data.shape[0])]
        test_feats = [extract_1d_features(test_data[i,:]) for i in range(test_data.shape[0])]

        df_train = pd.DataFrame(train_feats)
        df_test = pd.DataFrame(test_feats)

        df_train.columns = [f'sensor_{sensor_id}_{c}' for c in df_train.columns]
        df_test.columns = [f'sensor_{sensor_id}_{c}' for c in df_test.columns]

        X_train_1d.append(df_train)
        X_test_1d.append(df_test)

    # Concatenate all 1D sensors
    if len(X_train_1d) > 0:
        X_train_1d = pd.concat(X_train_1d, axis=1)
        X_test_1d = pd.concat(X_test_1d, axis=1)
    else:
        X_train_1d = pd.DataFrame()
        X_test_1d = pd.DataFrame()

    # Load and process 3D sensors
    print("Extracting features for 3D sensors (acceleration, gyroscope, magnetometer)...")
    # Each group of 3 sensors forms one vector sensor (x,y,z)
    X_train_3d = []
    X_test_3d = []

    groups = get_3d_sensor_groups()
    for group in groups:
        # group is a tuple like (4,5,6)
        # Load data for each axis
        train_axes = []
        test_axes = []
        for sid in group:
            train_data = pd.read_pickle(os.path.join(LS_PATH, f'sensor_{sid}.pkl')).values
            test_data = pd.read_csv(os.path.join(TS_PATH, f'TS_sensor_{sid}.txt'), delimiter=' ', header=None).values

            # shape checks
            assert train_data.shape[0] == y.shape[0]
            assert train_data.shape[1] == N_TIMEPOINTS
            assert test_data.shape[0] == N_SAMPLES
            assert test_data.shape[1] == N_TIMEPOINTS

            train_axes.append(train_data)
            test_axes.append(test_data)

        # Stack them: now we have train_axes as [x_data, y_data, z_data], each shape (n_samples, 512)
        # Convert to np.array with shape (n_samples, 3, 512)
        train_axes = np.stack(train_axes, axis=1)
        test_axes = np.stack(test_axes, axis=1)

        # Extract features per sample
        train_feats = []
        test_feats = []

        for i in range(train_axes.shape[0]):
            x = train_axes[i,0,:]
            y_ = train_axes[i,1,:]
            z = train_axes[i,2,:]
            f = extract_3d_features(x, y_, z)
            train_feats.append(f)

        for i in range(test_axes.shape[0]):
            x = test_axes[i,0,:]
            y_ = test_axes[i,1,:]
            z = test_axes[i,2,:]
            f = extract_3d_features(x, y_, z)
            test_feats.append(f)

        df_train = pd.DataFrame(train_feats)
        df_test = pd.DataFrame(test_feats)

        # Name columns: e.g. for group (4,5,6) -> "hand_acceleration" if we can map from sensor_id to location.
        sensor_type = SENSOR_TYPE_MAP[group[0]]
        df_train.columns = [f'group_{group[0]}_{sensor_type}_{c}' for c in df_train.columns]
        df_test.columns = [f'group_{group[0]}_{sensor_type}_{c}' for c in df_test.columns]

        X_train_3d.append(df_train)
        X_test_3d.append(df_test)

    if len(X_train_3d) > 0:
        X_train_3d = pd.concat(X_train_3d, axis=1)
        X_test_3d = pd.concat(X_test_3d, axis=1)
    else:
        X_train_3d = pd.DataFrame()
        X_test_3d = pd.DataFrame()

    # Combine 1D and 3D features
    X_train_features = pd.concat([X_train_1d, X_train_3d], axis=1)
    X_test_features = pd.concat([X_test_1d, X_test_3d], axis=1)

    return X_train_features, X_test_features


if __name__ == "__main__":
    print("Starting sensor-specific feature extraction...")
    X_train_features, X_test_features = extract_features()

    output_dir = os.path.join('.', 'processed_features')
    os.makedirs(output_dir, exist_ok=True)
    X_train_features_path = os.path.join(output_dir, 'X_train_features.pkl')
    X_test_features_path = os.path.join(output_dir, 'X_test_features.pkl')

    X_train_features.to_pickle(X_train_features_path)
    X_test_features.to_pickle(X_test_features_path)

    print(f"Feature extraction completed. Files saved to:\n{X_train_features_path}\n{X_test_features_path}")

    # Next steps:
    # 1. Update your model training scripts (e.g., gradient_boosting.py, random_forest.py)
    #    to load X_train_features.pkl and X_test_features.pkl instead of raw data.
    # 2. Evaluate the models. If performance improves or changes, consider:
    #    - Further feature selection
    #    - Adjusting or removing certain features (like frequency features from temperature sensors)
    #    - Incorporating domain knowledge (e.g., computing step count from acceleration)
