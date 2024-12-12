import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
import argparse
import logging

from sklearn.decomposition import PCA
from tensorflow.keras import layers, models

"""
Feature Extraction Script with Optional PCA or Auto-Encoder Reduction

This script:
1. Extracts features from raw sensor data (time-domain, frequency-domain).
2. Optionally applies dimensionality reduction using PCA or an Auto-Encoder.

Command-line arguments:
--reduction_method {none,pca,ae} : Choose the dimensionality reduction method.
--n_components : Number of components for PCA or latent dimension for AE.
--data_path : Path to data directory.
--method : Preprocessing method used (default 'spline').

Output:
- Always saves the full feature set to 'X_train_features.pkl' and 'X_test_features.pkl'.
- If PCA is chosen, also saves 'X_train_features_pca.pkl' and 'X_test_features_pca.pkl'.
- If AE is chosen, also saves 'X_train_features_ae.pkl' and 'X_test_features_ae.pkl'.

Dependencies:
- Ensure 'activities.pkl' and processed data are available.
- Keras (TensorFlow) is required for AE.
"""

# Default constants
METHOD = 'spline'
DATA_PATH = '.'
N_SAMPLES = 3500
N_TIMEPOINTS = 512
SAMPLING_FREQ = 100.0  # Assuming 100Hz, adjust if known.

########################
# Sensor Type Definitions
########################

SENSOR_TYPE_MAP = {
    2: 'heart_rate',
    3: 'temperature', 13: 'temperature', 23: 'temperature',
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
    x_feats = extract_1d_features(x)
    y_feats = extract_1d_features(y)
    z_feats = extract_1d_features(z)
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_feats = extract_1d_features(mag)

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

def get_3d_sensor_groups():
    groups = [
        (4,5,6), (7,8,9), (10,11,12),
        (14,15,16), (17,18,19), (20,21,22),
        (24,25,26), (27,28,29), (30,31,32)
    ]
    return groups

def get_1d_sensors():
    return [2, 3, 13, 23]

def extract_features(data_path=DATA_PATH, method=METHOD):
    activity_path = os.path.join(data_path, f'processed/')
    y = pd.read_pickle(os.path.join(activity_path, 'activities.pkl'))

    LS_path = os.path.join(data_path, 'processed', method)
    TS_path = os.path.join(data_path, 'TS')

    # 1D sensors
    X_train_1d = []
    X_test_1d = []
    print("Extracting features for 1D sensors (heart_rate, temperature)...")
    for sensor_id in get_1d_sensors():
        train_data = pd.read_pickle(os.path.join(LS_path, f'sensor_{sensor_id}.pkl')).values
        test_data = pd.read_csv(os.path.join(TS_path, f'TS_sensor_{sensor_id}.txt'), delimiter=' ', header=None).values

        train_feats = [extract_1d_features(train_data[i,:]) for i in range(train_data.shape[0])]
        test_feats = [extract_1d_features(test_data[i,:]) for i in range(test_data.shape[0])]

        df_train = pd.DataFrame(train_feats)
        df_test = pd.DataFrame(test_feats)

        df_train.columns = [f'sensor_{sensor_id}_{c}' for c in df_train.columns]
        df_test.columns = [f'sensor_{sensor_id}_{c}' for c in df_test.columns]

        X_train_1d.append(df_train)
        X_test_1d.append(df_test)

    if len(X_train_1d) > 0:
        X_train_1d = pd.concat(X_train_1d, axis=1)
        X_test_1d = pd.concat(X_test_1d, axis=1)
    else:
        X_train_1d = pd.DataFrame()
        X_test_1d = pd.DataFrame()

    # 3D sensors
    print("Extracting features for 3D sensors (acceleration, gyroscope, magnetometer)...")
    X_train_3d = []
    X_test_3d = []
    groups = get_3d_sensor_groups()
    for group in groups:
        LS_path_group = [os.path.join(LS_path, f'sensor_{sid}.pkl') for sid in group]
        TS_path_group = [os.path.join(TS_path, f'TS_sensor_{sid}.txt') for sid in group]

        train_axes = []
        test_axes = []
        for sid_path_ls, sid_path_ts in zip(LS_path_group, TS_path_group):
            train_data = pd.read_pickle(sid_path_ls).values
            test_data = pd.read_csv(sid_path_ts, delimiter=' ', header=None).values
            train_axes.append(train_data)
            test_axes.append(test_data)

        train_axes = np.stack(train_axes, axis=1)  # (n_samples, 3, 512)
        test_axes = np.stack(test_axes, axis=1)

        train_feats = []
        test_feats = []
        sensor_type = SENSOR_TYPE_MAP[group[0]]

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

    X_train_features = pd.concat([X_train_1d, X_train_3d], axis=1)
    X_test_features = pd.concat([X_test_1d, X_test_3d], axis=1)

    return X_train_features, X_test_features


def apply_pca(X_train, X_test, n_components):
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def build_autoencoder(input_dim, latent_dim):
    # Simple AE model
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    decoded = layers.Dense(256, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    encoder = models.Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def apply_autoencoder(X_train, X_test, latent_dim):
    print(f"Applying Auto-Encoder with latent dimension {latent_dim}...")
    input_dim = X_train.shape[1]

    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    # Train AE on training set
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, validation_split=0.1, verbose=1)

    X_train_ae = encoder.predict(X_train)
    X_test_ae = encoder.predict(X_test)
    return X_train_ae, X_test_ae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature extraction with optional PCA/AE reduction.')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data directory')
    parser.add_argument('--method', type=str, default='spline', help='Preprocessing method')
    parser.add_argument('--reduction_method', type=str, default='none', choices=['none', 'pca', 'ae'],
                        help='Dimensionality reduction method to use: none, pca, ae')
    parser.add_argument('--n_components', type=int, default=50, help='Number of components for PCA or latent_dim for AE')

    args = parser.parse_args()

    print("Starting feature extraction...")
    X_train_features, X_test_features = extract_features(data_path=args.data_path, method=args.method)

    output_dir = os.path.join('.', 'processed_features')
    os.makedirs(output_dir, exist_ok=True)

    # Always save the full feature set
    X_train_features_path = os.path.join(output_dir, 'X_train_features.pkl')
    X_test_features_path = os.path.join(output_dir, 'X_test_features.pkl')

    X_train_features.to_pickle(X_train_features_path)
    X_test_features.to_pickle(X_test_features_path)

    if args.reduction_method == 'pca':
        X_train_red, X_test_red = apply_pca(X_train_features, X_test_features, args.n_components)
        X_train_red_df = pd.DataFrame(X_train_red)
        X_test_red_df = pd.DataFrame(X_test_red)
        X_train_red_path = os.path.join(output_dir, f'X_train_features_pca.pkl')
        X_test_red_path = os.path.join(output_dir, f'X_test_features_pca.pkl')
        X_train_red_df.to_pickle(X_train_red_path)
        X_test_red_df.to_pickle(X_test_red_path)
        print(f"PCA reduced features saved to {X_train_red_path} and {X_test_red_path}")

    elif args.reduction_method == 'ae':
        # Convert to numpy array if not already
        X_train_arr = X_train_features.values
        X_test_arr = X_test_features.values

        X_train_red, X_test_red = apply_autoencoder(X_train_arr, X_test_arr, args.n_components)
        X_train_red_df = pd.DataFrame(X_train_red)
        X_test_red_df = pd.DataFrame(X_test_red)
        X_train_red_path = os.path.join(output_dir, f'X_train_features_ae.pkl')
        X_test_red_path = os.path.join(output_dir, f'X_test_features_ae.pkl')
        X_train_red_df.to_pickle(X_train_red_path)
        X_test_red_df.to_pickle(X_test_red_path)
        print(f"AE reduced features saved to {X_train_red_path} and {X_test_red_path}")

    else:
        print("No dimensionality reduction applied. Only full feature sets are saved.")
