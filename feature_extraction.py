import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from keras import layers, models
from sklearn.decomposition import PCA

"""
Feature Extraction:
- extract_features(X_raw): Convert raw sensor time series into statistical & spectral features.
- apply_pca(X_train, X_test, n_components): Reduce dimensionality using PCA.
- apply_autoencoder(X_train, X_test, latent_dim): Use an auto-encoder for non-linear dimensionality reduction.
No file loading here; all data is passed as DataFrames or arrays.
"""

SAMPLING_FREQ = 100.0
ONE_D_SENSORS = [2, 3, 13, 23]
THREE_D_GROUPS = [
    (4,5,6), (7,8,9), (10,11,12),
    (14,15,16), (17,18,19), (20,21,22),
    (24,25,26), (27,28,29), (30,31,32)
]

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
    return len(zero_crossings) / len(signal) if len(signal) > 0 else 0

def spectral_entropy(signal):
    fft_vals = np.abs(rfft(signal))
    psd = fft_vals**2
    psd_sum = psd.sum() + 1e-12
    psd_norm = psd / psd_sum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def extract_1d_features(series):
    feats = {}
    # Time-domain features
    mean_val = np.mean(series)
    median_val = np.median(series)
    std_val = np.std(series)
    min_val = np.min(series)
    max_val = np.max(series)
    range_val = max_val - min_val
    energy_val = np.sum(series**2)

    # Handle skew and kurtosis carefully to avoid NaN if all values are identical
    if std_val == 0:
        # Constant signal: skew = 0, kurtosis = 0
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = stats.skew(series)
        # If skew is NaN due to floating precision, force to 0
        if np.isnan(skew_val):
            skew_val = 0.0
        kurt_val = stats.kurtosis(series)
        if np.isnan(kurt_val):
            kurt_val = 0.0

    # Frequency-domain features
    fft_vals = rfft(series)
    fft_freq = rfftfreq(len(series), d=1/SAMPLING_FREQ)
    fft_power = np.abs(fft_vals)**2
    idx_peak = np.argmax(fft_power)
    dominant_freq = fft_freq[idx_peak]

    total_power = fft_power.sum() + 1e-12
    spectral_centroid = np.sum(fft_freq * fft_power) / total_power
    spec_entropy = spectral_entropy(series)

    feats['mean'] = mean_val
    feats['median'] = median_val
    feats['std'] = std_val
    feats['min'] = min_val
    feats['max'] = max_val
    feats['range'] = range_val
    feats['skew'] = skew_val
    feats['kurtosis'] = kurt_val
    feats['zcr'] = zero_crossing_rate(series)
    feats['energy'] = energy_val
    feats['dominant_freq'] = dominant_freq
    feats['spectral_centroid'] = spectral_centroid
    feats['spectral_entropy'] = spec_entropy

    # None of these steps produce NaN if input is clean and constant arrays handled
    return feats

def extract_3d_features(x, y, z):
    x_feats = extract_1d_features(x)
    y_feats = extract_1d_features(y)
    z_feats = extract_1d_features(z)
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_feats = extract_1d_features(mag)

    feats = {}
    # Combine them into a single feature vector
    # Each axis' features are prefixed with x_, y_, z_, mag_ respectively
    for k, v in x_feats.items():
        feats[f'x_{k}'] = v
    for k, v in y_feats.items():
        feats[f'y_{k}'] = v
    for k, v in z_feats.items():
        feats[f'z_{k}'] = v
    for k, v in mag_feats.items():
        feats[f'mag_{k}'] = v
    return feats

def extract_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from raw DataFrame X_raw of shape (n_samples, 31*512),
    no missing values assumed (after data_prep).
    Each sensor or group of sensors yields a set of features, concatenated horizontally.
    """

    # 1D sensors
    X_1d_list = []
    for sensor_id in ONE_D_SENSORS:
        start = (sensor_id - 2)*512
        end = start + 512
        sensor_data = X_raw.iloc[:, start:end].values
        feats = [extract_1d_features(sensor_data[i,:]) for i in range(sensor_data.shape[0])]
        df_feats = pd.DataFrame(feats)
        df_feats.columns = [f'sensor_{sensor_id}_{c}' for c in df_feats.columns]
        X_1d_list.append(df_feats)
    X_1d = pd.concat(X_1d_list, axis=1) if X_1d_list else pd.DataFrame()

    # 3D sensors
    X_3d_list = []
    for group in THREE_D_GROUPS:
        axes_data = []
        for sid in group:
            start = (sid - 2)*512
            end = start+512
            sid_data = X_raw.iloc[:, start:end].values
            axes_data.append(sid_data)

        train_axes = np.stack(axes_data, axis=1)  # (n_samples,3,512)
        sensor_type = SENSOR_TYPE_MAP[group[0]]
        feats_3d = []
        for i in range(train_axes.shape[0]):
            x, y_, z = train_axes[i,0,:], train_axes[i,1,:], train_axes[i,2,:]
            f = extract_3d_features(x, y_, z)
            feats_3d.append(f)
        df_feats_3d = pd.DataFrame(feats_3d)
        df_feats_3d.columns = [f'group_{group[0]}_{sensor_type}_{c}' for c in df_feats_3d.columns]
        X_3d_list.append(df_feats_3d)
    X_3d = pd.concat(X_3d_list, axis=1) if X_3d_list else pd.DataFrame()

    X_features = pd.concat([X_1d, X_3d], axis=1)

    # Check for any NaNs due to unexpected issues
    if X_features.isna().sum().sum() > 0:
        # This should not happen if input is clean and code handles constants
        # But if it does, we can fill them or raise a warning
        print("Warning: NaN values found in extracted features. Filling NaNs with 0.0.")
        X_features = X_features.fillna(0.0)

    return X_features

def apply_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int):
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def build_autoencoder(input_dim, latent_dim: int):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    decoded = layers.Dense(256, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    encoder = models.Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def apply_autoencoder(X_train: np.ndarray, X_test: np.ndarray, latent_dim: int):
    print(f"Applying Auto-Encoder with latent dimension {latent_dim}...")
    input_dim = X_train.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, validation_split=0.1, verbose=1)
    X_train_ae = encoder.predict(X_train)
    X_test_ae = encoder.predict(X_test)
    return X_train_ae, X_test_ae
