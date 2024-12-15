import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq
from keras import layers, models
from sklearn.decomposition import PCA

"""
Feature Extraction:
- extract_features(X_raw): Convert raw sensor time series into statistical & spectral features per sample.
- For 1D sensors: compute statistics on each sensor's 512-point time series.
- For 3D sensors: compute per-axis features and combine them, plus magnitude and cross-axis correlation.

Added features:
- RMS (root mean square)
- Interquartile range (IQR)
- Median absolute deviation (MAD)
- Cross-axis correlations (for 3D sensors only)

At the end, all features become columns in a final DataFrame.

We assume X_raw is a DataFrame: (n_samples, 31*512) with each sensor occupying a block of 512 columns.
No missing values are expected here.

Dimensionality reduction:
- apply_pca(X_train, X_test, n_components)
- apply_autoencoder(X_train, X_test, latent_dim)

NOTE: All computations are done per sample.
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
    if len(signal) == 0:
        return 0
    zero_crossings = np.nonzero(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)

def spectral_entropy(signal):
    fft_vals = np.abs(rfft(signal))
    psd = fft_vals**2
    psd_sum = psd.sum() + 1e-12
    psd_norm = psd / psd_sum
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return entropy

def safe_skew(series):
    val = stats.skew(series)
    return val if not np.isnan(val) else 0.0

def safe_kurtosis(series):
    val = stats.kurtosis(series)
    return val if not np.isnan(val) else 0.0

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
    # Additional time-domain features
    rms_val = np.sqrt(np.mean(series**2))
    iqr_val = np.percentile(series, 75) - np.percentile(series, 25)
    mad_val = np.mean(np.abs(series - mean_val))  # median absolute deviation around mean

    # Handle skew and kurtosis carefully
    skew_val = 0.0 if std_val == 0 else safe_skew(series)
    kurt_val = 0.0 if std_val == 0 else safe_kurtosis(series)

    # Frequency-domain features
    fft_vals = rfft(series)
    fft_freq = rfftfreq(len(series), d=1/SAMPLING_FREQ)
    fft_power = np.abs(fft_vals)**2
    idx_peak = np.argmax(fft_power)
    dominant_freq = fft_freq[idx_peak]

    total_power = fft_power.sum() + 1e-12
    spectral_centroid = np.sum(fft_freq * fft_power) / total_power
    spec_entropy = spectral_entropy(series)

    # Fill dictionary
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
    feats['rms'] = rms_val
    feats['iqr'] = iqr_val
    feats['mad'] = mad_val
    feats['dominant_freq'] = dominant_freq
    feats['spectral_centroid'] = spectral_centroid
    feats['spectral_entropy'] = spec_entropy

    return feats

def extract_3d_features(x, y, z):
    # Per-axis features
    x_feats = extract_1d_features(x)
    y_feats = extract_1d_features(y)
    z_feats = extract_1d_features(z)
    # Magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_feats = extract_1d_features(mag)

    # Cross-axis correlations
    # Pearson correlation between axes
    # If any axis is constant, correlation is 0 by definition
    def safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    xy_corr = safe_corr(x, y)
    xz_corr = safe_corr(x, z)
    yz_corr = safe_corr(y, z)

    feats = {}
    for k, v in x_feats.items():
        feats[f'x_{k}'] = v
    for k, v in y_feats.items():
        feats[f'y_{k}'] = v
    for k, v in z_feats.items():
        feats[f'z_{k}'] = v
    for k, v in mag_feats.items():
        feats[f'mag_{k}'] = v

    # Add cross-axis correlation features
    feats['xy_corr'] = xy_corr
    feats['xz_corr'] = xz_corr
    feats['yz_corr'] = yz_corr

    return feats

def extract_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from raw DataFrame X_raw of shape (n_samples, 31*512).

    Each sample is a row, each sensor is a block of 512 columns.
    We compute features per sample and per sensor (1D) or sensor group (3D).
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

        # stack to (n_samples, 3, 512)
        train_axes = np.stack(axes_data, axis=1)
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

    # Check for NaNs
    if X_features.isna().sum().sum() > 0:
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
