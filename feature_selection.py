import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sandbox import MISSING_VALUE, analyze_df_missing_patterns


def preprocess_and_load_data(raw_dir: str = 'LS', missing_threshold: float = 0.25, seq_threshold: int = 50):
    """
    Preprocess raw sensor data and return preprocessed data and labels.

    Args:
        raw_dir (str): Directory containing raw sensor data.
        missing_threshold (float): Percentage threshold for missing values.
        seq_threshold (int): Maximum number of missing sequences allowed.

    Returns:
        tuple: Dictionary of sensors (key: sensor ID, value: DataFrame), synchronized activity labels (y),
               and a dictionary of original feature indices for each sensor.
    """
    sensors = {}
    original_indices = {}
    y_file = os.path.join(raw_dir, 'activity_Id.txt')
    y = pd.read_csv(y_file, delimiter=' ', header=None).squeeze()

    global_row_mask = pd.Series([True] * 3500)

    # Create global mask for all sensors
    for i in range(2, 33):
        sensor_file = os.path.join(raw_dir, f'LS_sensor_{i}.txt')
        data = pd.read_csv(sensor_file, delimiter=' ', header=None)
        data.replace(MISSING_VALUE, np.nan, inplace=True)

        patterns_df = analyze_df_missing_patterns(data)
        series_to_drop = (
            (patterns_df['missing_percentage'] > missing_threshold) |
            (patterns_df['num_sequences'] > seq_threshold)
        )
        global_row_mask &= ~series_to_drop

    # Clean sensors and keep original feature indices
    for i in range(2, 33):
        sensor_file = os.path.join(raw_dir, f'LS_sensor_{i}.txt')
        data = pd.read_csv(sensor_file, delimiter=' ', header=None)
        data.replace(MISSING_VALUE, np.nan, inplace=True)

        data_cleaned = data[global_row_mask].reset_index(drop=True)
        sensors[i] = data_cleaned

        # Store original feature indices
        original_indices[i] = np.arange(data.shape[1])

    # Clean labels
    y_cleaned = y[global_row_mask].reset_index(drop=True)

    return sensors, y_cleaned, original_indices


def feature_selection_comparison(sensors: dict, y: np.ndarray, original_indices: dict, model, max_features=100, output_dir='feature_selection_results'):
    """
    Perform feature selection using SelectKBest and RFE with the specified model.

    Args:
        sensors (dict): A dictionary of sensor data (key: sensor_id, value: feature matrix).
        y (np.ndarray): Labels vector.
        original_indices (dict): A dictionary mapping sensor IDs to original feature indices.
        model: Model to be used for RFE.
        max_features (int): Maximum number of features to select.
        output_dir (str): Directory to save results.

    Returns:
        None
    """
    model_name = type(model).__name__
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    for sensor_id, X in sensors.items():
        sensor_file = os.path.join(output_dir, f"sensor_{sensor_id}_results.txt")
        with open(sensor_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Sensor {sensor_id} ===\n")
            X = X.values

            # Step 1: Filter Techniques (SelectKBest)
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            scores = selector.scores_

            # Sort by descending scores and keep top features
            sorted_indices = np.argsort(-scores)
            top_features_filter = sorted_indices[:max_features]

            # Map back to original feature indices
            original_feature_indices = original_indices[sensor_id][top_features_filter]

            f.write(f"\nTop {max_features} features (SelectKBest):\n")
            for idx, original_idx in zip(top_features_filter, original_feature_indices):
                f.write(f"Feature {idx + 1} (original: {original_idx + 1}): Score = {scores[idx]:.4f}\n")

            # Step 2: Wrapper Techniques (RFE with the specified model)
            rfe = RFE(model, n_features_to_select=max_features // 2)
            rfe.fit(X[:, top_features_filter], y)

            # Get top features from RFE
            rfe_selected_features = np.where(rfe.support_)[0]
            original_rfe_indices = original_feature_indices[rfe_selected_features]

            f.write(f"\nTop {max_features // 2} features ({model_name} - RFE):\n")
            for idx, original_idx in zip(rfe_selected_features, original_rfe_indices):
                f.write(f"Feature {idx + 1} (original: {original_idx + 1})\n")

            # Evaluate model performance with cross-validation
            cv_scores = cross_val_score(model, X[:, top_features_filter][:, rfe_selected_features], y, cv=5)
            mean_score = np.mean(cv_scores)
            f.write(f"\n{model_name} Mean CV Score: {mean_score:.4f}\n")


if __name__ == "__main__":
    # Load data for all sensors
    data_dir = 'LS'  # Adjust this path to your data directory
    sensors, y, original_indices = preprocess_and_load_data(data_dir)

    # Perform feature selection and comparison
    feature_selection_comparison(
        sensors=sensors,
        y=y,
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        original_indices=original_indices,
        max_features=100,
        output_dir='feature_selection_results'
    )

    print("Feature selection results saved.")
