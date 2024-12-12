"""
Step 1. Start with Gradient Boosting Classifier:
- Test performance using raw normalized data (or CNN-extracted embeddings - see optional step 1.2).
- Compare results to Random Forest as a baseline.

Step 2. Feature Selection:
- If performance with all features is slow or suboptimal, explore RFE or SelectFromModel.
- This will reduce dimensionality while retaining the most relevant information for predicting the activity.
- Compare with results obtained from step 1 alone.

(optional) Step 1.2 Experiment with CNNs for Feature Extraction:
- Train a CNN to extract meaningful, lower-dimensional representations (dense vectors) of the time-series data.
- These embeddings will serve as the new dataset, replacing the original high-dimensional raw time-series data.
- With this new dataset, repeat from step 1. If results are still poor, perform additional Feature Selection step)

In total, four workflows to explore:
a) Step 1 alone
b) Step 1 and 2
c) Step 1 and 1.2
d) Step 1, 1.2 and 2
"""

import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from data_prep import process_data, SensorDataAnalyzer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from typing import Tuple
from tqdm import tqdm
from typing import Union
from xgboost import XGBClassifier

TS_N_TIME_SERIES = 3500
FEATURES = range(2, 33)
root_path = '.'

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data(data_path: str, method: str='spline') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads training and test data, activities, and subjects from processed data.
       Applies the same scaling to the test set as used in the training set.
    """
    logger.info('Loading raw data...')
    LS_path = os.path.join(data_path, f'processed/{method}')
    TS_path = os.path.join(data_path, 'TS')
    activity_path = os.path.join(data_path, 'processed')
    scaler_dir = os.path.join(LS_path, 'scalers')

    if not os.path.exists(LS_path):
        raise FileNotFoundError("Preprocessed data not found. Run process_data first.")

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()

    for f in tqdm(FEATURES, desc='Loading and scaling test sets'):
        # Load already preprocessed (and scaled) training data
        ls_data = pd.read_pickle(os.path.join(LS_path, f'sensor_{f}.pkl'))
        # Raw test data
        ts_data = pd.read_csv(os.path.join(TS_path, f'TS_sensor_{f}.txt'), delimiter=' ', header=None).values

        # Load scaler for this sensor
        scaler_path = os.path.join(scaler_dir, f'scaler_{f}.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found for sensor {f} at {scaler_path}")
        global_min, global_max = joblib.load(scaler_path)

        # Apply scaling to test data
        # Note: training data is already scaled by data_prep.py
        ts_data_scaled = (ts_data - global_min) / (global_max - global_min)

        X_train = pd.concat([X_train, ls_data], axis=1, ignore_index=True)
        X_test = pd.concat([X_test, pd.DataFrame(ts_data_scaled)], axis=1, ignore_index=True)

    y_train = pd.read_pickle(os.path.join(activity_path, 'activities.pkl'))
    subjects = pd.read_pickle(os.path.join(activity_path, 'subjects.pkl'))

    assert X_train.shape[0] == y_train.shape[0], 'Mismatch in X_train and y_train'
    assert X_train.shape[0] == subjects.shape[0], 'Mismatch in X_train and subjects'
    assert X_test.shape[0] == TS_N_TIME_SERIES, 'Invalid number of samples in X_test'

    return X_train, X_test, y_train, subjects

def apply_rfe(X_train: pd.DataFrame, y_train: pd.Series, n_features: int=100) -> np.ndarray:
    """Applies RFE to select n_features from raw data."""
    logger.info("Applying RFE...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    support = rfe.support_
    logger.info(f"RFE completed. Selected {support.sum()} features out of {X_train.shape[1]}")
    return support

def fit_model(X_train, y_train, model_type='rf'):
    """Fits a model (RF or XGB) with GridSearchCV and returns best estimator. Always trains fresh, no loading."""
    logger.info(f'Fitting a new {model_type.upper()} model. This may take a while...')
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['auto', 'sqrt']
        }
        # param_grid = {
        #     'n_estimators': [50, 100],
        #     'max_depth': [5, 10]
        # }
    else:
         # Attempt GPU mode first
        try:
            model = XGBClassifier(
                random_state=42,
                tree_method='gpu_hist',    # GPU mode
                predictor='gpu_predictor'  # GPU predictor
            )
        except XGBoostError:
            # If GPU is not available, fallback to CPU mode
            model = XGBClassifier(
                random_state=42,
                tree_method='hist',        # CPU mode
                predictor='cpu_predictor'  # CPU predictor
            )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3]
        }
        # param_grid = {
        #     'n_estimators': [50, 100],
        #     'max_depth': [5, 10],
        #     'learning_rate': [0.1]
        # }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_and_save(y_pred, output_path='results_summary.csv'):
    # [Unchanged logic]
    unique, counts = np.unique(y_pred, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    total = len(y_pred)
    proportions = {cls: cnt/total for cls, cnt in class_distribution.items()}

    summary_df = pd.DataFrame({
        'Class': list(class_distribution.keys()),
        'Count': list(class_distribution.values()),
        'Proportion': list(proportions.values())
    }).sort_values(by='Class').reset_index(drop=True)

    results_dir = os.path.join(root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, output_path)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

def write_submission(y, submission_path='example_submission.csv'):
    """
    Writes the predictions to a CSV file in the required submission format.
    Parameters:
        y (numpy.ndarray): Array of predicted class labels.
        submission_path (str): Path to the submission file. Default is 'example_submission.csv'.
    Raises:
        ValueError: If any predicted class label is outside the range [1, 14].
        ValueError: If the number of predicted values is not 3500.
    **Notes:**
    - The function ensures the parent directory of the submission file exists.
    - If the submission file already exists, it will be removed before writing the new file.
    - The function writes the predictions in the format 'Id,Prediction' with 1-based indexing for the Id.
    """
    submission_path = os.path.join("submissions", submission_path)
    parent_dir = os.path.dirname(submission_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(submission_path):
        os.remove(submission_path)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))

    # Write submission file
    with open(submission_path, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    logger.info(f'Submission saved to {submission_path}.')

def get_subject_splits(subjects: pd.Series) -> list:
    """
    Given the subjects series, returns a list of (train_idx, val_idx, val_sub) splits
    for leave-one-subject-out cross-validation.
    """
    unique_subs = subjects.unique()
    splits = []
    for val_sub in unique_subs:
        val_idx = subjects[subjects == val_sub].index
        train_idx = subjects[subjects != val_sub].index
        splits.append((train_idx, val_idx, val_sub))
    return splits

def run_scenario(data_path: str, method: str, model_type: str, scenario: str,
                 n_features: int=50, latent_dim: int=50, fold: int=None):
    """
    Scenarios:
    A = Raw data
    B = Raw + RFE
    C = FE only
    D = FE + PCA
    E = FE + AE

    If fold is provided, do LOSO (leave-one-subject-out). Otherwise use full training set.
    """
    X_train_raw, X_test_raw, y_train, subjects = load_raw_data(data_path, method=method)

    if fold is not None:
        splits = get_subject_splits(subjects)
        if fold < 1 or fold > len(splits):
            raise ValueError(f"Fold {fold} out of range. {len(splits)} subjects total.")
        train_idx, val_idx, val_sub = splits[fold-1]
        logger.info(f"LOSO CV: Using subject {val_sub} as validation")
        X_train_portion = X_train_raw.loc[train_idx]
        y_train_portion = y_train.loc[train_idx]
    else:
        X_train_portion = X_train_raw
        y_train_portion = y_train

    if scenario == 'A':
        X_train_processed = X_train_portion
        X_test_processed = X_test_raw
    elif scenario == 'B':
        support = apply_rfe(X_train_portion, y_train_portion, n_features=n_features)
        X_train_processed = X_train_portion.loc[:, support]
        X_test_processed = X_test_raw.loc[:, support]
    else:
        # Feature Extraction cases
        from feature_extraction import extract_features, apply_pca, apply_autoencoder
        # Now we call extract_features without 'method'
        X_train_fe = extract_features(X_train_portion)
        X_test_fe = extract_features(X_test_raw)

        if scenario == 'C':
            X_train_processed = X_train_fe
            X_test_processed = X_test_fe
        elif scenario == 'D':
            X_train_pca, X_test_pca = apply_pca(X_train_fe, X_test_fe, latent_dim)
            X_train_processed = pd.DataFrame(X_train_pca)
            X_test_processed = pd.DataFrame(X_test_pca)
        elif scenario == 'E':
            X_train_ae, X_test_ae = apply_autoencoder(X_train_fe.values, X_test_fe.values, latent_dim)
            X_train_processed = pd.DataFrame(X_train_ae)
            X_test_processed = pd.DataFrame(X_test_ae)
        else:
            raise ValueError("Scenario must be one of A,B,C,D,E.")

    # Optional: Check for NaNs before fitting (debugging)
    if X_train_processed.isna().sum().sum() > 0:
        logger.warning("NaNs detected in X_train_processed. Consider filling or investigating upstream steps.")
    if X_test_processed.isna().sum().sum() > 0:
        logger.warning("NaNs detected in X_test_processed. Consider filling or investigating upstream steps.")

    clf = fit_model(X_train_processed, y_train_portion, model_type=model_type)
    logger.info("Predicting on test set...")
    y_pred = clf.predict(X_test_processed)

    suffix = f"_fold{fold}" if fold is not None else ""
    evaluate_and_save(y_pred, output_path=f"{model_type}_{scenario}{suffix}_summary.csv")
    write_submission(y_pred, submission_path=f"{model_type}_{scenario}{suffix}_submission.csv")

    # Compute LOSO validation results if fold is specified
    if fold is not None:
        if X_val_processed is not None:
            y_val_pred = clf.predict(X_val_processed)
            # Compute accuracy
            from sklearn.metrics import accuracy_score
            val_acc = accuracy_score(y_val_portion, y_val_pred)
            logger.info(f"LOSO Validation Accuracy for scenario {scenario}, model {model_type}, fold {fold}: {val_acc:.4f}")
            # Save LOSO result to a file, e.g., results/loso_results.csv
            loso_path = os.path.join(root_path, 'results', 'loso_results.csv')
            os.makedirs(os.path.join(root_path, 'results'), exist_ok=True)
            header_needed = not os.path.exists(loso_path)
            with open(loso_path, 'a') as f:
                if header_needed:
                    f.write("Scenario,Model,Fold,Validation_Accuracy\n")
                f.write(f"{scenario},{model_type},{fold},{val_acc}\n")
        else:
            logger.warning("No validation portion found for LOSO validation accuracy computation.")

def summarize_results(y_pred: Union[pd.Series, np.ndarray], summary_path='example_results_summary.csv') -> pd.DataFrame:
    # Calculate the distribution of predicted classes
    unique, counts = np.unique(y_pred, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Calculate the proportion of each class
    total_predictions = len(y_pred)
    proportions = {cls: count / total_predictions for cls,
                   count in class_distribution.items()}

    # Create a DataFrame to hold the class distribution and proportions
    summary_df = pd.DataFrame({
        'Class': list(class_distribution.keys()),
        'Count': list(class_distribution.values()),
        'Proportion': list(proportions.values())
    })

    # Sort the DataFrame by class
    summary_df = summary_df.sort_values(by='Class').reset_index(drop=True)
    # print(summary_df)

    # Save the summary table to a CSV file
    summary_dir = os.path.join(root_path, 'results')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, summary_path)

    summary_df.to_csv(summary_path, index=False, mode='w')
    logger.info(f"Summary saved to {summary_path}")

    return summary_df


def fit_or_load_model(model: Union[RandomForestClassifier, GradientBoostingClassifier, XGBClassifier], param_grid: dict, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray], visualize: bool = False, save: bool = False) -> Union[GradientBoostingClassifier, RandomForestClassifier]:
    """Fits the specified model using GridSearchCV or loads a pre-fitted model."""
    # Check the input types
    if not isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
        raise ValueError(
            'Model must be an instance of RandomForestClassifier, GradientBoostingClassifier, or XGBClassifier')
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
        raise ValueError('X_train must be a DataFrame or a 2D ndarray')
    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise ValueError('y_train must be a Series or a 1D ndarray')

    # Get the model name
    model_name = type(model).__name__

    # Validate param_grid keys
    model_params = model.get_params()
    for param in param_grid.keys():
        if param not in model_params:
            raise ValueError(
                f"Parameter '{param}' is not a valid parameter for the model {model_name}")

    # Create the directory to save the fitted models
    fitted_models_path = os.path.join(root_path, 'fitted_models')
    os.makedirs(fitted_models_path, exist_ok=True)

    # Create the model path based on the model name
    model_path = os.path.join(
        fitted_models_path, f'{model_name.lower()}_model.pkl')

    if not os.path.exists(model_path):
        logger.info(
            f'Fitting a {model_name} model. This may take a while...')

        # Fit the model using GridSearchCV
        clf = model
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        logger.info("Best parameters found: ", best_params)

        joblib.dump(grid_search.best_estimator_, model_path)
        clf = grid_search.best_estimator_

        # Plot the training and testing curves
        plot_training_testing_curves(
            grid_search.cv_results_, model_name.lower(), fitted_models_path, visualize, save)
    else:
        logger.info(f'Loading a pre-fitted {model_name} model...')
        clf = joblib.load(model_path)

    return clf


def plot_training_testing_curves(results: dict, model_name: str, fitted_models_path: str, visualize: bool=False, save: bool=True) -> None:
    """Plots the training and testing curves."""
    plt.figure(figsize=(12, 6))

    if 'param_n_estimators' in results:
        # Plot training and testing scores
        plt.plot(results['param_n_estimators'], results['mean_train_score'],
                 label='Training Score', marker='o', color='blue')
        plt.plot(results['param_n_estimators'], results['mean_test_score'],
                 label='Testing Score', marker='x', color='red')

        plt.xlabel('Number of Estimators')
        plt.ylabel('Score')
        plt.title('Training and Testing Curves')
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(fitted_models_path, f'{model_name}.png'))
        if visualize:
            plt.show()
        plt.close()
    else:
        print("The parameter 'param_n_estimators' is not available in the results.")
