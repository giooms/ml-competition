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


def load_data(data_path: str, method: str = 'spline') -> tuple:
    """Loads training and testing data from the specified directory."""
    logger.info(f'Loading data...')
    LS_path = os.path.join(data_path, f'processed/{method}')
    TS_path = os.path.join(data_path, 'TS')
    activity_path = os.path.join(data_path, f'processed/')

    if not os.path.exists(LS_path):
        logger.info(f'Preprocessed data not found. Processing data...')
        analyzer = SensorDataAnalyzer(data_path)
        process_data(analyzer, method)

    # Create the training and testing samples
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    for f in tqdm(FEATURES, desc='Loading training and test sets'):
        ls_data = pd.read_pickle(os.path.join(LS_path, f'sensor_{f}.pkl'))
        ts_data = pd.read_csv(os.path.join(
            TS_path, f'TS_sensor_{f}.txt'), delimiter=' ', header=None)

        X_train = pd.concat([X_train, ls_data], axis=1, ignore_index=True)
        X_test = pd.concat([X_test, ts_data], axis=1, ignore_index=True)

    # Create training labels
    y_train = pd.read_pickle(os.path.join(activity_path, 'activities.pkl'))

    # Debug
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)

    # Check data shapes
    assert X_train.shape[0] == y_train.shape[0], 'Number of samples in X_train and y_train do not match'
    assert X_test.shape[0] == TS_N_TIME_SERIES, 'Invalid number of samples in X_test'

    return X_train, X_test, y_train


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


def summarize_results(y_pred, summary_path='example_results_summary.csv'):
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
            verbose=1
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


def plot_training_testing_curves(results, model_name, fitted_models_path, visualize, save):
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
