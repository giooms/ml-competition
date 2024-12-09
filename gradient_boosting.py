import joblib
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from data_prep import process_data, SensorDataAnalyzer
from main import write_submission
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TS_N_TIME_SERIES = 3500
FEATURES = range(2, 33)


def load_data(data_path: str, method: str = 'spline') -> tuple:
    """Loads training and testing data from the specified directory."""
    logger.info(f'Loading data')
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
    print(activity_path)
    y_train = pd.read_pickle(os.path.join(activity_path, 'activities.pkl'))

    # Debug
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)

    # Check data shapes
    assert X_train.shape[0] == y_train.shape[0], 'Number of samples in X_train and y_train do not match'
    assert X_test.shape[0] == TS_N_TIME_SERIES, 'Invalid number of samples in X_test'

    return X_train, X_test, y_train


def fit_or_load_model(fitted_models_path, X_train, y_train, logger, visualize=False):
    """Fits a gradient boosting model using GridSearchCV or loads a pre-fitted model."""
    model_path = os.path.join(
        fitted_models_path, 'gradient_boosting_model.pkl')

    if not os.path.exists(model_path):
        logger.info('Fitting a gradient boosting model. This may take a while.')

        # Hyperparameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'max_depth': [2, 3, 4, 5, 6, 7, 10],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }

        # Fit the model using GridSearchCV
        clf = GradientBoostingClassifier(random_state=0)
        grid_search = GridSearchCV(
            estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # Visualize the results
        if visualize:
            # Extract the results
            results = grid_search.cv_results_

            # Plot the training and testing curves
            plt.figure(figsize=(12, 6))

            # Plot training and testing scores
            plt.plot(results['param_n_estimators'], results['mean_train_score'], label='Training Score', marker='o', color='blue')
            plt.plot(results['param_n_estimators'], results['mean_test_score'], label='Testing Score', marker='x', color='red')

            plt.xlabel('Number of Estimators')
            plt.ylabel('Score')
            plt.title('Training and Testing Curves')
            plt.legend()
            plt.grid(True)
            plt.show(block=False)  # Non-blocking show
            plt.close()

        best_params = grid_search.best_params_
        print("Best parameters found: ", best_params)

        joblib.dump(grid_search.best_estimator_, model_path)
        clf = grid_search.best_estimator_
    else:
        logger.info('Loading a pre-fitted gradient boosting model')
        clf = joblib.load(model_path)

    return clf


if __name__ == '__main__':
    root_path = './'
    fitted_models_path = os.join(root_path, 'fitted_models')
    os.makedirs(fitted_models_path, exist_ok=True)

    # Load the data
    X_train, X_test, y_train = load_data(data_path=root_path)
    print(X_train.head())

    # Fit or load the model
    clf = fit_or_load_model(fitted_models_path, X_train, y_train, logger)

    # Predict on test set
    y_pred = clf.predict(X_test)
    write_submission(y_pred, 'gradient_boosting.csv')


"""
# For use in CNN to embed each sensor's time series individually
slice_size = 512
num_slices = X_train.shape[1] // slice_size

for i in range(num_slices):
    slice_data = X_train.iloc[:, i*slice_size:(i+1)*slice_size]
    # Process slice_data with your CNN
"""
