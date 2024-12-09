import helpers as hlp
import joblib
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fit_or_load_model(fitted_models_path, X_train, y_train, logger, visualize=False, save=True):
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
            estimator=clf,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)

        # === PLOTTING SECTION ===
        results = grid_search.cv_results_

        # Plot the training and testing curves
        plt.figure(figsize=(12, 6))

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
            plt.savefig(os.path.join(
                fitted_models_path, "gradient_boost.png"))
        if visualize:
            plt.show()
        plt.close()
        # === END ===

        best_params = grid_search.best_params_
        logger("Best parameters found: ", best_params)

        joblib.dump(grid_search.best_estimator_, model_path)
        clf = grid_search.best_estimator_
    else:
        logger.info('Loading a pre-fitted gradient boosting model')
        clf = joblib.load(model_path)

    return clf


if __name__ == '__main__':
    # Load the data
    X_train, X_test, y_train = hlp.load_data(data_path='./')
    print(X_train.head())
    print(X_test.head())
    print(y_train.head())

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

    # Fit or load the model
    clf = GradientBoostingClassifier(random_state=0)
    clf = hlp.fit_or_load_model(
        clf, param_grid, X_train, y_train, visualize=False, save=True)

    # Predict on test set
    y_pred = clf.predict(X_test)
    hlp.write_submission(y_pred, 'gradient_boosting.csv')
    _ = hlp.summarize_results(y_pred, 'gradient_boosting_summary.csv')


"""
# For use in CNN to embed each sensor's time series individually
slice_size = 512
num_slices = X_train.shape[1] // slice_size

for i in range(num_slices):
    slice_data = X_train.iloc[:, i*slice_size:(i+1)*slice_size]
    # Process slice_data with your CNN
"""
