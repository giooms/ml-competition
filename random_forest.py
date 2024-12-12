import helpers as hlp
import logging
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('Random Forest Classifier')

    # Load the data
    X_train, X_test, y_train = hlp.load_data(data_path='./', method='spline', use_rfe=True, n_features_to_select=50)
    # print(X_train.head())
    # print(X_test.head())
    # print(y_train.head())

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 8],
        'max_features': ['auto', 'sqrt']
    }

    # Fit the model using GridSearchCV
    clf = RandomForestClassifier(random_state=0)
    clf = hlp.fit_or_load_model(
        clf, param_grid, X_train, y_train, visualize=False, save=True)

    # Predict on test set
    logger.info('Predicting on test set...')
    y_pred = clf.predict(X_test)
    hlp.write_submission(y_pred, 'random_forest_rfe.csv')
    _ = hlp.summarize_results(y_pred, 'random_forest_summary_rfe.csv')
