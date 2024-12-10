import helpers as hlp
import logging
from sklearn.ensemble import GradientBoostingClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('Gradient Boosting Classifier')

    # Load the data
    X_train, X_test, y_train = hlp.load_data(data_path='./')
    # print(X_train.head())
    # print(X_test.head())
    # print(y_train.head())

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 5, 8],
        'subsample': [0.7, 1.0],
        'max_features': ['auto', 'sqrt']
    }

    # Fit the model using GridSearchCV
    clf = GradientBoostingClassifier(random_state=0)
    clf = hlp.fit_or_load_model(
        clf, param_grid, X_train, y_train, visualize=False, save=True)

    # Predict on test set
    logger.info('Predicting on test set...')
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
