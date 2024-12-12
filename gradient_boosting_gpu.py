import argparse
import logging
import helpers as hlp
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Gradient Boosting with GPU and optional RFE.')
    parser.add_argument('--use_rfe', action='store_true', help='Use RFE for feature selection')
    parser.add_argument('--n_features', type=int, default=100, help='Number of features to select with RFE')
    parser.add_argument('--data_path', type=str, default='./', help='Path to data directory')
    parser.add_argument('--method', type=str, default='spline', help='Preprocessing method')
    parser.add_argument('--random_state', type=int, default=0, help='Random state for reproducibility')
    args = parser.parse_args()

    logger.info('Gradient Boosting Classifier with GPU and optional RFE')

    # Determine file name suffix based on RFE usage
    if args.use_rfe:
        file_suffix = f"_rfe{args.n_features}"
    else:
        file_suffix = ""

    # Load the data with or without RFE
    X_train, X_test, y_train = hlp.load_data(
        data_path=args.data_path,
        method=args.method,
        use_rfe=args.use_rfe,
        n_features_to_select=args.n_features
    )

    # Adjust labels if needed (XGB expects 0-based for multiclass)
    y_train = y_train - 1

    # Hyperparameter grid for GridSearchCV (XGBoost parameters)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'gamma': [0, 0.1, 0.3]
    }

    # Use GPU-accelerated XGBoost
    clf = XGBClassifier(random_state=args.random_state, tree_method='gpu_hist', predictor='gpu_predictor')
    clf = hlp.fit_or_load_model(clf, param_grid, X_train, y_train, visualize=False, save=True)

    # Predict on test set
    logger.info('Predicting on test set...')
    y_pred = clf.predict(X_test)
    # Convert predictions back to 1-based
    y_pred = y_pred + 1

    submission_filename = f"gradient_boosting_gpu{file_suffix}.csv"
    summary_filename = f"gradient_boosting_gpu_summary{file_suffix}.csv"

    hlp.write_submission(y_pred, submission_filename)
    _ = hlp.summarize_results(y_pred, summary_filename)