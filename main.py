import argparse
import logging
import helpers as hlp


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Gradient Boosting GPU with scenarios.')
    parser.add_argument('--scenario', type=str, required=True, choices=['A','B','C','D','E'], help='Scenario to run.')
    parser.add_argument('--fold', type=int, default=None, help='Fold number for LOSO. If not provided, no LOSO.')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data directory')
    parser.add_argument('--method', type=str, default='spline', help='Preprocessing method')
    parser.add_argument('--n_features', type=int, default=50, help='Number of features for RFE if scenario B')
    parser.add_argument('--latent_dim', type=int, default=50, help='Components for PCA or AE latent dimension.')
    parser.add_argument('--model_type', type=str, default='xgb', help='Model type (xgb or rf).')

    args = parser.parse_args()

    hlp.run_scenario(
        data_path=args.data_path,
        method=args.method,
        model_type=args.model_type,
        scenario=args.scenario,
        n_features=args.n_features,
        latent_dim=args.latent_dim,
        fold=args.fold
    )
