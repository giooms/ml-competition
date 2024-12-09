"""
Step 1. Start with Gradient Boosting (e.g., XGBoost, LightGBM):
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
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
