#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def load_data(data_path):
    """
    Loads training and testing data from the specified directory.
    Args:
        data_path (str): The path to the directory containing the data files.
    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): Training data with shape (N_TIME_SERIES, len(FEATURES) * 512).
            - y_train (numpy.ndarray): Training labels.
            - X_test (numpy.ndarray): Testing data with shape (N_TIME_SERIES, len(FEATURES) * 512).
    The function expects the following directory structure:
    ```
    data_path/
        LS/
            LS_sensor_2.txt
            LS_sensor_3.txt
            ...
            LS_sensor_32.txt
            activity_Id.txt
        TS/
            TS_sensor_2.txt
            TS_sensor_3.txt
            ...
            TS_sensor_32.txt
    ```
    Each sensor file should contain data with shape (N_TIME_SERIES, 512).
    """
    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')

    # Creating two arrays, X_ train and test, with shape (N_TIME_SERIES, len(FEATURES) * 512)
    # Each array has N_TIME_SERIES rows, and each row is a flattened 1D representation of the feature space
    # The size of each row is len(FEATURES) * 512
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    # Load training/testing data for each feature and assign it to the corresponding slice of X_train/X_test
    for f in FEATURES:
        # Training data
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        # Testing data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data

    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test


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

    print(f'Submission saved to {submission_path}.')

if __name__ == '__main__':
    X_train, y_train, X_test = load_data(data_path='./')

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    y_test = clf.predict(X_test)

    write_submission(y_test)
