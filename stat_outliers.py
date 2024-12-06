import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from hampel import hampel
from scipy.interpolate import UnivariateSpline
"""
        # Remove statistical outliers using IQR method
        # TUKEY_CONSTANT = 1.5
        # for i in range(data.shape[1]):
        #     valid_data = data[:, i][~np.isnan(data[:, i])]
        #     if len(valid_data) > 0:
        #         q1, q3 = np.percentile(valid_data, [25, 75])
        #         iqr = q3 - q1
        #         lower_bound = q1 - TUKEY_CONSTANT * iqr
        #         upper_bound = q3 + TUKEY_CONSTANT * iqr
        #         data[:, i] = np.where((data[:, i] < lower_bound) | (
        #             data[:, i] > upper_bound), np.nan, data[:, i])
 """


class StatisticalOutlierHandler:
    def __init__(self):
        # Assign a distribution to each sensor
        self.sensor_distributions = {}
        symmetric = np.array(
            [7, 8, 9, 11, 12, 15, 17, 18, 19, 21, 27, 28, 29, 31])
        multimodal = np.array(
            [2, 3, 4, 5, 6, 10, 13, 14, 16, 20, 22, 23, 24, 25, 26, 30, 32])
        for id in range(2, 33):
            if id in symmetric:
                self.sensor_distributions[id] = 'symmetric'
            elif id in multimodal:
                self.sensor_distributions[id] = 'multimodal'

    def handle_outliers(self, data: np.ndarray, sensor_id: int) -> np.ndarray:
        assert sensor_id in self.sensor_distributions, 'Sensor ID not found'
        distribution = self.sensor_distributions[sensor_id]
        if distribution == 'symmetric':
            return self._handle_symmetric(data)
        elif distribution == 'multimodal':
            return self._handle_multimodal(data)
        return

    def _handle_symmetric(self, data: np.ndarray) -> np.ndarray:
        """Use Tukey's fences to detect and handle outliers, preserving NaNs"""
        TUKEY_FACTOR = 1.5
        # Flatten the data to compute percentiles for whole dataset
        flattened_data = data.flatten()
        q1, q3 = np.nanpercentile(flattened_data, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - TUKEY_FACTOR * iqr
        upper_fence = q3 + TUKEY_FACTOR * iqr

        # Use np.where to preserve NaNs
        data = np.where(np.isnan(data), data, np.clip(
            data, lower_fence, upper_fence))
        return data

    def _handle_multimodal(self, data: np.ndarray, window_size: int = 5, n_sigma: float = 3.0) -> np.ndarray:
        """Hampel filter"""
        def apply_hampel(x):
            if np.isnan(x).any():
                mask = np.isnan(x)
                x_non_nan = x[~mask]
                x_filtered = hampel(x_non_nan, window_size=window_size, n_sigma=n_sigma).filtered_data
                x[~mask] = x_filtered
                return x
            return hampel(x, window_size=window_size, n_sigma=n_sigma).filtered_data

        return np.apply_along_axis(apply_hampel, axis=1, arr=data)


if __name__ == '__main__':
    # Example usage
    handler = StatisticalOutlierHandler()
    for i in range(2, 3):
        data = np.loadtxt(f'./LS/LS_sensor_{i}.txt')
        result = handler._handle_multimodal(data)
        print(result)
