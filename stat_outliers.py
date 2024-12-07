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
        symmetric = np.array(
            [7, 8, 9, 11, 12, 15, 17, 18, 19, 21, 27, 28, 29, 31])
        # multimodal = np.array([2, 3, 4, 5, 6, 10, 13, 14, 16, 20, 22, 23, 24, 25, 26, 30, 32])

        # Assign a distribution to each sensor
        self.sensor_distributions = {
            id: 'symmetric' if id in symmetric else 'multimodal' for id in range(2, 33)
        }

    def handle_outliers(self, data: pd.DataFrame, sensor_id: int) -> pd.DataFrame:
        assert sensor_id in self.sensor_distributions, 'Sensor ID not found'
        assert not data.isna().any().any(), 'Data still contains NaNs'

        distribution = self.sensor_distributions[sensor_id]
        if distribution == 'symmetric':
            return self._handle_symmetric(data)
        elif distribution == 'multimodal':
            return self._handle_multimodal(data)
        else:
            raise ValueError('Invalid distribution type')

    def _handle_symmetric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use Tukey's fences to detect and handle outliers"""
        TUKEY_FACTOR = 1.5
        for i in range(data.shape[0]):
            row = data.iloc[i, :]
            quantiles = row.quantile([0.25, 0.75])
            q1 = quantiles.loc[0.25]
            q3 = quantiles.loc[0.75]
            iqr = q3 - q1
            lower_fence = q1 - TUKEY_FACTOR * iqr
            upper_fence = q3 + TUKEY_FACTOR * iqr
            data.iloc[i, :] = row.clip(lower_fence, upper_fence)
        return data

    def _handle_multimodal(self, data: pd.DataFrame, window_size: int = 5, n_sigma: float = 3.0) -> pd.DataFrame:
        """Hampel filter"""
        filtered_df = data.apply(lambda x: hampel(
            x, window_size=window_size, n_sigma=n_sigma).filtered_data, axis=1)
        return filtered_df


if __name__ == '__main__':
    # Example usage
    handler = StatisticalOutlierHandler()

    # Symmetric data (normal in this case)
    mean = 0
    std_dev = 1
    data = np.random.normal(mean, std_dev, (100, 32))
    df = pd.DataFrame(data)
    result = handler._handle_symmetric(df)
    # print(result)

    # (potentially) Multimodal data
    data = np.random.randn(100, 32)
    df = pd.DataFrame(data)
    result = handler._handle_multimodal(df)
    # print(result)
