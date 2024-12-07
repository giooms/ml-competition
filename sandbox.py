import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

def _interpolate(data: pd.DataFrame) -> pd.DataFrame:
    """Linear interpolation for missing values in each row independently."""
    processed = data.copy()
    for i in range(processed.shape[0]):
        row = processed.iloc[i]
        mask = ~row.isna()
        if mask.any():
            f = interp1d(row.index[mask], row[mask], bounds_error=False, fill_value="extrapolate")
            missing_indices = row.index[~mask]
            replaced_values = f(missing_indices)
            print(f"Row {i}, Indices {missing_indices}, Replaced Values {replaced_values}")
            processed.iloc[i, ~mask] = replaced_values
    return processed

np.random.seed(42)
dataset = np.random.rand(100, 30)
# print(dataset)
df = pd.DataFrame(dataset)
# print(df)
df.iloc[0, 0] = np.nan
df.iloc[1, 1] = np.nan
df.iloc[2, 2] = np.nan
df = _interpolate(df)
print(df)
