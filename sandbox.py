import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
np.random.seed(0)

def analyze_missing_pattern(df_row):
    """
    Analyzes the pattern of missing values in a row.
    Returns:
        - Number of missing values
        - Length of longest continuous sequence of missing values
        - Number of separate missing sequences
    """
    missing_mask = df_row.isna()
    total_missing = missing_mask.sum()

    # If no missing values, return early
    if total_missing == 0:
        return 0, 0, 0

    # Convert to numeric (True -> 1, False -> 0)
    numeric_mask = missing_mask.astype(int)

    # Find sequences of missing values
    # When diff() == 1, it's the start of a sequence
    # When diff() == -1, it's the end of a sequence
    diff_mask = numeric_mask.diff()
    sequence_starts = diff_mask[diff_mask == 1].index
    sequence_ends = diff_mask[diff_mask == -1].index

    # Handle edge cases (missing values at start/end)
    if numeric_mask.iloc[0] == 1:
        sequence_starts = pd.Index(
            [numeric_mask.index[0]]).append(sequence_starts)
    if numeric_mask.iloc[-1] == 1:
        sequence_ends = sequence_ends.append(
            pd.Index([numeric_mask.index[-1]]))

    # Calculate lengths of sequences
    sequence_lengths = [end - start + 1 for start,
                        end in zip(sequence_starts, sequence_ends)]
    max_sequence = max(sequence_lengths) if sequence_lengths else 0
    num_sequences = len(sequence_lengths)

    return total_missing, max_sequence, num_sequences


def analyze_df_missing_patterns(df):
    results = []
    for idx, row in df.iterrows():
        total_missing, max_seq, num_seq = analyze_missing_pattern(row)
        results.append({
            'total_missing': total_missing,
            'missing_percentage': (total_missing / len(row)) * 100,
            'longest_sequence': max_seq,
            'num_sequences': num_seq,
            'avg_sequence_length': total_missing / num_seq if num_seq > 0 else 0
        })
    return pd.DataFrame(results, index=df.index)


# Usage example:
# df is your DataFrame where each row is a 512-point time series
MISSING_VALUE = -999999.99
series_to_drop = pd.Series([False] * 3500)
dataset = {}
for i in range(2,33):
    print(f"Processing sensor {i}")
    df = pd.read_csv(f'LS/LS_sensor_{i}.txt', delimiter=' ', header=None)
    df.replace(MISSING_VALUE, np.nan, inplace=True)
    dataset[i] = df
    patterns_df = analyze_df_missing_patterns(df)
    series_to_drop = series_to_drop | (
                patterns_df['missing_percentage'] > 0.25)

    series_to_drop = series_to_drop | (
                patterns_df['num_sequences'] > 50)

for sensor_id, df in dataset.items():
    print(f"sensor {sensor_id}")
    print(len(df))
    processed_data = df[~series_to_drop]
    print(len(processed_data))
    # check if processed_data contains nans
    print(processed_data.isna().values.any())
    print("")

# patterns_df = analyze_df_missing_patterns(df)
# print((patterns_df['total_missing'] > 0).sum())

# try:
#     # df = df.interpolate(method='zero', axis=0)
#     df.interpolate(method='linear', axis=1, limit_direction='both', inplace=True)
#     # df.interpolate(method='polynomial', order=2, axis=1, limit_direction='both', inplace=True)
#     print("success")
# except ValueError as e:
#     print(f"Interpolation error: {e}")

# patterns_df = analyze_df_missing_patterns(df)
# print((patterns_df['total_missing'] > 0).sum())

# You can then filter based on your criteria, for example:
# scattered_missing = patterns_df[
#     (patterns_df['missing_percentage'] <= 25) &  # Less than 25% missing
#     (patterns_df['longest_sequence'] <= 10) &    # No long sequences
#     (patterns_df['num_sequences'] >= 3)          # Multiple scattered sequences
# ]

# continuous_missing = patterns_df[
#     (patterns_df['longest_sequence'] >= 50)      # Long continuous sequences
# ]

# # Drop rows with long continuous missing sequences
# df_cleaned = df.drop(continuous_missing.index)
