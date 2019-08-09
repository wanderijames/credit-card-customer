"""Detect outliers"""
import numpy as np


def detect(data) -> (int, dict):
    """Use IQR with Tukey's Method for identfying outliers

    :param data: pandas DataFrame
    """
    outliers_dist = {}
    outliers_count = {}
    # For each feature find the data points with extreme high or low values
    for feature in data.keys():
        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(data[feature], 25)
        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(data[feature], 75)
        # Use the interquartile range to calculate an
        # outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)
        # Get the outliers indices distrubution amongst the features
        feature_outliers = data[
            ~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]
        indices = feature_outliers.index.tolist()
        if not indices:
            continue
        outliers_count[feature] = len(indices)
        for i in indices:
            features = outliers_dist.get(i, [])
            if feature not in features:
                features.append(feature)
            outliers_dist[i] = features
    outliers = [pair[0] for pair in outliers_dist.items() if len(pair[1]) > 2]
    return outliers, outliers_count
