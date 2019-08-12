"""Adopted from Udacity ML Nanodegree class"""
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib") # noqa

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # noqa
###########################################

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

# Color map
CMAP: list = [
    "red", "blue", "gray", "green",
    "orange", "purple", "pink", "maroon",
    "cyan", "magenta", "navy", "black",
    "olive", "yellow"]


def pca_results(good_data: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    """
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    """

    # Dimension indexing
    dimensions = [
        'Dim {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(
        np.round(pca.components_, 4), columns=list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(24, 8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind='bar')
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i - 0.40, ax.get_ylim()[1] + 0.05, "Variance\n %.4f" % (ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def pc_variance(pca_results: pd.DataFrame) -> pd.DataFrame:
    """Show PCA components variance with original features"""
    _pca_results = pca_results[['Variance']]
    _pca_results.insert(
        loc=1,
        column='Cumulative Variance',
        value=pca_results['Variance'].cumsum())
    plot_data = (100 * _pca_results).round(0)

    plt.figure(figsize=(12, 5))
    plt.bar(
        plot_data.index, plot_data["Cumulative Variance"],
        label='Men means', color='grey')
    plt.bar(plot_data.index, plot_data["Variance"], color='red')
    plt.plot(pca_results.index.tolist(), plot_data["Variance"], color="yellow")
    plt.ylabel('Variance')
    plt.xlabel('Principal Components')
    plt.suptitle("Percentage of variance (information) for by each PC")


def cluster_results(reduced_data, preds, pca_samples, centers=[]):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and selected sample data
    '''
    plot_data = reduced_data.copy()
    plot_data.insert(0, "Cluster", preds)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2',
                     color=CMAP[i], label='Cluster %i' % (i), s=30)

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black',
                   alpha=1, linewidth=2, marker='o', s=200)
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100)

    # Plot transformed sample points
    ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1],
               s=150, linewidth=4, color='black', marker='x')

    # Set plot title
    ax.set_title(
        "Cluster Learning on PCA-Reduced Data "
        "- Centroids Marked by Number\nTransformed "
        "Sample Data Marked by Black Cross")


def biplot(
        good_data: pd.DataFrame,
        reduced_data: pd.DataFrame,
        pca: PCA):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.

    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)

    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize=(14, 8))
    # scatterplot of the reduced data
    ax.scatter(
        x=reduced_data.loc[:, 'Dimension 1'],
        y=reduced_data.loc[:, 'Dimension 2'],
        facecolors='b',
        edgecolors='b',
        s=70,
        alpha=0.5)

    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
                 head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0] * text_pos, v[1] * text_pos, good_data.columns[i],
                color='black',
                ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16)
    return ax


def compare_cluster_means(
        good_data_with_label: pd.DataFrame):
    """Compare the clusters"""
    scaler = MinMaxScaler()
    cluster_analysis_df = good_data_with_label\
        .groupby("Cluster")\
        .mean()\
        .reset_index()\
        .drop(columns=["Cluster", 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS'])
    cluster_analysis_df = pd.DataFrame(
        scaler.fit_transform(cluster_analysis_df),
        columns=cluster_analysis_df.columns) * 100
    ax = cluster_analysis_df.plot.bar(figsize=(12, 5), legend=True, color=CMAP)
    fontP = FontProperties()
    fontP.set_size('x-small')
    ax.legend(bbox_to_anchor=(0.9, 0.5), prop=fontP)
    return ax

def kmeans_cluster_analysis(data: pd.DataFrame, range_n_clusters: list):
    """Visualize clusters and their scores

    :param data: Data for analysis
    :param range_n_clusters: List of clusters for analisys
    """  
    X = data.copy()
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 4)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.", fontsize=8)
        ax1.set_xlabel("The silhouette coefficient values", fontsize=8)
        ax1.set_ylabel("Cluster label", fontsize=8)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X.loc[:, 'Dimension 1'], X.loc[:, 'Dimension 2'], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.", fontsize=8)
        ax2.set_xlabel("Feature space for the 1st feature", fontsize=8)
        ax2.set_ylabel("Feature space for the 2nd feature", fontsize=8)

        plt.suptitle(("Silhouette analysis for K-Means clustering on sample data "
                      "with n_clusters = {}".format(n_clusters)),
                     fontsize=10, fontweight='bold')
