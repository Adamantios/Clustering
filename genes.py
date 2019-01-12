import time
import helpers
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import metrics
from definitions import SAVE_PRED_RESULTS, PLOTTING_MODE
from typing import Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Create a logger.
logger = helpers.Logger(folder='logs', filename='genes')

# If plots are enabled, create a plotter.
if PLOTTING_MODE != 'none':
    plotter = helpers.Plotter(folder='plots/genes', mode=PLOTTING_MODE)


def get_x_y() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x and y train and test pairs.

    :return: tuple with numpy arrays containing x_train, y_train, x_test and y_test.
    """
    logger.log('Loading Dataset...')
    x_train, y_train = helpers.datasets.load_genes()
    logger.log(str(len(y_train)) + ' train data loaded')

    x_test, y_test = helpers.datasets.load_genes(train=False)
    logger.log(str(len(y_test)) + ' test data loaded')

    return x_train, y_train, x_test, y_test


def preprocess(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepocesses data.

    :param x_train: the training data.
    :param y_train: the training labels.
    :param x_test: the test data.
    :return: Preprocessed x_train and x_test.
    """
    logger.log('Prepocessing...')

    # Scale data.
    logger.log('\tScaling data with params:')
    scaler = MinMaxScaler()
    logger.log('\t{}'.format(scaler.get_params()))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply LLE.
    logger.log('\tApplying LLE with params:')
    embedding = LocallyLinearEmbedding(n_neighbors=100, n_jobs=-1, random_state=0)
    embedding_params = embedding.get_params()
    logger.log('\t' + str(embedding_params))
    x_train = embedding.fit_transform(x_train)
    x_test = embedding.transform(x_test)

    # Plot the graph embedding result.
    if PLOTTING_MODE != 'none':
        plotter.subfolder = 'graphs/LLE'
        plotter.filename = 'embedding'
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'LLE'
        plotter.scatter(x_train, y_train, class_labels=helpers.datasets.get_gene_name)

    return x_train, x_test


def cluster(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fits a clustering model.

    :param x: the x train values.
    :param y: the label values.
    :return: the clustering labels.
    """
    logger.log('Creating model...')
    clustering = SpectralClustering(affinity='nearest_neighbors', n_clusters=5, n_neighbors=250, random_state=0,
                                    n_jobs=-1)
    clustering_params = clustering.get_params()
    logger.log('Applying Spectral Clustering with params: \n{}'.format(clustering_params))

    logger.log('Fitting...')
    start_time = time.perf_counter()
    clustering.fit(x)
    end_time = time.perf_counter()
    logger.log('Model has been fit in {:.3} seconds.'.format(end_time - start_time))

    if PLOTTING_MODE != 'none':
        # Plot resulting clusters.
        plotter.subfolder = 'graphs/Spectral Clustering/clusters'
        plotter.filename = 'after_LLE_c={}-n={}'.format(clustering_params['n_clusters'],
                                                        clustering_params['n_neighbors'])
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'Spectral Clustering after LLE\nClusters: {}, Neighbors: {}' \
            .format(clustering_params['n_clusters'], clustering_params['n_neighbors'])
        plotter.scatter(x, clustering.labels_, clustering=True)

        # Plot classes compared to clusters.
        plotter.subfolder = 'graphs/Spectral Clustering/classes'
        plotter.scatter(x, y, clusters=clustering.labels_, class_labels=helpers.datasets.get_gene_name)

    return clustering.labels_


def show_clustering_info(x: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray, folder: str = 'results',
                         filename: str = 'genes', extension: str = 'xlsx', sheet_name: str = 'results') -> None:
    """
    Shows information about the predicted data and saves them to an excel file.

    :param x: the x data.
    :param y_true: the known label values.
    :param y_predicted: the predicted label values.
    :param folder: the folder to save the results excel file.
    :param filename: the name of the excel file.
    :param extension: the file's extension.
    :param sheet_name: the excel's sheet name.
    """
    hcv = metrics.homogeneity_completeness_v_measure(y_true, y_predicted)

    # Create results dictionary.
    results = {'Adjusted Random Index': [metrics.adjusted_rand_score(y_true, y_predicted)],
               'Homogeneity': [hcv[0]],
               'Completeness': [hcv[1]],
               'V Measure': [hcv[2]],
               'Silhouette Coefficient': [metrics.silhouette_score(x, y_predicted)]}

    # Log results.
    logger.log('Model\'s Results:')
    for key, values in results.items():
        for value in values:
            logger.log('{text}: {number:.{points}g}'.format(text=key, number=value, points=4))

    # Create excel if save is True.
    if SAVE_PRED_RESULTS:
        helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def assign_to_clusters(x_train: np.ndarray, clusters: np.ndarray, x_test: np.ndarray, y_true: np.ndarray) -> None:
    """
    Assigns new data to existing clusters, using nearest neighbors classification.

    :param x_train: the data which have been clustered.
    :param clusters: the clusters.
    :param x_test: the data to be assigned to clusters.
    :param y_true: the data class labels.
    """
    logger.log('Creating Nearest Neighbors classifier with params:')
    clf = KNeighborsClassifier()
    clf_params = clf.get_params()
    logger.log(clf_params)
    clf.fit(x_train, clusters)
    y_pred = clf.predict(x_test)

    if PLOTTING_MODE != 'none':
        # Plot data vs clusters.
        plotter.subfolder = 'classification'
        plotter.filename = 'data_vs_clusters-k={}'.format(clf_params['n_neighbors'])
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'Classified data vs Clusters'
        plotter.scatter_classified_comparison(x_train, clusters, x_test, y_true, y_pred,
                                              'Test data vs clusters',
                                              'Test data assigned to clusters\nk={}'.format(clf_params['n_neighbors']),
                                              helpers.datasets.get_gene_name)


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # Preprocess data.
    x_train, x_test = preprocess(x_train, y_train, x_test)

    # Apply clustering.
    y_predicted = cluster(x_train, y_train)

    # Show prediction information.
    show_clustering_info(x_train, y_train, y_predicted)

    # Assign test data to clusters.
    assign_to_clusters(x_train, y_predicted, x_test, y_test)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
