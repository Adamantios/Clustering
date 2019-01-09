import time

from sklearn.preprocessing import MinMaxScaler

import helpers
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import SpectralEmbedding, Isomap, LocallyLinearEmbedding, TSNE
from sklearn import metrics
from definitions import SAVE_PRED_RESULTS, PLOTTING_MODE
from typing import Tuple

# Create a logger.
logger = helpers.Logger(folder='logs', filename='genes')

# If plots are enabled, create a plotter.
if PLOTTING_MODE != 'none':
    plotter = helpers.Plotter(folder='plots/genes', mode=PLOTTING_MODE)


def get_x_y() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Gets x and y train and test pairs. """
    logger.log('Loading Dataset...')
    x_train, y_train = helpers.datasets.load_genes()
    logger.log(str(len(y_train)) + ' train data loaded')

    x_test, y_test = None, None
    # x_test, y_test = helpers.datasets.load_genes(train=False)
    # logger.log(str(len(y_test)) + ' test data loaded')

    return x_train, y_train, x_test, y_test


def preprocess(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepocesses data.

    :return: Preprocessed x and y.
    """
    logger.log('Prepocessing...')

    # Scale data.
    logger.log('\tScaling data with params:')
    scaler = MinMaxScaler()
    logger.log('\t{}'.format(scaler.get_params()))
    x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # Apply spectral embedding.
    logger.log('\tApplying Spectral Embedding with params:')
    embedding = LocallyLinearEmbedding(n_neighbors=150, n_jobs=-1)
    embedding_params = embedding.get_params()
    logger.log('\t' + str(embedding_params))
    x_train = embedding.fit_transform(x_train)

    # if PLOTTING_MODE != 'none':
    #     plotter.subfolder = 'graphs/t-SNE'
    #     plotter.filename = 'embedding'
    #     plotter.xlabel = 'first feature'
    #     plotter.ylabel = 'second feature'
    #     plotter.title = 't-SNE'
    #     plotter.scatter(x_train, y_train, class_labels=helpers.datasets.get_gene_name)

    return x_train, x_test


def cluster(x: np.ndarray) -> np.ndarray:
    """
    Fits a clustering model.

    :param x: the x train values.
    :return: the clustering labels.
    """
    logger.log('Creating model...')
    clustering = SpectralClustering(affinity='nearest_neighbors', n_clusters=5, n_neighbors=5, random_state=0,
                                    n_jobs=-1)
    clustering_params = clustering.get_params()
    logger.log('Applying Spectral Clustering with params: \n{}'.format(clustering_params))

    logger.log('Fitting...')
    start_time = time.perf_counter()
    clustering.fit(x)
    end_time = time.perf_counter()
    logger.log('Model has been fit in {:.3} seconds.'.format(end_time - start_time))

    if PLOTTING_MODE != 'none':
        plotter.subfolder = 'graphs/Spectral Clustering'
        plotter.filename = 'after_IsoMAP_c={}-n={}'.format(clustering_params['n_clusters'],
                                                           clustering_params['n_neighbors'])
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'Spectral Clustering after IsoMAP\nClusters: {}, Neighbors: {}' \
            .format(clustering_params['n_clusters'], clustering_params['n_neighbors'])
        plotter.scatter(x, clustering.labels_, clustering=True, class_labels=helpers.datasets.get_gene_name)

    return clustering.labels_


def show_prediction_info(x: np.ndarray, y_true: np.ndarray, y_predicted: np.ndarray, folder: str = 'results',
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


def display_classification_results(x_test: np.ndarray, y_test: np.ndarray, y_predicted: np.ndarray) -> None:
    """
    Randomly plots some correctly classified eegs and some misclassified eegs.
    :param x_test: the test eeg data.
    :param y_test: the eeg test labels.
    :param y_predicted: the predicted labels.
    """
    logger.log('Plotting some random correctly classified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = np.where(y_test == y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.filename = 'correct'
    plotter.subfolder = 'eegs'
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4)

    logger.log('Plotting some random misclassified EEGs.')
    # Get indexes of misclassified digits.
    eegs_indexes = np.where(y_test != y_predicted)[0]
    # Plot some random misclassified digits.
    plotter.filename = 'misclassified'
    plotter.plot_classified_eegs(x_test[eegs_indexes, :], y_predicted[eegs_indexes], y_test[eegs_indexes], num=4)


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # Preprocess data.
    x_train, x_test = preprocess(x_train, y_train, x_test)

    # Apply clustering.
    y_predicted = cluster(x_train)

    # Show prediction information.
    show_prediction_info(x_train, y_train, y_predicted)

    # Show some of the classification results.
    # if PLOTTING_MODE != 'none':
    #     display_classification_results(y, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
