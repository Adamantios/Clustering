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
logger = helpers.Logger(folder='logs', filename='genes_test')

# If plots are enabled, create a plotter.
if PLOTTING_MODE != 'none':
    plotter = helpers.Plotter(folder='plots/genes_test', mode=PLOTTING_MODE)


def get_x_y() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Gets x and y train and test pairs. """
    logger.log('Loading Dataset...')
    x_train, y_train = helpers.datasets.load_genes()
    logger.log(str(len(y_train)) + ' train data loaded')

    x_test, y_test = None, None
    # x_test, y_test = helpers.datasets.load_genes(train=False)
    # logger.log(str(len(y_test)) + ' test data loaded')

    return x_train, y_train, x_test, y_test


def embed(embedding, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepocesses data.

    :return: Preprocessed x and y.
    """
    # Apply spectral embedding.
    logger.log('\tApplying Spectral Embedding with params:')
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


def cluster(clustering, x: np.ndarray, embed_name: str) -> np.ndarray:
    """
    Fits a clustering model.

    :param x: the x train values.
    :return: the clustering labels.
    """
    logger.log('Creating model...')
    clustering_params = clustering.get_params()
    logger.log('Applying Spectral Clustering with params: \n{}'.format(clustering_params))

    logger.log('Fitting...')
    start_time = time.perf_counter()
    clustering.fit(x)
    end_time = time.perf_counter()
    logger.log('Model has been fit in {:.3} seconds.'.format(end_time - start_time))

    if PLOTTING_MODE != 'none':
        plotter.subfolder = 'tests/graphs/Spectral Clustering'
        plotter.filename = 'after_{}_c={}-n={}'.format(embed_name, clustering_params['n_clusters'],
                                                       clustering_params['n_neighbors'])
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'Spectral Clustering after {}\nClusters: {}, Neighbors: {}' \
            .format(embed_name, clustering_params['n_clusters'], clustering_params['n_neighbors'])
        plotter.scatter(x, clustering.labels_, clustering=True, class_labels=helpers.datasets.get_gene_name)

    return clustering.labels_


def show_prediction_info(x: np.ndarray, y_true: np.ndarray, y_predicted: dict, folder: str = 'results',
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
    hcv0 = metrics.homogeneity_completeness_v_measure(y_true, y_predicted['IsoMAP'])
    hcv1 = metrics.homogeneity_completeness_v_measure(y_true, y_predicted['LLE'])
    hcv2 = metrics.homogeneity_completeness_v_measure(y_true, y_predicted['SE'])
    hcv3 = metrics.homogeneity_completeness_v_measure(y_true, y_predicted['TSNE'])

    # Create results dictionary.
    results = {'Adjusted Random Index': [metrics.adjusted_rand_score(y_true, y_predicted['IsoMAP']),
                                         metrics.adjusted_rand_score(y_true, y_predicted['LLE']),
                                         metrics.adjusted_rand_score(y_true, y_predicted['SE']),
                                         metrics.adjusted_rand_score(y_true, y_predicted['TSNE'])],
               'Homogeneity': [hcv0[0], hcv1[0], hcv2[0], hcv3[0]],
               'Completeness': [hcv0[1], hcv1[1], hcv2[1], hcv3[1]],
               'V Measure': [hcv0[2], hcv1[2], hcv2[2], hcv3[2]],
               'Silhouette Coefficient': [metrics.silhouette_score(x, y_predicted['IsoMAP']),
                                          metrics.silhouette_score(x, y_predicted['LLE']),
                                          metrics.silhouette_score(x, y_predicted['SE']),
                                          metrics.silhouette_score(x, y_predicted['TSNE'])]}

    # Log results.
    logger.log('Model\'s Results:')
    for key, values in results.items():
        for value in values:
            logger.log('{text}: {number:.{points}g}'.format(text=key, number=value, points=4))

    # Create excel if save is True.
    helpers.utils.create_excel(results, folder, filename, extension, sheet_name)


def run_embedding_test(x_train, y_train, x_test):
    all_clusters, all_neighbors = [3, 5, 7], [5, 20, 100, 250]
    y_predicted: dict = {}

    embedding_model = Isomap(n_neighbors=479, n_jobs=-1)
    isomap_x_train, isomap_x_test = embed(embedding_model, x_train, y_train, x_test)

    embedding_model = LocallyLinearEmbedding(n_neighbors=100, n_jobs=-1, random_state=0)
    lle_x_train, lle_x_test = embed(embedding_model, x_train, y_train, x_test)

    embedding_model = SpectralEmbedding(affinity='nearest_neighbors', n_neighbors=100, n_jobs=-1, random_state=0)
    se_x_train, se_x_test = embed(embedding_model, x_train, y_train, x_test)

    embedding_model = TSNE()
    tsne_x_train, tsne_x_test = embed(embedding_model, x_train, y_train, x_test)

    for clusters in all_clusters:
        for neighbors in all_neighbors:
            clustering_model = SpectralClustering(affinity='nearest_neighbors', n_clusters=clusters,
                                                  n_neighbors=neighbors,
                                                  random_state=0, n_jobs=-1)
            y_predicted['IsoMAP'] = cluster(clustering_model, isomap_x_train, 'IsoMAP')
            y_predicted['LLE'] = cluster(clustering_model, lle_x_train, 'LLE')
            y_predicted['SE'] = cluster(clustering_model, se_x_train, 'SE')
            y_predicted['TSNE'] = cluster(clustering_model, tsne_x_train, 'TSNE')

            # Show prediction information.
            show_prediction_info(x_train, y_train, y_predicted,
                                 folder='results/{}clusters'.format(clusters),
                                 filename='clusters={}-k={}'.format(clusters, neighbors))


def main():
    # Get x and y pairs.
    x_train, y_train, x_test, y_test = get_x_y()

    # Scale data.
    logger.log('\tScaling data with params:')
    scaler = MinMaxScaler()
    logger.log('\t{}'.format(scaler.get_params()))
    x_train = scaler.fit_transform(x_train)

    run_embedding_test(x_train, y_train, x_train)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
