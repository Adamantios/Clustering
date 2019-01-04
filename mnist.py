import time
import helpers
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn import metrics
from definitions import SAVE_PRED_RESULTS, PLOTTING_MODE
from typing import Tuple

# Create a logger.
from helpers.preprocessing import cut_images

logger = helpers.Logger(folder='logs', filename='seizure_detection')

# If plots are enabled, create a plotter.
if PLOTTING_MODE != 'none':
    plotter = helpers.Plotter(folder='plots', mode=PLOTTING_MODE)


def get_x_y() -> Tuple[np.ndarray, np.ndarray]:
    """ Gets x and y train and test pairs. """
    logger.log('Loading Dataset...')
    x, y = helpers.datasets.load_digits()
    logger.log(str(len(y)) + ' data loaded')

    return x, y


def preprocess(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Prepocesses data.

    :param x: the data values.
    :param y: the labels values.
    :return: Preprocessed x and y.
    """
    logger.log('Prepocessing...')

    # Scale data.
    logger.log('\tScaling data...')
    x /= 255

    x = cut_images(x)

    # Apply spectral embedding.
    logger.log('\tApplying Spectral Embedding with params:')
    embedding = SpectralEmbedding()
    logger.log('\t' + str(embedding.get_params()))
    x = embedding.fit_transform(x)

    if PLOTTING_MODE != 'none':
        plotter.subfolder = 'graphs'
        plotter.filename = 'embedding'
        plotter.xlabel = 'first feature'
        plotter.ylabel = 'second feature'
        plotter.title = 'Graph Embedding'
        plotter.scatter(x, y)

    return x


def cluster(x: np.ndarray) -> np.ndarray:
    """
    Fits a clustering model.

    :param x: the x train values.
    :return: the clustering labels.
    """
    logger.log('Creating model...')
    clustering = SpectralClustering(n_clusters=2,
                                    assign_labels="discretize",
                                    random_state=0)
    logger.log('Applying Spectral Clustering with params: \n{}'.format(clustering.get_params()))

    logger.log('Fitting...')
    start_time = time.perf_counter()
    clustering.fit(x)
    end_time = time.perf_counter()
    logger.log('Model has been fit in {:.3} seconds.'.format(end_time - start_time))

    return clustering.labels_


def show_prediction_info(y_true: np.ndarray, y_predicted: np.ndarray, folder: str = 'results', filename: str = 'mnist',
                         extension: str = 'xlsx', sheet_name: str = 'results') -> None:
    """
    Shows information about the predicted data and saves them to an excel file.

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
               'V Measure': [hcv[2]]}

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
    x, y = get_x_y()

    # Preprocess data.
    x = preprocess(x, y)

    # Apply clustering.
    y_predicted = cluster(x)

    # Show prediction information.
    show_prediction_info(y, y_predicted)

    # Show some of the classification results.
    # if PLOTTING_MODE != 'none':
    #     display_classification_results(y, y_predicted)

    # Close the logger.
    logger.close()


if __name__ == '__main__':
    main()
