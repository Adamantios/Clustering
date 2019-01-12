import itertools
from os.path import join
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from helpers.utils import create_folder


class DimensionError(Exception):
    pass


class Plotter:
    def __init__(self, folder: str = 'plots', mode: str = 'show'):
        # Get plotter's mode and check it's value.
        self.mode = mode
        if self.mode != 'show' and self.mode != 'save' and self.mode != 'both':
            raise ValueError('Plotter\'s mode can be \'save\', \'show\' or \'both\'.\nGot {} instead.'
                             .format(self.mode))

        # Create a folder for the plots, if needed.
        if self.mode == 'save' or self.mode == 'both':
            self._folder = create_folder(folder)

        self.subfolder: str = ''
        self.suptitle: str = ''
        self.title: str = ''
        self.filename: str = 'plot'
        self.extension: str = 'png'
        self.xlabel: str = ''
        self.ylabel: str = ''

    def _create_plot_folder(self) -> None:
        """" Create a plot's subfolder. """
        if self.mode == 'save' or self.mode == 'both':
            create_folder(self._folder + '/' + self.subfolder)

    def _save_and_show(self, fig: plt.Figure) -> None:
        """ Save and plot a figure. """
        if self.mode == 'save' or self.mode == 'both':
            filename = self.filename + '.' + self.extension
            self._save_path = join(self._folder, self.subfolder, filename)
            fig.savefig(self._save_path)

        if self.mode == 'show' or self.mode == 'both':
            plt.show()

    def reset_params(self):
        """ Resets the parameters. """
        self.mode = 'show'
        self.subfolder: str = ''
        self.suptitle: str = ''
        self.title: str = ''
        self.filename: str = 'plot'
        self.extension: str = 'png'
        self.xlabel: str = ''
        self.ylabel: str = ''

    @staticmethod
    def _check_dims(x: np.ndarray) -> None:
        """
        Checks if the passed data dimension is 2D.

        :param x: the data to be checked.
        """
        # If one dimensional data have been passed, raise an error.
        if x.ndim == 1:
            raise DimensionError('Expected 2 dimensions.Got 1 instead.')

        # If more than two dimensional data have been passed, raise an error.
        elif x.shape[1] > 2:
            raise DimensionError('Cannot plot more than 2 dimensions.')

    def _prepare_scatter(self, x: np.ndarray, x_test: np.ndarray = None, class_labels: Callable[[int], str] = None):
        """
        Check if possible and prepare to plot.

        :param x: the data to be plot.
        :param x_test: the test data to be plot.
        :param class_labels: an optional function which gets the class labels from their indexes.
        :return: the colors the clusters_colors and the class_labels.
        """
        # If labels getter function has not been passed, use the indexes.
        if class_labels is None:
            def class_labels(index: int): return index

        self._check_dims(x)

        if x_test is not None:
            self._check_dims(x_test)

        self._create_plot_folder()

        # Use a style.
        plt.style.use('seaborn-white')

        # Create colors for the plots.
        colors = itertools.cycle(
            ['#fff100', '#ff8c00', '#e81123', '#ec008c', '#68217a', '#00188f', '#00bcf2', '#00b294', '#009e49',
             '#bad80a'])

        clusters_colors = itertools.cycle(
            ['darkred', 'black', 'rosybrown', 'olivedrab', 'darkcyan', 'orangered', 'purple', 'crimson', 'darkseagreen',
             'tan'])

        return colors, clusters_colors, class_labels

    def scatter(self, x: np.ndarray, y: np.ndarray, class_labels: Callable[[int], str] = None, clustering: bool = False,
                clusters: np.ndarray = None) -> None:
        """
        Plots and saves a scatterplot with the first one, two or three features.

        :param x: the features to plot.
        :param y: the class labels.
        :param class_labels: an optional function which gets the class labels from their indexes.
        :param clustering: whether it is after clustering or not.
        :param clusters: the clustering labels.
        """
        colors, clusters_colors, class_labels = self._prepare_scatter(x, class_labels=class_labels)

        # Create a figure and an ax.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Get the class labels and count each label's instances.
        labels, counts = np.unique(y, return_counts=True)
        # For every class, scatter it's principal components.
        for i, count in zip(labels, counts):
            if clustering:
                label = 'Cluster {}'.format(i)
            else:
                label = '{} class'.format(class_labels(i))

            ax.scatter(x[y == i, 0], x[y == i, 1], alpha=0.5, label=label, color=next(colors))

        if clusters is not None:
            # Get the class labels and count each label's instances.
            labels, counts = np.unique(clusters, return_counts=True)

            for i, count in zip(labels, counts):
                # Get next cluster color.
                color = next(clusters_colors)

                # Draw cluster connections.
                connections = Polygon(x[clusters == i], linewidth=.3, fill=False, joinstyle='bevel',
                                      alpha=.8, color=color)
                ax.add_patch(connections)

        # Set legend, title, x and y labels and clear x and y ticks.
        ax.legend()
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks([])
        ax.set_yticks([])

        self._save_and_show(fig)

    def scatter_classified_comparison(self, x: np.ndarray, clusters: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                                      y_pred: np.ndarray, sub1: str, sub2: str,
                                      class_labels: Callable[[int], str] = None) -> None:
        """
        Plots and saves a scatterplot with the first one, two or three features.

        :param x: the clustered data.
        :param clusters: the clustering labels.
        :param x_test: the data to be classified.
        :param y_test: the class labels of the data to be classified.
        :param y_pred: the predicted clusters of the classified data.
        :param sub1: first plot's title.
        :param sub2: second plot's title.
        :param class_labels: an optional function which gets the class labels from their indexes.
        """
        colors, clusters_colors, class_labels = self._prepare_scatter(x, x_test, class_labels)

        # Create a figure and 2 axes for the plots.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Get the class labels of the data to be classified.
        labels = np.unique(y_test)
        # For every class, scatter its data.
        for i in labels:
            ax.scatter(x_test[y_test == i, 0], x_test[y_test == i, 1], alpha=0.5,
                       label='{} class'.format(class_labels(i)), color=next(colors))

        # Get the predicted and the real clusters.
        labels_pred, cluster_labels = np.unique(y_pred), np.unique(clusters)
        for pred, cluster in zip(labels_pred, cluster_labels):
            # Get cluster color.
            color = next(clusters_colors)
            # Scatter the cluster's data.
            ax2.scatter(x_test[y_pred == pred, 0], x_test[y_pred == pred, 1], alpha=0.5,
                        label='Classified at cluster {}'.format(pred), color=color)

            # Find cluster means.
            mean = x[clusters == cluster].mean(axis=0)
            # Create cluster label annotation string.
            annotation = 'Cluster {}'.format(cluster)
            # Add annotation to the plots.
            ax.annotate(annotation, mean,
                        horizontalalignment='center', verticalalignment='center',
                        size=20, weight='bold', color=color)
            ax2.annotate(annotation, mean,
                         horizontalalignment='center', verticalalignment='center',
                         size=20, weight='bold', color=color)

        # Set suptitle.
        fig.suptitle(self.title)

        # Set legend, title, x and y labels and clear x and y ticks.
        ax.legend()
        ax.set_title(sub1)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks([])
        ax.set_yticks([])

        # Set legend, title, x and y labels and clear x and y ticks.
        ax2.legend()
        ax2.set_title(sub2)
        ax2.set_xlabel(self.xlabel)
        ax2.set_ylabel(self.ylabel)
        ax2.set_xticks([])
        ax2.set_yticks([])

        self._save_and_show(fig)
