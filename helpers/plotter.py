import random
import numpy as np
from os.path import join
from typing import Generator, Tuple, Callable
from matplotlib import pyplot as plt
from helpers.utils import create_folder
from helpers.datasets import get_eeg_name


class TooManyDimensionsError(Exception):
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

    def _plot_eeg(self, eeg: np.ndarray):
        """
        Plots an eeg.

        :param eeg: the eeg to be plotted.
        """
        self._create_plot_folder()

        # Use a style.
        plt.style.use('seaborn-white')

        # Create a subplot.
        fig, ax = plt.subplots(figsize=(9, 4))
        # Create a super title.
        fig.suptitle('{}\n{}'.format(self.suptitle, self.title), fontsize='large')
        # Create the plot.
        ax.plot(eeg)

        # Remove xticks, add xlabel and ylabel.
        ax.set_xticks([])
        ax.set_xlabel("1 second", fontsize='large')
        ax.set_ylabel("EEG Value", fontsize='large')

        self._save_and_show(fig)

    def _plot_digit(self, digit: np.ndarray) -> None:
        """
        Plots and saves an image of a digit.

        :param digit: the digit to be plotted.
        """
        self._create_plot_folder()

        # Change the shape of the image to 2D.
        digit.shape = (28, 28)

        # Create a subplot.
        fig, ax = plt.subplots(figsize=(3, 3.5))
        # Create a super title.
        fig.suptitle('{}\n{}'.format(self.suptitle, self.title), fontsize='large')
        # Create an image from the digit's pixels. The pixels are not rgb so cmap should be gray.
        ax.imshow(digit, cmap='gray')
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        self._save_and_show(fig)

    @staticmethod
    def _random_picker(x: np.ndarray, num: int) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Create a generator which contains a certain number of randomly chosen values from a np array and an index.

        :param x: the np array
        :param num: the number of the random values to be generated.
            If the number is None, bigger than the list or less than zero, randomizes the whole list.
        :return: Generator with a random value and its index.
        """
        # If the number is None, bigger than the list or less than zero, set the number with the lists length.
        if num is None or num > len(x) or num < 0:
            num = len(x)

        # Get num random samples from the list.
        rand_samples = random.sample(range(len(x)), num)

        # For each random sample, yield the sample and its index.
        for sample, i in zip(rand_samples, range(1, len(rand_samples) + 1)):
            yield sample, i

    def plot_classified_eegs(self, x, y_pred, y_true, num=None):
        """
        Plots and saves a certain number of classified eegs.

        :param x: the eegs.
        :param y_pred: the predicted values of the classified eeg.
        :param y_true: the real value of the classified eegs.
        :param num: the number of eegs to be plotted.
        """
        filename = self.filename

        # Plot num of eegs randomly.
        for eeg, i in self._random_picker(x, num):
            self.suptitle = 'Classified as {}'.format(get_eeg_name(y_pred[eeg]))
            self.title = 'Correct condition is {}'.format(get_eeg_name(y_true[eeg]))
            self.filename = '{}{}'.format(filename, str(i))
            self._plot_eeg(x[eeg])

    def plot_classified_digits(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, num: int = None) -> None:
        """
        Plots and saves a certain number of classified digits.

        :param x: the digit.
        :param y_pred: the predicted value of the classified digit.
        :param y_true: the real value of the classified digit.
        :param num: the number of digits to be plotted.
        """
        filename = self.filename

        # Plot num of digits randomly.
        for digit, i in self._random_picker(x, num):
            self.suptitle = 'Classified as {}'.format(y_pred[digit])
            self.title = 'Correct condition is {}'.format(y_true[digit])
            self.filename = '{}{}'.format(filename, str(i))
            self._plot_digit(x[digit])

    def heatmap_correlation(self, data: np.ndarray) -> None:
        """
        Create and save a heatmap, representing correlation.

        :param data: the correlated data to be plotted.
        """
        self._create_plot_folder()

        # Use a style.
        plt.style.use('seaborn-white')

        # Create a subplot.
        fig, ax = plt.subplots()

        # Create the heatmap.
        img = ax.matshow(data, aspect='auto')

        # Add a colorbar, showing the percentage of the correlation.
        plt.colorbar(img)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        self._save_and_show(fig)

    def scatter(self, x: np.ndarray, y: np.ndarray, class_labels: Callable[[int], str] = None) -> None:
        """
        Plots and saves a scatterplot with the first one, two or three features.

        :param x: the features to plot.
        :param y: the class labels.
        :param class_labels: an optional function which gets the class labels from their indexes.
        """
        if class_labels is None:
            def class_labels(index: int): return index

        # If we only have one principal component in a 1D array, i.e. M convert it to a 2D M x 1.
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        # If the principal components are more than two, the plot cannot be represented.
        elif x.shape[1] > 2:
            raise TooManyDimensionsError('Cannot plot more than 2 dimensions.')

        self._create_plot_folder()

        # Use a style.
        plt.style.use('seaborn-white')

        # Create a figure.
        fig = plt.figure(figsize=(10, 8))

        # Get the class labels and count each label's instances.
        labels, counts = np.unique(y, return_counts=True)

        # If there is one pc, plot 1D.
        if x.shape[1] == 1:
            # Create an ax.
            ax = fig.add_subplot(111)

            # For every class, scatter it's principal components.
            for i, count in zip(labels, counts):
                ax.scatter(x[y == i, 0], np.zeros((count, 1)), alpha=0.5,
                           label='{} class'.format(class_labels(i)))
                ax.legend()

            ax.set_title(self.title)
            # Set xlabel and clear x and y ticks.
            ax.set_xlabel(self.xlabel)
            ax.set_xticks([])
            ax.set_yticks([])

        # If there are 2 pcs plot 2D.
        elif x.shape[1] == 2:
            # Create an ax.
            ax = fig.add_subplot(111)

            # For every class, scatter it's principal components.
            for i, count in zip(labels, counts):
                ax.scatter(x[y == i, 0], x[y == i, 1], alpha=0.5, label='{} class'.format(class_labels(i)))
                ax.legend()

            ax.set_title(self.title)
            # Set x and y labels and clear x and y ticks.
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.set_xticks([])
            ax.set_yticks([])

        self._save_and_show(fig)
