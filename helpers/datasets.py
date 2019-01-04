import numpy as np
from pathlib import Path
from typing import Tuple, List
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from definitions import __SEIZURE_PATH, __MNIST_PATH

Dataset = Tuple[np.ndarray, np.ndarray]


def _get_mnist_labels() -> List[str]:
    """
    Creates labels for the mnist dataset attributes.

    :return: List of strings containing the labels.
    """
    # Create a list with the prediction label's name.
    names = ['number']

    # For every pixel, create a label containing the word 'pixel', followed by its index.
    for i in range(784):
        names.append('pixel' + str(i))

    return names


def load_digits() -> Dataset:
    """
    Loads the mnist handwritten digits dataset.

    :return: Tuple of numpy arrays containing the mnist handwritten digits x and y.
    """
    # Read the dataset and get its values.
    dataset = read_csv(__MNIST_PATH, names=_get_mnist_labels())

    # Get x and y.
    x = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    return x, y


def load_seizure() -> Dataset:
    """
    Loads the epileptic seizure dataset.

    :return: Tuple of numpy arrays containing the epileptic seizure x and y.
    """
    # Read the dataset.
    dataset = read_csv(__SEIZURE_PATH, engine='python')
    # Drop irrelevant data.
    dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
    # Get x and y.
    x, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values

    return x, y


def database_split(x: np.ndarray, y: np.ndarray, train_filename: str, test_filename: str) -> None:
    """
    Splits a csv dataset to 60% train and 40% test files, stratified.

    :param x: numpy array containing the data.
    :param y: numpy array containing the labels.
    :param train_filename: the train filename to be created.
    :param test_filename: the test filename to be created.
    """
    # Split to train and test pairs.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=0)

    # Concatenate the train and test pairs arrays into single arrays.
    train = np.column_stack((x_train, y_train))
    test = np.column_stack((x_test, y_test))

    # Create Dataframes from the train and test arrays and write them to a csv file.
    DataFrame(train, dtype=np.str).to_csv(train_filename, index=False)
    DataFrame(test, dtype=np.str).to_csv(test_filename, index=False)


def get_eeg_name(class_num) -> str:
    """
    Return the name of the eeg corresponding to the given index.

    :param class_num: the index of the eeg name.
    :return: The eeg name.
    """
    class_names = {
        1: 'Eyes open',
        2: 'Eyes closed',
        3: 'Healthy cells',
        4: 'Cancer cells',
        5: 'Epileptic seizure'
    }
    return class_names.get(class_num, 'Invalid')
