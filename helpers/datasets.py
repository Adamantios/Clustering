from pathlib import Path
from typing import Tuple

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split

from definitions import __GENES_TRAIN_PATH, __GENES_TEST_PATH, __GENES_DATA_PATH, __GENES_LABELS_PATH, __VOICE_PATH, \
    __VOICE_TRAIN_PATH, __VOICE_TEST_PATH

Dataset = Tuple[np.ndarray, np.ndarray]


def load_voice(train: bool = True) -> Dataset:
    """
    Loads the voices dataset.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Tuple of numpy arrays containing the robot wall following dataset x and y.
    """
    # Create Path objects using the paths where the train and test files should be.
    train_file = Path(__VOICE_TRAIN_PATH)
    test_file = Path(__VOICE_TEST_PATH)

    # If the files from the given paths do not exist, create them by splitting the genes dataset.
    if not train_file.is_file() or not test_file.is_file():
        # Read the dataset an get its values.
        # Use string datatypes, so that we take the information as it is.
        # If floats were to be used, then the labels would be converted to floats too.
        dataset = read_csv(__VOICE_PATH, engine='python')
        # Get x and y.
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        database_split(x, y, train_file.absolute(), test_file.absolute())

    # Create a filename based on the train value.
    filename = train_file.absolute() if train else test_file.absolute()
    # Read the dataset.
    dataset = read_csv(filename, engine='python')
    # Get x and y.
    x, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values

    return x, y


def load_genes(train: bool = True) -> Dataset:
    """
    Loads the genes dataset.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Tuple of numpy arrays containing the genes x and y.
    """
    # Create Path objects using the paths where the train and test files should be.
    train_file = Path(__GENES_TRAIN_PATH)
    test_file = Path(__GENES_TEST_PATH)

    # If the files from the given paths do not exist, create them by splitting the genes dataset.
    if not train_file.is_file() or not test_file.is_file():
        # Read the dataset an get its values.
        # Use string datatypes, so that we take the information as it is.
        # If floats were to be used, then the labels would be converted to floats too.
        x = read_csv(__GENES_DATA_PATH, engine='python')
        y = read_csv(__GENES_LABELS_PATH, engine='python')
        # Get x and y.
        x, y = x.iloc[:, 1:].values, y.iloc[:, 1].values

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        database_split(x, y, train_file.absolute(), test_file.absolute())

    # Create a filename based on the train value.
    filename = train_file.absolute() if train else test_file.absolute()
    # Read the dataset.
    dataset = read_csv(filename, engine='python')
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


def get_voice_name(class_num) -> str:
    """
    Return the gender of the given index.

    :param class_num: the index.
    :return: The gender.
    """
    class_names = {
        0: 'Female',
        1: 'Male'
    }
    return class_names.get(class_num, 'Invalid')


def get_gene_name(class_num) -> str:
    """
    Return the name of the gene corresponding to the given index.

    :param class_num: the index of the gene's name.
    :return: The gene's name.
    """
    class_names = {
        0: 'BRCA',
        1: 'COAD',
        2: 'KIRC',
        3: 'LUAD',
        4: 'PRAD'
    }
    return class_names.get(class_num, 'Invalid')
