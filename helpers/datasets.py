import numpy as np
from typing import Tuple
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from definitions import __SEIZURE_PATH, __MNIST_PATH, __WALL_FOLLOWING_PATH

Dataset = Tuple[np.ndarray, np.ndarray]


def load_wall_following() -> Dataset:
    """
    Loads the robot wall following dataset.

    :return: Tuple of numpy arrays containing the robot wall following dataset x and y.
    """
    # Read the dataset and get its values.
    dataset = read_csv(__WALL_FOLLOWING_PATH, engine='python')

    # Get x and y.
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    return x, y


def load_digits() -> Dataset:
    """
    Loads the mnist handwritten digits dataset.

    :return: Tuple of numpy arrays containing the mnist handwritten digits x and y.
    """
    # Read the dataset and get its values.
    dataset = read_csv(__MNIST_PATH, engine='python')

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


def get_wall_following_name(class_num) -> str:
    """
    Return the name of the turn corresponding to the given index.

    :param class_num: the index of the turn class.
    :return: The turn's class name.
    """
    class_names = {
        0: 'Move Forward',
        1: 'Sharp Right Turn',
        2: 'Slight Left Turn',
        3: 'Slight Right Turn'
    }
    return class_names.get(class_num, 'Invalid')
