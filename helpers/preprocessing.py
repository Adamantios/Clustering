import numpy


def cut_images(images: numpy.ndarray, top: int = 2, bottom: int = 1) -> numpy.ndarray:
    """
    Cuts top rows from the top and bottom rows from the bottom of the passed images.

    :param images: Numpy array containing the images to be cut.
    :param top: the number of the rows to be cut from the top.
    :param bottom: the number of the rows to be cut from the bottom.
    :return: Numpy array containing the cut images.
    """
    top *= 28
    bottom *= 28

    # Return the array with the bottom rows deleted, from the array with the top rows deleted.
    return numpy.delete(numpy.delete(images, slice(0, top), 1), slice(784 - top - bottom, 784 - top), 1)
