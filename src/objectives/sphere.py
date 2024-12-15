import numpy as np


def sphere(x):
    """
    Computes the d-dimensional Sphere function.
    :param x: np.array, shape (d,) - point at which to evaluate the function.
    :return: float - value of the Sphere function at x.
    """
    return np.sum(x ** 2)
