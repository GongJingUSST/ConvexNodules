import numpy as np 
from scipy import special

def lossf(w, X, y, l1, l2):
    """
    Вычисление функции потерь.
    :param w: numpy.array размера  (M,) dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: float, value of loss function
    """
    lossf = np.log(1 + np.exp(-np.matmul(np.matmul(X, w).T, y))) + \
                   l1 * np.linalg.norm(w, ord=1) + \
                   l2 * np.square(np.linalg.norm(w, ord=2))
    return lossf


def gradf(w, X, y, l1, l2):
    """
    Вычисление градиента функции потерь.
    :param w: numpy.array размера  (M,), dtype = np.float
    :param X: numpy.array размера  (N, M), dtype = np.float
    :param y: numpy.array размера  (N,), dtype = np.int
    :param l1: float, l1 коэффициент регуляризатора 
    :param l2: float, l2 коэффициент регуляризатора 
    :return: numpy.array размера  (M,), dtype = np.float, gradient vector d lossf / dw
    """
    p = np.matmul(np.matmul(X, w).T, y)
    gradw = (special.expit(p) - 1) * np.matmul(X.T, y) + l1 * np.sign(w) + 2 * l2 * w
    return gradw