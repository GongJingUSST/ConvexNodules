import numpy as np 
from scipy import special

class objective:
    
    def __init__(self, dim, w=None, l1=1e-4, l2=1e-4):
        
        if w is None:
            self.w = np.random.randn(dim)
            self.w = np.clip(self.w, -3, 3)
        else:
            self.w = w
        self.l1 = l1 
        self.l2 = l2
        
        
    def __call__(self, X):
        return np.matmul(X, self.w).T
    

    def lossf(self, X, y):
        """
        Вычисление функции потерь.
        :param w: numpy.array размера  (M,) dtype = np.float
        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :param l1: float, l1 коэффициент регуляризатора 
        :param l2: float, l2 коэффициент регуляризатора 
        :return: float, value of loss function
        """
        lossf = np.log(1 + np.exp(np.matmul(np.matmul(X, self.w).T, y))) + \
                       self.l1 * np.linalg.norm(self.w, ord=1) + \
                       self.l2 * np.square(np.linalg.norm(self.w, ord=2))
        return lossf


    def gradf(self, X, y):
        """
        Вычисление градиента функции потерь.
        :param w: numpy.array размера  (M,), dtype = np.float
        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :param l1: float, l1 коэффициент регуляризатора 
        :param l2: float, l2 коэффициент регуляризатора 
        :return: numpy.array размера  (M,), dtype = np.float, gradient vector d lossf / dw
        """
        p = np.matmul(np.matmul(X, self.w).T, y)
        gradw = (special.expit(p) - 1) * np.matmul(X.T, y) + self.l1 * np.sign(self.w) + 2 * self.l2 * self.w
        return gradw
    
    
    
    
    