from support.datatools import *
from support.paths import PATH
from objectives.logist import objective
from tqdm import tqdm
from numpy import *


def logloss(X, y):
    return log(1 + exp(matmul(X.T, y)))


class Coordinator:
    def __init__(self, objective, optimizer, loss):
        self.losses = {
            "logloss": logloss
        }

        message = "The loss param should match one of: " + ', '.join(self.losses.keys())
        assert loss in self.losses.keys(), message
        self.loss =  self.losses[loss]
        self.loss_name = loss
        self.optimizer = optimizer
        self.objective = objective
        
    
    def predict_proba(self, X):
        """
        Предсказание вероятности принадлежности объекта к классу 1.
        Возвращает np.array размера (N,) чисел в отрезке от 0 до 1.
        :param X: numpy.array размера  (N, M), dtype = np.float
        :return: numpy.array размера  (N,), dtype = np.int
        """
        return special.expit(objective(X))
    
    
    def predict(self, X):
        """
        Предсказание класса для объекта.
        Возвращает np.array размера (N,) элементов 1 или -1.
        :param X: numpy.array размера  (N, M), dtype = np.float
        :return:  numpy.array размера  (N,), dtype = np.int
        """
        return self.predict_proba(X) > .5
    
    
    def evaluate_generator(self, data_generator, nb_iterations):
        objective_loss = list()
        addition_loss = list()
        for i in tqdm(range(nb_iterations)):
            X, y = next(data_generator)
            objective_loss.append(self.objective.lossf(X, y))
            addition_loss.append(self.loss(self.objective(X), y))
        return objective_loss, addition_loss
    
    
    def predict_generator(self, data_generator, nb_iterations):
        predicted = list()
        for i in tqdm(range(nb_iterations)):
            X, y = next(data_generator)
            predicted += objective(X).tolist()
        return predicted
        

    def fit_generator(self, train_data, 
                      nb_iterations, nb_epoch, 
                      validation_data=None, 
                      nb_val_iterations=None,
                      verbose=0
                     ):
        """
        Обучение логистической регрессии.
        Настраивает self.w коэффициенты модели.
        Если self.verbose == True, то выводите значение 
        функции потерь на итерациях метода оптимизации. 
        :param X: numpy.array размера  (N, M), dtype = np.float
        :param y: numpy.array размера  (N,), dtype = np.int
        :return: self
        """
        history = {
            'objective_loss': [],
            'addition_loss': [],
            'objective_val_loss': [-1],
            'addition_val_loss': [-1]
        }
        
        for epoch in tqdm(range(nb_epoch)):
            objective_loss = list()
            addition_loss = list()
            
            for i in tqdm(range(nb_iterations)):
                X, y = next(train_data)
                objective_loss.append(self.objective.lossf(X, y))
                addition_loss.append(self.loss(self.objective(X), y))
                history['objective_loss'].append(mean(objective_loss))
                history['addition_loss'].append(mean(addition_loss))
                if verbose:
                    print("Epoch " + str(epoch) + "/" + str(nb_epoch)) 
                    if validation_data is not None:
                        print("Current objective val loss is " + str(history['objective_val_loss'][-1]))
                        print("Current val " + self.loss_name + " is " + str(history['addition_val_loss'][-1]))
                        print("Iteration " + str(i) + ".")
                        print("Current objective loss is " + str(history['objective_loss'][-1]))
                        print("Current " + self.loss_name + " is " + str(history['objective_loss'][-1]))
                self.w = self.optimizer(X, y)  
            
            if validation_data is not None:
                objective_val_loss, addition_val_loss = \
                    self.evaluate_generator(validation_data, nb_val_iterations)
                history['objective_val_loss'].append(mean(objective_val_loss))
                history['addition_val_loss'].append(mean(addition_val_loss))
        return  history