import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None # weight
        self.intercept = None # bias

    def fit(self, X, y):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X - X_mean,2))
        numerator = np.sum((X - X_mean)*(y - y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * X)

    def predict(self, X) -> np.ndarray:
        """
        predict value for input
        :param X: new independent variable
        :return: predict value for input (np.ndarray)
        """
        return self.slope * np.array(X) * self.intercept

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5):


    def fit(self, X, y):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """


    def predict(self, X) -> np.ndarray:
        """
        predict value for input
        :param X: new independent variable
        :return: predict value for input (np.ndarray)
        """

