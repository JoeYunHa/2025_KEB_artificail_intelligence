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
        self.n_neighbors = n_neighbors


    def fit(self, X, y):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X_test) -> np.ndarray:
        """
        predict value for input
        :param X: new independent variable
        :return: predict value for input (np.ndarray)
        """
        # def comp(a,b):
        #     return abs(a) > abs(b)
        # diff_list = [(n - X) for n in self.X_list.flatten()]

        # 아래는 교수님께서 올려주신 방법
        predictions = []
        for x_test in X_test:
            distances = np.sqrt(np.sum((x_test - self.X_train)**2, axis=1))
            # argsort -> 배열을 직접 정렬하지 않고, 정렬된 인덱스 반환
            indices = np.argsort(distances)[:self.n_neighbors]
            prediction = np.mean(self.y_train[indices])
            predictions.append(prediction)

            return np.array(prediction).reshape(-1,1)


















