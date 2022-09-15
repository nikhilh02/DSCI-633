import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.X = X
        self.y = y
        return

    def dist(self, x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        totalLen = len(self.X)
        distances = []
        if self.metric == "minkowski":
            for row in range(0, totalLen):
                distances.append((sum(abs(self.X.iloc[row] - x) ** self.p)) ** (1 / self.p))


        elif self.metric == "euclidean":
            for row in range(0, totalLen):
                distances.append(np.sqrt(sum((self.X.iloc[row] - x) ** 2)))


        elif self.metric == "manhattan":
            for row in range(0, totalLen):
                distances.append((sum(abs(x - self.X.iloc[row]))))


        elif self.metric == "cosine":
            for row in range(0, totalLen):
                x_y = sum(self.X.iloc[row] * x)
                vector_y = np.sqrt(sum(self.X.iloc[row] ** 2))
                vector_x = np.sqrt(sum(x ** 2))
                cos_x_y = x_y / (vector_y * vector_x)
                dist = 1 - cos_x_y
                distances.append(dist)

        else:
            raise Exception("Unknown criterion.")

        return distances

    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)
        distances = np.array(distances)
        minList = distances.argsort()[:(self.n_neighbors)]

        classList = self.y[minList]
        output = classList.value_counts().to_dict()

        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            # Calculate the probability of data point x belonging to each class
            # e.g. prob = {"2": 1/3, "1": 2/3}
            prob = neighbors
            probs.append(prob)
        probs = pd.DataFrame(probs, columns=self.classes_)
        probs = probs.replace(np.nan, 0)
        probs = probs / self.n_neighbors
        return probs



