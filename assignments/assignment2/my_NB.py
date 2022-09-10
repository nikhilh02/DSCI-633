import pandas as pd
import numpy as np
from collections import Counter


class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model

        self.classes_ = list(set(list(y)))
        # for calculation of P(y)
        self.P_y = Counter(y)
        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        self.P = {}

        for classVal in self.classes_:
            self.P[classVal] = {}

        # Calculating self.P

        featureList = list(X.columns)

        for feature in featureList:
            for classVal in self.classes_:
                self.P[classVal][feature] = {}
                featureLikelihood = X[feature][y[y == classVal].index.values.tolist()].value_counts().to_dict()
                for featureVal, featureCount in featureLikelihood.items():
                    l = ((featureCount + self.alpha) / (self.P_y[classVal] + (len(Counter(X[feature]).keys())*self.alpha)))
                    self.P[classVal][feature][featureVal] = l

        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # Hint: predicted class is the class with highest prediction probability (from self.predict_proba)
        probs = self.predict_proba(X)
        # predictions = "Write your own code"
        predictions = []
        for prob in probs.to_numpy():
            predictions.append(self.classes_[np.argmax(prob)])

        return predictions
