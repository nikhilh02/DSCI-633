import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math

class my_AdaBoost:

    def __init__(self, base_estimator=None, n_estimators=50):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # If one base estimator predicts perfectly,
            # Use that base estimator only
            tempAlpha = 0
            if error == 0:
                self.alpha = [1]
                self.estimators = [self.estimators[i]]
                break
            # Compute alpha for estimator i (don't forget to use k for multi-class)
            elif error > 0.5:
                tempAlpha = math.log((1 - error) / error)
            else:
                tempAlpha = math.log((1 - error) / error) + math.log(k - 1)

            # self.alpha.append("write your own code")
            self.alpha.append(tempAlpha)

            # Update wi
            # w = "write your own code"

            tempWeights = []
            for i in (range(len(w))):
                if (diffs[i]):
                    tempWeights.append(w[i] * np.exp(tempAlpha))
                else:
                    tempWeights.append(w[i])

            w = np.array(tempWeights)
            weightSum = np.sum(w)
            w = w / weightSum

        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)

        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # Note that len(self.estimators) can sometimes be different from self.n_estimators
        # write your code below
        probs = {}
        probsList = []

        predictionsTable = []
        for x in range(self.n_estimators):
            predictionsTable.append(self.estimators[x].predict(X))

        predictionsTable = pd.DataFrame(predictionsTable)

        totalCol = predictionsTable.shape[1]
        totalRow = predictionsTable.shape[0]

        for colVal in range(totalCol):

            classLevelProb = {}
            for label in self.classes_:
                classLevelProb[label] = 0

            for rowVal in range(totalRow):
                labelVal = predictionsTable.iloc[rowVal, colVal]
                classLevelProb[labelVal] = classLevelProb[labelVal] + self.alpha[rowVal]

            probsTemp = {}
            for label in self.classes_:
                probsTemp[label] = classLevelProb[label]
            probsList.append(probsTemp)

        probs = pd.DataFrame(probsList, columns=self.classes_)
        return probs