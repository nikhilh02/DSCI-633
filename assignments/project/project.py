import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys
from pdb import set_trace
import numpy as np
##################################
sys.path.insert(0,'../..')
from my_evaluation import my_evaluation
from sklearn.model_selection import RandomizedSearchCV

class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    def fit(self, X, y):
        # do not exceed 29 mins

        self.preprocessor1 = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        self.preprocessor2 = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        self.preprocessor3 = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        self.preprocessor4 = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)

        titleProcess = self.preprocessor1.fit_transform(X["title"])
        titleProcess = titleProcess.toarray()
        descProcess = self.preprocessor2.fit_transform(X["description"])
        descProcess = descProcess.toarray()
        locProcess = self.preprocessor3.fit_transform(X["location"])
        locProcess = locProcess.toarray()

        numVal = X[["telecommuting", "has_company_logo", "has_questions"]]
        numVal = numVal.to_numpy()
        
        XX = np.concatenate([titleProcess, descProcess, locProcess, numVal], axis=1)

        decision_keys = {"loss": ("hinge", "log_loss", "perceptron"), "penalty": ("l2", "l1"), "alpha": [0.0001, 0.01]}
        model = SGDClassifier()
        XX = pd.DataFrame(XX)

        self.clf = RandomizedSearchCV(model, decision_keys, cv=5)

        self.clf.fit(XX,y)

        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions

        titleProcess = self.preprocessor1.transform(X["title"])
        titleProcess = titleProcess.toarray()
        descProcess = self.preprocessor2.transform(X["description"])
        descProcess = descProcess.toarray()
        locProcess = self.preprocessor3.transform(X["location"])
        locProcess = locProcess.toarray()

        numVal = X[["telecommuting", "has_company_logo", "has_questions"]]
        numVal = numVal.to_numpy()
        
        XX = np.concatenate([titleProcess, descProcess, locProcess, numVal], axis=1)

        XX = pd.DataFrame(XX)

        predictions = self.clf.predict(XX)
        return predictions
