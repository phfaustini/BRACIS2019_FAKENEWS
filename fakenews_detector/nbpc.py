from math import exp, pi, sqrt

import numpy as np
import pandas as pd


class NBPC():
    def __init__(self):
        self.mean = {}
        self.var = {}
        self.t = 1.1

    def fit(self, X: np.ndarray) -> None:
        """ First, this method computes mean and variance for each feature in training set.
        Then, for each sample in training set, it computes the probability of that sample.
        This probability is the product of all of its features, given the gaussian formula.
        The member self.t receives the sample with the lowest probability score.

        :param X: ndarray, shape (n_samples, n_features), Input data.
        """
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        for sample in X:
            p = 1
            for j in range(sample.shape[0]):
                p *= self._calculate_probability_distribution(mean=self.mean[j], variance=self.var[j], feature=sample[j])
            if p < self.t:
                self.t = p

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.
        :param X: array-like, shape (n_samples, n_features)
        :return: y_pred : array, shape (n_samples,), Class labels for samples in X.
        """
        return np.array(list(map(self._predict_instance, X)))

    def _calculate_probability_distribution(self, mean: float, variance: float, feature: float) -> float:
        a = (1 / sqrt(2*pi*variance))
        b = (-1*(feature - mean)**2)/(2*variance)
        return a*exp(b)

    def _predict_instance(self, sample: np.ndarray) -> int:
        p = 1
        for j in range(sample.shape[0]):
            p *= self._calculate_probability_distribution(mean=self.mean[j], variance=self.var[j], feature=sample[j])
        return 1 if p >= self.t else -1

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: array-like, shape (n_samples, n_features)
        :return: dec : array-like, shape (n_samples,), Returns the decision function of the samples.
        """
        return np.array(list(map(self._decision_function, X)))

    def _decision_function(self, sample: np.ndarray) -> float:
        """
        It returns the sample's score. If such score is equal or greater
        than self.t threshold, sample is an inlier. Otherwise, it is an outlier.
        :return: float : sample's score probability.
        """
        p = 1
        for j in range(sample.shape[0]):
            p *= self._calculate_probability_distribution(mean=self.mean[j], variance=self.var[j], feature=sample[j])
        return p
