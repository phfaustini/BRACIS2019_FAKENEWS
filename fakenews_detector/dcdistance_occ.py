'''
Copyright 2019 <Pedro Faustini>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
from typing import Callable

import numpy as np


class DCDistanceOCC():
    """DCDistance classifier for One-Class Classification.

    Parameters
    ----------
    t : float
        Threshold value in range (0, 1]

    distance : callable
        A function that takes two numbers and returns their distance (e.g. scipy.spatial.distance.cosine)

    Attributes
    ----------

    X_training_reduced : array-like, shape = [n_instances]
        Distance betweetn class_vector and ith sample in training data.

    t : float
        Threshold parameter.

    d : float
        Threshold distance. If a test object has distance equal to or smaller than t, object is an inlier.
        The higher this parameter is, more objects tend to be considered inliers.

    distance : function
        Function that computes the distance between two vectors.

    class_vector : array-like, shape = [n_features]
        Sum (along axis=0) of all vectors in X_training.
    """

    def __init__(self, t: float, distance: Callable):
        self.X_training_reduced = None
        self.t = t
        self.d = 0
        self.distance = distance
        self.class_vector = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Calculates class vectors and updates threshold t member.
        In order to update t, the method first takes the maximum distance between a training object
        and class vector. Then, multiply the result for the initial value of t.
        :param X: np.ndarray, bag-of-words.
        :return: np.ndarray with dimensionality reduced to 1 features per sample in training data.
        """
        self.class_vector = X.sum(axis=0)
        dcdistances = np.array([self.distance(self.class_vector, x) for x in X])
        self.X_training_reduced = dcdistances
        self.d = self.t * dcdistances.max()
        return dcdistances

    def _predict_instance(self, X: np.ndarray) -> int:
        return 1 if self.distance(self.class_vector, X) <= self.d else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.
        :param X: array-like, shape (n_samples, n_features)
        :return: y_pred : array, shape (n_samples,), Class labels for samples in X.
        """
        return np.array(list(map(self._predict_instance, X)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: array-like, shape (n_samples, n_features)
        :return: dec : array-like, shape (n_samples,), Returns the decision function of the samples.
        """
        return np.array(list(map(self._decision_function, X)))

    def _decision_function(self, sample: np.ndarray) -> float:
        """
        It returns the sample's distances to the class vector.
        :return: array with the distances between samples and class vector.
        """
        return self.distance(self.class_vector, sample)
