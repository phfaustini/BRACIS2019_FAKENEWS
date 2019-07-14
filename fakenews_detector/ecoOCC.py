import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class EcoOCC(KMeans):
    def __init__(self, n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=10**(-4), precompute_distances='auto', verbose=0, copy_x=True, random_state=42, n_jobs=10, algorithm='auto'):
        super(EcoOCC, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances=precompute_distances, verbose=verbose, copy_x=copy_x, random_state=random_state, n_jobs=n_jobs, algorithm=algorithm)
        self.distances = None
        self.radii = None
        self.randomstate = random_state
        self.ks = []

    def _compute_radii(self) -> None:
        clusters = self.cluster_centers_.shape[0]
        self.radii = np.zeros(clusters)
        for element in self.distances:
            cluster = element.argmin()
            distance = element.min()
            if distance > self.radii[cluster]:
                self.radii[cluster] = distance

    def fit(self, X: np.ndarray) -> None:
        """ First, KMeans is run with k ranging from
        [2, sqrt(n)], where n is the number of samples in training data X.
        The model is then fitted with the best k (it is the one with the
        highest silhouette score).
        The distances between each sample and its centroid is stored in
        self.distance. Finally, the radius of each centroid is calculated
        and stored in self.radii as an array, shape (centroid,).

        :param X: ndarray, shape (n_samples, n_features), Input data.
        """
        instances = X.shape[0]
        rounds = int(np.round(np.sqrt(instances)))
        best_silhouete = -1.1
        clusters = 2
        for k in range(2, rounds):
            self.n_clusters = k
            cluster_labels = super(EcoOCC, self).fit(X).labels_
            silhouette = round(silhouette_score(X, cluster_labels, random_state=self.randomstate), 5)
            if silhouette > best_silhouete:
                best_silhouete = silhouette
                clusters = k
        self.n_clusters = clusters
        self.ks.append(clusters)
        self.distances = super(EcoOCC, self).fit(X).transform(X)
        self._compute_radii()

    def _predict_instance(self, sample: np.ndarray) -> int:
        """
        Each cluster has a radius. If the distance between the sample object
        and every centroid is higher than the radius of that centroid's cluster,
        sample is an outlier.
        Radius is the maximum distance between the cluster's centroid and
        any object from that cluster.
        :return: -1 if sample is not in any cluster (oulier), 1 otherwise.
        """
        clusters = self.cluster_centers_.shape[0]
        for cluster in range(clusters):
            if euclidean(sample, self.cluster_centers_[cluster]) < self.radii[cluster]:
                return 1
        return -1

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

    def _decision_function(self, sample: np.ndarray) -> np.ndarray:
        """
        It returns the sample's distances to the centroids.
        :return: array with the distances to each centroid. Index 0 has the
        distance to centroid number 0, and so on.
        """
        clusters = self.cluster_centers_.shape[0]
        distances = []
        for cluster in range(clusters):
            distances.append(euclidean(sample, self.cluster_centers_[cluster]))
        return np.array(distances)
