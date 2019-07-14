import os
from glob import glob
import warnings

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD as LSA

from .dcdistance_occ import DCDistanceOCC


warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0})


class ModelOCC(object):

    RANDOM_STATE = 42
    DTYPES_TEXT = {'uppercase': np.float64, 'exclamation': np.float64, "has_exclamation": np.float64, 'question': np.float64, 'adj': np.float64, 'adv': np.float64, 'noun': np.float64, 'spell_errors': np.float64, 'lexical_size': np.float64, 'Text': str, 'polarity': np.float64, 'number_sentences': np.float64, 'len_text': np.float64, 'label': np.float64, 'words_per_sentence': np.float64, 'swear_words': np.float64}
    FEATURES_TEXT = ["uppercase", "exclamation", "has_exclamation", "question", "adj", "adv", "noun", "spell_errors", "lexical_size", "polarity", "number_sentences", "len_text", "words_per_sentence", "swear_words"]

    def __init__(self, chunk_size_false=12, chunk_size_true=12):
        self._chunk_size_false = chunk_size_false
        self._chunk_size_true = chunk_size_true
        self.false_data = None
        self.true_data = None

    def _scale_columns(self, data: pd.DataFrame, columns: list, scaler='standard') -> tuple:
        """
        Scale data.
        :param scaler: either 'standard' for zscore, or 'minmax' for min-max scaling.
        :param columns: a list of strings, as ["number_urls", "number_sentences"]
        :return: a tuple (data, scaler) where data is a pd.DataFrame() with the provided columns scaled.
        and scaler is a StandardScaler object.
        """
        if scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        if len(columns) > 0:
            X_scaled = scaler.fit_transform(data.loc[:, columns])
            data[columns] = X_scaled
        return data, scaler

    def load_data(self, folders: list, dtypes=DTYPES_TEXT) -> pd.DataFrame():
        """
        Returns a pd.DataFrame with objects of all .csv files
        inside folders listed in folders parameter.

        :param folder: a list of strings list of folders
            e.g.: 'datasets/WhatsApp/br/Structured/False/'
        :returns: a pd.DataFrame with objects' data.
        """
        df = pd.DataFrame()
        c = 0
        for folder in folders:
            files = sorted(glob("{0}/*".format(folder)))
            for f in files:
                try:
                    pd_obj = pd.read_csv("{0}".format(f), index_col=False, dtype=dtypes, header=0)
                    pd_obj['topic'] = [c]
                    df = df.append(pd_obj)
                except:
                    print("Error loading .csv {0}!".format(f))
            c += 1
        df.index = range(df.shape[0])
        return df

    def dataset_statistics(self, dataset='WhatsApp/whats_br') -> dict:
        f = ModelOCC.FEATURES_TEXT[:]
        false_folder = sorted(glob('datasets/{0}/Structured/False/*/'.format(dataset)))
        if self.false_data is None: self.false_data = self.load_data(folders=false_folder)
        d = {}
        for feature in f:
            d[feature+"_mean"] = self.false_data[feature].mean()
            d[feature+"_std"] = self.false_data[feature].std()
        return d

    def _split(self, df: pd.DataFrame(), chunk_size: int) -> list:  # http://yaoyao.codes/pandas/2018/01/23/pandas-split-a-dataframe-into-chunks
        """Splits a pd.DataFrame into chunks of size chunk_size.
        Last chunk may have less than chunk_size samples.
        :param df: pd.DataFrame()
        :chunk_size: int telling how many samples each chunk must have.
        :return: a list with each element being a pd.DataFrame.
        """
        def index_marks(nrows, chunk_size):
            return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)
        indices = index_marks(df.shape[0], chunk_size)
        dataframes = np.split(df, indices)
        return list(filter(lambda x: not x.empty, dataframes))

    def _exclude(self, l: list, i: int) -> list:
        """From a given list l and index i,
        returns a new list with all elements in
        the original list, except for the one at index i.
        """
        if i == 0:
            return l[i+1:]
        return l[:i] + l[i+1:]

    def classify_inliersoutliers(self, clf, dataset='WhatsApp/whats_br', print_results=True) -> tuple:
        """Classification with outliers.
        It is used a dataset split in training and test data.
        General performance metrics are evaluated.
        :return: a tuple of np.arrays (f1score, precision, recall).
        """
        f = ModelOCC.FEATURES_TEXT[:]
        false_folder = sorted(glob('datasets/{0}/Structured/False/*/'.format(dataset)))
        true_folder = sorted(glob('datasets/{0}/Structured/True/*/'.format(dataset)))
        if self.false_data is None: self.false_data = self.load_data(folders=false_folder)
        if self.true_data is None: self.true_data = self.load_data(folders=true_folder)
        split_data_false = self._split(self.false_data, chunk_size=self._chunk_size_false)
        split_data_true = self._split(self.true_data, chunk_size=self._chunk_size_true)
        f1score_scores = np.array([])
        recall_scores = np.array([])
        precision_scores = np.array([])
        for i in range(len(split_data_false)):
            test_false_data = split_data_false[i]
            train_data = pd.concat(self._exclude(split_data_false, i))
            train_data, scaler = self._scale_columns(train_data, f)
            clf.fit(train_data.loc[:, f].values)
            for true_data in split_data_true:
                test_data = pd.concat([test_false_data, true_data])
                X_test_transformed = scaler.transform(test_data.loc[:, f].values)
                y_pred = clf.predict(X_test_transformed)
                f1score = f1_score(test_data.label, y_pred, average='binary', pos_label=1)
                precision = precision_score(test_data.label, y_pred, average='binary', pos_label=1)
                recall = recall_score(test_data.label, y_pred, average='binary', pos_label=1)
                f1score_scores = np.append(f1score_scores, f1score)
                precision_scores = np.append(precision_scores, precision)
                recall_scores = np.append(recall_scores, recall)
        if print_results:
            print("f1score = {0}% \u00B1{1}%".format(int(round(f1score_scores.mean()*100)), int(round(f1score_scores.std()*100))))
            print("precision = {0}% \u00B1{1}%".format(int(round(precision_scores.mean()*100)), int(round(precision_scores.std()*100))))
            print("recall = {0}% \u00B1{1}%".format(int(round(recall_scores.mean()*100)), int(round(recall_scores.std()*100))))
        return f1score_scores, precision_scores, recall_scores

    def manual_gridsearch_occsvm(self, nu: list, kernel: list, gamma: list, degree: list, dataset='WhatsApp/whats_br') -> tuple:
        """Peforms a gridsearch on OCC SVM parameters.).
        :param nu: a list of floats.
        :param kernel: a list os strings. Acceptable kernels: linear, rbf, poly and sigmoid.
        :param gamma: a list of floats.
        :param degree: a list os ints. Used only with poly kernel.
        :return: 2-uple of pd.DataFrame with results for inliers and inliers/outliers test objects.
        """
        results_outliers = pd.DataFrame(columns=['nu', 'kernel', 'gamma', 'degree', 'f1score_mean', 'f1score_std', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std'])
        for k in kernel:
            for n in nu:
                for g in gamma:
                    for d in degree:
                        clf = OneClassSVM(nu=n, kernel=k, degree=d, gamma=g)
                        f1score_scores, precision_scores, recall_scores = self.classify_inliersoutliers(clf=clf, dataset=dataset, print_results=False)
                        results_outliers = results_outliers.append({'nu': n, 'kernel': k, 'gamma': g, 'degree': d, 'f1score_mean': f1score_scores.mean(), 'f1score_std': f1score_scores.std(), 'precision_mean': precision_scores.mean(), 'precision_std': precision_scores.std(), 'recall_mean': recall_scores.mean(), 'recall_std': recall_scores.std()}, ignore_index=True)
                        if k != 'poly':  # Degree param is only used with poly kernel.
                            break
        best_outliers_idx = results_outliers['f1score_mean'].argmax()
        best_params_outliers = results_outliers.iloc[best_outliers_idx]
        return results_outliers, best_params_outliers

    def manual_gridsearch_dcdistance(self, thresholds: list, distance=cosine, dataset='WhatsApp/whats_br') -> tuple:
        """Peforms a gridsearch on DCDistanceOCC parameters.
        """
        results_outliers = pd.DataFrame(columns=['t', 'f1score_mean', 'f1score_std', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std'])
        for t in thresholds:
            clf = DCDistanceOCC(t=t, distance=distance)
            f1score_scores, precision_scores, recall_scores = self.classify_inliersoutliers(clf=clf, dataset=dataset, print_results=False)
            results_outliers = results_outliers.append({'t': t, 'f1score_mean': f1score_scores.mean(), 'f1score_std': f1score_scores.std(), 'precision_mean': precision_scores.mean(), 'precision_std': precision_scores.std(), 'recall_mean': recall_scores.mean(), 'recall_std': recall_scores.std()}, ignore_index=True)
        best_outliers_idx = results_outliers['f1score_mean'].argmax()
        best_params_outliers = results_outliers.iloc[best_outliers_idx]
        return results_outliers, best_params_outliers

    def plot(self, x: np.ndarray, y: np.ndarray, filename: str, x_label=r'$\nu$', y_label=r'\textbf{f1score}'):
        f = plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(x, y)
        plt.xlabel(x_label, fontsize=11)
        plt.ylabel(y_label, fontsize=11)
        f.savefig(filename, dpi=100, bbox_inches='tight')
        print("------------------")
