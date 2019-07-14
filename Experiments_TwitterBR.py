import pickle

import numpy as np
from sklearn.svm import OneClassSVM
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

#from fakenews_detector.preprocessor_twitter import PreProcessorTwitter
from fakenews_detector.model_occ import ModelOCC
from fakenews_detector.nbpc import NBPC
from fakenews_detector.ecoOCC import EcoOCC
from fakenews_detector.dcdistance_occ import DCDistanceOCC

RESULTS_FOLDER = 'results/Twitter'
DATASET = "Twitter/tweets_br"
c = ModelOCC(chunk_size_false=440, chunk_size_true=459)

#p = PreProcessorTwitter("datasets/Twitter/tweets_br/")
#p.convert_rawdataset_to_dataset(platform='Twitter', dataset='tweets_br', class_label='False')
#p.convert_rawdataset_to_dataset(platform='Twitter', dataset='tweets_br', class_label='True')


print("--------------------")
print("ECOOCC CST")
print("--------------------")
clf = EcoOCC()
f1score_scores_ecoocc_features, _, _  = c.classify_inliersoutliers(clf, dataset=DATASET)
pd.Series(f1score_scores_ecoocc_features).to_pickle('results/Twitter/f1scores_twitter_features_ecoocc.pkl')
print("k = {0} \u00B1{1}".format(np.mean(clf.ks), np.std(clf.ks)))
with open('results/Twitter/ks_twitter_features.pkl', 'wb') as f:
    pickle.dump(clf.ks, f)
print()


print("--------------------")
print("NAIVE BAYES CST")
print("--------------------")
clf = NBPC()
f1score_scores_nbpc_features, _, _ = c.classify_inliersoutliers(clf, dataset=DATASET)
pd.Series(f1score_scores_nbpc_features).to_pickle('results/Twitter/f1scores_twitter_features_nbpc.pkl')
print()


print("--------------------")
print("DCDISTANCE CST")
print("--------------------")
results_outliers, best_params = c.manual_gridsearch_dcdistance(thresholds=[i/100 for i in range(1, 100, 5)], dataset=DATASET, distance=euclidean)
print("Best params = \n{0}".format(best_params))
clf = DCDistanceOCC(t=best_params['t'], distance=euclidean)
f1score_scores_dcdistance_features, _, _  = c.classify_inliersoutliers(clf, dataset=DATASET, print_results=False)
results_outliers.to_pickle('results/Twitter/twitter_features_dcdistance_griddict.pkl')
pd.Series(f1score_scores_dcdistance_features).to_pickle('results/Twitter/f1scores_twitter_features_dcdistance.pkl')
print()


print("--------------------")
print("OCC SVM CST")
print("--------------------")
outliers, best_params = c.manual_gridsearch_occsvm(nu=[i/10 for i in range(1, 11)], kernel=['poly', 'rbf'], gamma=[.1, .2, .3, .4, .5, .6, .7, .8, .9], degree=[3], dataset=DATASET)
print("Best params = \n{0}".format(best_params))
clf = OneClassSVM(nu=best_params['nu'], kernel=best_params['kernel'], degree=best_params['degree'], gamma=best_params['gamma'])
f1score_scores_svm_features, _, _ = c.classify_inliersoutliers(clf, dataset=DATASET, print_results=False)
pd.Series(f1score_scores_svm_features).to_pickle('results/Twitter/f1scores_twitter_features_svm.pkl')
print()


f = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel(r'\textbf{f1score}', fontsize=11)
df = pd.DataFrame({'DCDistanceOCC': f1score_scores_dcdistance_features, 'OCCSVM': f1score_scores_svm_features, 'EcoOCC': f1score_scores_ecoocc_features, 'NBPC': f1score_scores_nbpc_features})
boxplot = df.boxplot(column=['DCDistanceOCC', 'OCCSVM', 'EcoOCC', 'NBPC'])
f.savefig("{0}/f1score_twitter_features.svg".format(RESULTS_FOLDER), dpi=100, bbox_inches='tight')
f.clear()
