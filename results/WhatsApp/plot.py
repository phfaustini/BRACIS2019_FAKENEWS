import pickle
import pandas as pd 
import matplotlib.pyplot as plt

f1score_scores_dcdistance_features = pd.read_pickle('f1scores_whatsapp_features_dcdistance.pkl')
f1score_scores_svm_features = pd.read_pickle('f1scores_whatsapp_features_svm.pkl')
f1score_scores_ecoocc_features = pd.read_pickle('f1scores_whatsapp_features_ecoocc.pkl')
f1score_scores_nbpc_features = pd.read_pickle('f1scores_whatsapp_features_nbpc.pkl')


f = plt.figure() 
plt.rc('text', usetex=True) 
plt.rc('font', family='serif') 
plt.ylabel(r'\textbf{f1score}', fontsize=18)
df = pd.DataFrame({'DCDistanceOCC': f1score_scores_dcdistance_features, 'OCCSVM': f1score_scores_svm_features, 'EcoOCC': f1score_scores_ecoocc_features, 'NBPC': f1score_scores_nbpc_features}) 
boxplot = df.boxplot(column=['DCDistanceOCC', 'OCCSVM', 'EcoOCC', 'NBPC'], fontsize=14, grid=False) 
f.savefig("f1score_whatsapp_features.pdf", dpi=100, bbox_inches='tight') 
f.clear() 
