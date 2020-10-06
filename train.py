import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from joblib import dump, load




npzfile = np.load('data_training.npz')



pca = PCA(n_components=700,svd_solver='randomized',whiten=True).fit(npzfile['X'])
X_train_pca = pca.transform(npzfile['X'])
X_test_pca = pca.transform(npzfile1['X'])
print(np.shape(X_train_pca))


svclassifier = SVC(kernel='sigmoid', gamma='auto')
svclassifier.fit(X_train_pca, npzfile['Y'])
dump(svclassifier, 'svm_sigmoid_training.joblib')
