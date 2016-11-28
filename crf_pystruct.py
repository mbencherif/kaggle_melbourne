import h5py
import numpy as np
from pystruct.models import ChainCRF, GraphCRF, LatentNodeCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM, LatentSSVM
from sklearn.model_selection import train_test_split

mat_content = h5py.File('feat_train_1.mat')
EEG_feature = np.array(mat_content['feat_train'])
EEG_feature = EEG_feature.transpose()
mat_content = h5py.File('LABEL_train_1.mat')
EEG_label = np.array(mat_content['LABEL_train'])
EEG_label = EEG_label.transpose()

X_train, X_test, y_train, y_test = train_test_split(EEG_feature, EEG_label, test_size=0.4, random_state=0)
X_train = X_train.astype(float)
X_test = X_test.astype(float)
X_train_ = np.expand_dims(X_train, axis=1)
X_test_ = np.expand_dims(X_test, axis=1)

#latent_crf = LatentNodeCRF(n_labels=2, n_features=2140, n_hidden_states=2, inference_method='lp')
#ssvm = OneSlackSSVM(model=latent_crf, max_iter=200, C=100, n_jobs=-1, show_loss_every=10, inference_cache=50)
#latent_svm = LatentSSVM(ssvm)

# Random initialization
#H_init = 

#latent_svm.fit(X_train, Y_train, H_init)
#print("Training score with latent nodes: %f" % latent_svm.score(X, Y))
#H = latent_svm.predict_latent(X)

crf = ChainCRF()
svm = NSlackSSVM(model=crf, max_iter=200, C=1, n_jobs=1)
svm.fit(X_train, y_train)
ssvm.score(X_test, y_test)