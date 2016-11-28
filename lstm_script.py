import numpy as np
import h5py

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, TimeDistributed
# from keras.layers.core import Activation
#from keras.preprocessing import sequence
import pdb

np.random.seed(7)

# Load EEG data
mat_contents = h5py.File('EEG_train_2.mat')
EEG_train = mat_contents['EEG_train']
EEG_train = np.array(EEG_train)

# Load EEG labels
mat_contents = h5py.File('LABEL_train_2.mat')
LABEL_train = mat_contents['LABEL_train']
LABEL_train = np.array(LABEL_train)
LABEL_train = LABEL_train.T

# Train test split
X_train, X_test, y_train, y_test = train_test_split(EEG_train, LABEL_train, test_size=0.1, random_state=0)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Create model
model = Sequential()

#pdb.set_trace()

#model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(128, input_shape=(16, 24000), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train model
model.fit(X_train, y_train, nb_epoch=10, batch_size=16, class_weight={0:1,1:10})

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

pred_test = model.predict_classes(X_test, verbose=0)
con_mat = metrics.confusion_matrix(y_test, pred_test)
print(con_mat)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test)
print("AUC: %.2f" % (metrics.auc(fpr, tpr)))