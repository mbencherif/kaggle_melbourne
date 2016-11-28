import numpy as np
import h5py

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
# from keras.layers.core import Activation
#from keras.preprocessing import sequence

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
model.add(Convolution1D(64, 3, border_mode='same', activation='relu', input_shape=(16, 24000)))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',class_mode="binary")
# model.add(LSTM(100, input_shape=(16, 24000)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train model
#earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
#it seems early stopping makes no sense without validation data
model.fit(X_train, y_train, nb_epoch=5, batch_size=32, class_weight={0:1,1:10}) 

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Score: %.3f" % (scores))

pred_test = model.predict_classes(X_test, verbose=0)
con_mat = metrics.confusion_matrix(y_test, pred_test)
print(con_mat)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test)
print("AUC: %.2f" % (metrics.auc(fpr, tpr)))