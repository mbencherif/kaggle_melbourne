import caffe
import numpy as np
import h5py
import time

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

model_file = 'CRFRNN_EEG.prototxt'
#pretrained_file = 'CRFRNN_EEG.caffemodel'

print "Setting device..."
caffe.set_device(gpu_device)
caffe.set_mode_gpu()

print "Building model..."
net = caffe.Net(model_file, pretrained_file, caffe.TEST)

print "Reading data..."
mat_content = h5py.File('data/feat_train_2.mat')
EEG_feature = np.array(mat_content['feat_train'])
EEG_feature = EEG_feature.transpose()

mat_content = h5py.File('data/LABEL_train_2.mat')
EEG_label = np.array(mat_content['LABEL_train'])
EEG_label = EEG_label.transpose()

tic()
out = net.forward_all(**{net.inputs[0]: caffe_in})
toc()

predictions = out[net.outputs[0]]
