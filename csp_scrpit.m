clc
clear
close all

load EEG_train_1.mat
load LABEL_train_1.mat

num_signal = size(EEG_train,1);
num_channel = size(EEG_train,2);
num_trial = size(EEG_train,3);

permute(EEG_train,[2 1 3]);

indices = crossvalind('Kfold', num_trial, 10);

for i = 1:1
    testid = (indices == i); trainid = ~testid;
    
    train_X = EEG_train(:,:,trainid);
    train_Y = LABEL_train(trainid);
    test_X = EEG_train(:,:,testid);
    test_Y = LABEL_train(testid);
    num_train = size(train_X,3);
    num_test = size(test_X,3);
    
    spatial_filter = CSPovr(train_X, train_Y, 16);
    
    feat_train=[];
    for j = 1:num_train
        FeaData = spatial_filter * train_X(:,:,i);
        FeaData = diag(cov(FeaData'));
        FeaData = FeaData./sum(FeaData);
        FeaData = log(FeaData);
        feat_train = [feat_train; FeaData'];
    end
    
    feat_test=[];
    for j = 1:num_test
        FeaData = spatial_filter * EEG_3d_after(:,:,i);
        FeaData = diag(cov(FeaData'));
        FeaData = FeaData./sum(FeaData);
        FeaData = log(FeaData);
        feat_test = [feat_test; FeaData'];
    end
    
    save('feat_csp_train.mat', 'feat_train');
    save('feat_csp_test.mat', 'feat_test');
end