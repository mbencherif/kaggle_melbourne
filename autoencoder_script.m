clc
clear
close all

load EEG_train_1.mat
load LABEL_train_1.mat

num_signal = size(EEG_train,1);
num_channel = size(EEG_train,2);
num_trial = size(EEG_train,3);

indices = crossvalind('Kfold', num_trial, 10);

for i = 1:1
    testid = (indices == i); trainid = ~testid;
    
    train_X = EEG_train(:,:,trainid);
    train_Y = LABEL_train(trainid);
    test_X = EEG_train(:,:,testid);
    test_Y = LABEL_train(testid);
    num_train = size(train_X,3);
    num_test = size(test_X,3);
    
    deepnet = network;
    
    % Training Network
    for j = 1:num_train
        eegData = train_X(:,:,j)';
        T = zeros(2,num_signal);
        if train_Y(j) == 0
            T(1,:) = 1;
        else
            T(2,:) = 1;
        end
        
        if j == 1
            hiddenSize = 10;
            autoenc1 = trainAutoencoder(eegData,hiddenSize,'MaxEpochs',200,...
                'UseGPU',false,'L2WeightRegularization',0.001,...
                'SparsityRegularization',4,'SparsityProportion',0.05,...
                'DecoderTransferFunction','purelin','ShowProgressWindow',false);
            
            % Extract features in hidden layer
            features1 = encode(autoenc1,eegData);
            
            % Train a second autoencoder
            autoenc2 = trainAutoencoder(features1,hiddenSize,'MaxEpochs',200,...
                'UseGPU',false,'L2WeightRegularization',0.001,...
                'SparsityRegularization',4,'SparsityProportion',0.05,...
                'DecoderTransferFunction','purelin','ShowProgressWindow',false);
            
            % Extract features in hidden layer
            features2 = encode(autoenc2,features1);
            
            % Train a softmax layer for classification using features,
            % features2, from the second autoencoder, autoenc2
            softnet = trainSoftmaxLayer(features2,T,'LossFunction',...
                'crossentropy');
            
            % Stack the encoders and the softmax layer to form a deep network.
            deepnet = stack(autoenc1,autoenc2,softnet);
            
            % Train the deep network on the training data.
            % Can add further data to eegData to retrain the model.
            deepnet = train(deepnet,eegData,T,'UseParallel','yes');
            deepnet.adaptFcn = 'adaptwb';
        else
            deepnet = adapt(deepnet,eegData,T);
        end
    end
    
    % Validation
    pred = [];
    for j = 1:num_test
        eegData = test_X(:,:,j)';
        predictedValues = deepnet(eegData);
        if sum(predictedValues(1,:)) > sum(predictedValues(2,:))
            pred = [pred; 0];
        else
            pred = [pred; 1];
        end
    end
    
    save('trained_deepnet.mat','deepnet');
    clear deepnet
    
    err_rate = sum(test_Y ~= pred)/num_test;
    disp(['Autoencoder error rate: ', num2str(err_rate)]);
    conMat = confusionmat(test_Y, pred)
    spec = conMat(2,2)/(conMat(2,1)+conMat(2,2));
    disp(['Autoencoder specificity: ', num2str(spec)]);
    [roc_X,roc_Y,~,auc] = perfcurve(test_Y,pred,1);
    disp(['Autoencoder AUC: ' num2str(auc)]);
end
