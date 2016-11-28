clc
clear
close all

%% Data preprocessing

for subj = 1:1 % Temporarily 1 subject
    EEG_train = []; LABEL_train = [];
    EEG_test = []; 
    
    % training data
    dir_path = sprintf('data/train_%d', subj);
    file_list = dir(dir_path);
    count = 1;
    for i = 3:size(file_list,1)
        % load data
        file_name = sprintf('data/train_%d/%s', subj, file_list(i).name);
        load(file_name);
        eegdata = dataStruct.data;
%         k = strfind(file_list(i).name,'_');
%         label = str2num(file_list(i).name(k(2)+1));
        % period = dataStruct.sequence;
        
        % data drop-out
        if nnz(eegdata) == 0
            display(['Data in ', file_list(i).name, ' is bad.']);
            continue;
        end
        
        % preprocessing
        [B,A] = butter(8,[8 30] / 400 * 2);
        eegdata = filtfilt(B, A, double(eegdata));
        eegdata = detrend(eegdata);
        eegdata = downsample(eegdata,10);
        
        % Split into sub-windows
%         [nt,nc] = size(eegdata); fs = 400;
%         subsampLen = floor(fs * 60);
%         numSamps = floor(nt / subsampLen);
%         sampIdx = 1 : subsampLen : numSamps*subsampLen + 1;
%         for l = 1:numSamps
%             epoch = eegdata(sampIdx(l):sampIdx(l+1)-1,:);
%             EEG_train(:,:,count) = epoch;
%             count = count + 1;
%         end
        
        %eegdata = reshape(eegdata,24000,10)';
        
        % merge data
%         for j = 1:10    
%             EEG_train(:,:,count*10+j) = eegdata;
%         end
        EEG_train(:,:,count) = eegdata;
        count = count + 1;
%         LABEL_train = [LABEL_train; label];
    end
    
    % test data
%     dir_path = sprintf('test_%d', subj);
%     file_list = dir(dir_path);
%     count = 0;
%     for i = 3:size(file_list,1)
%         % load data
%         file_name = sprintf('test_%d/%s', subj, file_list(i).name);
%         load(file_name);
%         eegdata = dataStruct.data;
%         % period = dataStruct.sequence;
%         
%         % data drop-out
%         if nnz(eegdata) == 0
%             display(['Data in ', file_list(i).name, ' is bad.']);
%             continue;
%         end
%         
%         % preprocessing
%         [B,A] = butter(8,[8 30] / 400 * 2);
%         eegdata = filtfilt(B, A, double(eegdata));
%         eegdata = detrend(eegdata);
%         eegdata = downsample(eegdata,10);
%         
%         % Split into sub-windows
%         [nt,nc] = size(eegdata); fs = 400;
%         subsampLen = floor(fs * 60);
%         numSamps = floor(nt / subsampLen);
%         sampIdx = 1 : subsampLen : numSamps*subsampLen + 1;
%         for l = 1:numSamps
%             epoch = eegdata(sampIdx(l):sampIdx(l+1)-1,:);
%             EEG_test(:,:,count) = epoch;
%             count = count + 1;
%         end       
%     end
    
%     fileName = ['LABEL_train_', num2str(subj), '.mat'];
%     save(fileName, 'LABEL_train', '-v7.3');
    fileName = ['EEG_train_', num2str(subj), '.mat'];
    save(fileName, 'EEG_train', '-v7.3');
%     fileName = ['EEG_test_', num2str(subj), '.mat'];
%     save(fileName, 'EEG_test', '-v7.3');
end

%% Feature Extraction
% spatial_filter = CSPovr(EEG_train, LABEL_train, 16);
% feature_train=[];
% for i = 1:size(EEG_train,3)
%     FeaData = spatial_filter * EEG_train(:,:,i);
%     FeaData = diag(cov(FeaData'));
%     FeaData = FeaData./sum(FeaData);
%     FeaData = log(FeaData);
%     feature_train = [feature_train; FeaData'];
% end

% feature_test=[];
% for i = 1:size(EEG_test,3)
%     FeaData = spatial_filter * EEG_test(:,:,i);
%     FeaData = diag(cov(FeaData'));
%     FeaData = FeaData./sum(FeaData);
%     FeaData = log(FeaData);
%     feature_test = [feature_test; FeaData'];
% end

%% Classification
% num_train = size(EEG_train, 3);
% indices = crossvalind('Kfold', num_train, 10);
% for i = 1:10
%     test = (indices == i); train = ~test;
%     svmStruct = svmtrain(feature_train(train,:), LABEL_train(train));
%     pred = svmclassify(svmStruct, feature_train(test,:));
%     err_rate = sum(LABEL_train(test) ~= pred)/size(LABEL_train(test),1)
%     conMat = confusionmat(LABEL_train(test), pred)
% end