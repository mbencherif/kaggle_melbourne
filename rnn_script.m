clc
clear
close all

for subj = 2:2
    %% build network
    lrn_net = layrecnet();
%     lrn_net = narxnet(1:2,1:2,10);
    lrn_net.trainFcn = 'trainbr';
    lrn_net.trainParam.show = 5;
    lrn_net.trainParam.epochs = 5;
    
    %lrn_net.layers{1}.transferFcn = 'purelin';
    %lrn_net.layers{2}.transferFcn = 'purelin';
    
    %% load training data
    dir_path = sprintf('data/train_%d', subj);
    file_list = dir(dir_path);
    
    for i = 3:3%size(file_list,1)
        % load data
        file_name = sprintf('data/train_%d/%s', subj, file_list(i).name);
        load(file_name);
        eegdata = dataStruct.data;
        k = strfind(file_list(i).name,'_');
        label = str2num(file_list(i).name(k(2)+1));
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
        %eegdata = downsample(eegdata,10);
        
        if label == 0
            T = zeros(1, 240000);
        else
            T = ones(1, 240000);
        end

        % training rnn
        eegdata = con2seq(eegdata');
        T = con2seq(T);
        lrn_net = train(lrn_net, eegdata, T);    
    end
    
    %% load test data
    dir_path = sprintf('data/test_%d', subj);
    file_list = dir(dir_path);
    pred = [];
    
    for i = 3:3%size(file_list,1)
        % load data
        file_name = sprintf('data/test_%d/%s', subj, file_list(i).name);
        load(file_name);
        eegdata = dataStruct.data;
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
        %eegdata = downsample(eegdata,10);

        % prediction
        eegdata = con2seq(eegdata');
        Y = cell2mat(lrn_net(eegdata));
        if size(find(Y>=0.5),2) > size(find(Y<0.5),2)
            pred = [pred; 1];
        else
            pred = [pred; 0];
        end   
    end
    
    save('pred_rnn_sub2.mat', 'pred');
end
