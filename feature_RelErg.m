clc
clear
close all

load EEG_train.mat
load LABEL_train.mat

feature_train = [];
num_signal = size(EEG_train,1);
num_channel = size(EEG_train,2);
num_trial = size(EEG_train,3);

for trial = 1:num_trial
    x = EEG_train(:,:,trial)';
    feat = [];
    for i = 1:num_channel
        y = x(i,:);
        [c,l] = wavedec(y,4,'db1'); % daubechies wavelet, level 3
        energy_a4 = sumsqr(c(1:l(1)));
        energy_d1 = sumsqr(c(l(1)+1:l(1)+l(2)));
        energy_d2 = sumsqr(c(l(1)+l(2)+1:l(1)+l(2)+l(3)));
        energy_d3 = sumsqr(c(l(1)+l(2)+l(3)+1:l(1)+l(2)+l(3)+l(4)));
        energy_d4 = sumsqr(c(l(1)+l(2)+l(3)+l(4)+1:end));
        energy_total = energy_a4 + energy_d1 + energy_d2 + energy_d3 + energy_d4;
        re_a4 = energy_a4 / energy_total;
        re_d1 = energy_d1 / energy_total;
        re_d2 = energy_d2 / energy_total;
        re_d3 = energy_d3 / energy_total;
        re_d4 = energy_d4 / energy_total;
        feat = [feat re_a4 re_d1 re_d2 re_d3 re_d4];
    end
    feature_train = [feature_train; feat];
end

save('feature_RelErg.mat', 'feature_train', '-v7.3');