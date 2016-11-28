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
    norm_feat = [];
    x = EEG_train(:,:,trial)';
    for j = 1:num_signal/40
        y = x(:,1+(j-1)*40:j*40);
        for i = 1:num_channel
            psd=(abs(fft(y(i,:))).^2);
            % Select power spectral density bands
            %s(1,2*(i-1)+1) = sum(psd(17:25));
            %s(1,2*(i-1)+2) = sum(psd(26:71));
            s(1,i) = sum(psd);
        end
        norm_s = (s - min(s)) / (max(s) - min(s));
        norm_feat = [norm_feat norm_s];
    end
%     norm_feat = [norm_feat 0];
    feature_train = [feature_train;norm_feat];
end

save('feature_psd.mat', 'feature_train', '-v7.3');