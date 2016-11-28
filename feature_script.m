clc
clear
close all

for subj = 1:1
    % training data
    dir_path = sprintf('data/train_%d', subj);
    file_list = dir(dir_path);
    
    feat_train = [];
    for i = 3:size(file_list,1)
        % load data
        file_name = sprintf('data/train_%d/%s', subj, file_list(i).name);
        load(file_name);
        eegData = dataStruct.data;
        
        % data drop-out
        if nnz(eegData) == 0
            display(['Data in ', file_list(i).name, ' is bad.']);
            continue;
        end
        
        % preprocessing
        [B,A] = butter(8,[8 30] / 400 * 2);
        eegData = filtfilt(B, A, double(eegData));
        eegData = detrend(eegData);
        
        k = strfind(file_list(i).name,'_');
        label = str2num(file_list(i).name(k(2)+1));
        % period = dataStruct.sequence;
        fs = dataStruct.iEEGsamplingRate;
        [nt,nc] = size(eegData);
        
        subsampLen = floor(fs * 60);            % Num samples in 1 min window
        numSamps = floor(nt / subsampLen);      % Num of 1-min samples
        sampIdx = 1 : subsampLen : numSamps*subsampLen + 1;

        feat = [];
        for l = 1:numSamps
            %% Sample 1-min window
            epoch = eegData(sampIdx(l):sampIdx(l+1)-1,:);
        
            %% Compute Shannon's entropy, spectral edge and correlation matrix
            % Find the power spectrum at each frequency bands
            D = abs(fft(epoch));             % take FFT of each channel
            D(1,:) = 0;                        % set DC component to 0
            D = bsxfun(@rdivide,D,sum(D));     % normalize each channel
            lvl = [0.1 4 8 12 30 70 180];      % frequency levels in Hz
            lseg = round(subsampLen/fs*lvl)+1;         % segments corresponding to frequency bands
    
            dspect = zeros(length(lvl)-1,nc);
            for n=1:length(lvl)-1
                dspect(n,:) = 2*sum(D(lseg(n):lseg(n+1)-1,:));
            end
    
            % Find the Shannon's entropy
            spentropy = -sum(dspect.*log(dspect));
    
            % Find the spectral edge frequency
            sfreq = fs;
            tfreq = 40;
            ppow = 0.5;
    
            topfreq = round(subsampLen/sfreq*tfreq)+1;
            A = cumsum(D(1:topfreq,:));
            B = bsxfun(@minus,A,max(A)*ppow);
            [~,spedge] = min(abs(B));
            spedge = (spedge-1)/(topfreq-1)*tfreq;
    
            % Calculate correlation matrix and its eigenvalues (b/w channels)
            type_corr = 'Pearson';
            C = corr(dspect,'type',type_corr);
            C(isnan(C)) = 0;    % make NaN become 0
            C(isinf(C)) = 0;    % make Inf become 0
            lxchannels = real(sort(eig(C)));
    
            % Calculate correlation matrix and its eigenvalues (b/w freq)
            C = corr(dspect.','type',type_corr);
            C(isnan(C)) = 0;    % make NaN become 0
            C(isinf(C)) = 0;    % make Inf become 0
            lxfreqbands = real(sort(eig(C)));
        
            %% Spectral entropy for dyadic bands
            % Find number of dyadic levels
            ldat = floor(subsampLen/2);
            no_levels = floor(log2(ldat));
            seg = floor(ldat/2^(no_levels-1));
        
            % Find the power spectrum at each dyadic level
            dspect = zeros(no_levels,nc);
            for n=no_levels:-1:1
                dspect(n,:) = 2*sum(D(floor(ldat/2)+1:ldat,:));
                ldat = floor(ldat/2);
            end
        
            % Find the Shannon's entropy
            spentropyDyd = -sum(dspect.*log(dspect));
    
            % Find correlation between channels
            C = corr(dspect,'type',type_corr);
            C(isnan(C)) = 0;    % make NaN become 0
            C(isinf(C)) = 0;    % make Inf become 0
            lxchannelsDyd = sort(eig(C));
        
            %% Fractal dimensions
            no_channels = nc;
            fd = zeros(3,no_channels);
            for n=1:no_channels
                fd(:,n) = wfbmesti(epoch(:,n));
            end
        
            %% Hjorth parameters
            % Activity
            activity = var(epoch);   
            % Mobility
            mobility = std(diff(epoch))./std(epoch);
            % Complexity
            complexity = std(diff(diff(epoch)))./std(diff(epoch))./mobility;
        
            %% Statistical properties
            % Skewness
            skew = skewness(epoch);
            % Kurtosis
            kurt = kurtosis(epoch);
        
            %% Compile all the features
            feat = [feat spentropy(:)' spedge(:)' lxchannels(:)' lxfreqbands(:)' spentropyDyd(:)' ...
                lxchannelsDyd(:)' fd(:)' activity(:)' mobility(:)' complexity(:)' skew(:)' kurt(:)'];
    
        end
        %save('feat.mat', 'feat', '-v7.3');
        feat_train = [feat_train; feat];
    end
    
    % Save feature matrix
    fileName = ['feat_train_', num2str(subj), '.mat'];
    save(fileName, 'feat_train', '-v7.3');
end