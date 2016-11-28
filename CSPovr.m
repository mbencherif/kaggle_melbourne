function [SpatialFilter]=CSPovr(EEGDATA,LABELS,CSPnum)

[difflabel] = unique(LABELS);
classnum = length(difflabel);

Filter =[];
% Two classes
if (classnum==2)
    EEG = EEGDATA;
    label = LABELS;
    [W,A] = feature_CSP(EEG,label,2*CSPnum);
    Filter{1} = W;
end
% More than two classes
if (classnum>2)
    for i=1:classnum
        labelsA = difflabel(i);
        EEG = EEGDATA;
        label = LABELS;
        label((label~=labelsA)) = 100;
        [W,A] = feature_CSP(EEG,label,2*CSPnum);
        Filter{i} = W;
    end
end

SpatialFilter =[];
for i=1:length(Filter)
    SpatialFilter = [SpatialFilter;Filter{i}];
end

