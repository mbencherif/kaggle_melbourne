clc
clear
close all

load feat_train_3.mat
load LABEL_train_3.mat

feature_train = double(real(feat_train));
indices = crossvalind('Kfold', size(feature_train,1), 10);

sum_acc4 = 0.0;
specificity = 0.0;

thres = 0.5;

for i = 1:1
    testid = (indices == i); trainid = ~testid;
    train_X = feature_train(trainid,:);
    train_Y = LABEL_train(trainid);
    test_X = feature_train(testid,:);
    test_Y = LABEL_train(testid);
    
    % Regularized Binomial Regression
    [B1,FitInfo] = lassoglm(train_X,train_Y,'binomial','CV',10,'RelTol',5e-2,'Options',statset('UseParallel',true));
    test_pred = glmval(B1,test_X,'logit');
    pred_4 = test_pred > thres;
    err_4 = sum(test_Y ~= pred_4)/size(test_Y,1);
    disp(['Lasso GLM error rate: ', num2str(err_4)]);
    conMat_4 = confusionmat(test_Y, pred_4);
    spec_4 = conMat_4(2,2)/(conMat_4(2,1)+conMat_4(2,2));
    disp(['Lasso GLM specificity: ', num2str(spec_4)]);
    sum_acc4 = sum_acc4 + 1 - err_4;
    specificity_4 = specificity_4 + spec_4;
    
    [roc_X,roc_Y,~,auc] = perfcurve(test_Y,pred_4,1);    
    disp(['Trained LassoGLM with training data has an AUC of ' num2str(auc) '.'])
end

disp(['Total accuracy of Lasso GLM: ', num2str(sum_acc4 * 10), '%']);
disp(['Average Specificity of Lasso GLM: ', num2str(specificity_4 / 10)]);

save('lassoGLM_result.mat', 'B1', 'FitInfo', 'train_X', 'train_Y', 'test_X', 'test_Y', 'pred_4', 'roc_X', 'roc_Y')