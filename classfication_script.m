clc
clear
close all

load feature_psd_org.mat
load LABEL_train_1.mat

% feature_train = double(real(feat_train));
indices = crossvalind('Kfold', size(feature_train,1), 10);
sum_acc1 = 0.0; sum_acc2 = 0.0; sum_acc3 = 0.0; sum_acc4 = 0.0;
specificity_1 = 0.0; specificity_2 = 0.0; specificity_3 = 0.0; specificity_4 = 0.0;
totauc_1 = 0.0; totauc_2 = 0.0; totauc_3 = 0.0; totauc_4 = 0.0;

for i = 1:10
    testid = (indices == i); trainid = ~testid;
    train_X = feature_train(trainid,:);
    train_Y = LABEL_train(trainid);
    test_X = feature_train(testid,:);
    test_Y = LABEL_train(testid);
    
    % Support Vector Machine
    %options = optimset('maxiter', 10000);
    Mdl = fitcsvm(train_X, train_Y, 'KernelFunction', 'linear');
    pred_1 = predict(Mdl, test_X);
    
    err_1 = sum(test_Y ~= pred_1)/size(test_Y,1);
    disp(['SVM error rate: ', num2str(err_1)]);
    sum_acc1 = sum_acc1 + 1 - err_1;
    
    conMat_1 = confusionmat(test_Y, pred_1);
    spec_1 = conMat_1(2,2)/(conMat_1(2,1)+conMat_1(2,2));
    disp(['SVM specificity: ', num2str(spec_1)]);
    specificity_1 = specificity_1 + spec_1;
    
    [roc_X_1,roc_Y_1,~,auc_1] = perfcurve(test_Y,pred_1,1);
    disp(['SVM AUC: ' num2str(auc_1)]);
    totauc_1 = totauc_1 + auc_1;
    
    % k-Nearest Neighbors
    Mdl = fitcknn(train_X,train_Y,'NumNeighbors',5,'Standardize',1);
    pred_2 = predict(Mdl, test_X);
    
    err_2 = sum(test_Y ~= pred_2)/size(test_Y,1);
    disp(['kNN error rate: ', num2str(err_2)]);
    sum_acc2 = sum_acc2 + 1 - err_2;
    
    conMat_2 = confusionmat(test_Y, pred_2);
    spec_2 = conMat_2(2,2)/(conMat_2(2,1)+conMat_2(2,2));
    disp(['kNN specificity: ', num2str(spec_2)]);   
    specificity_2 = specificity_2 + spec_2;
    
    [roc_X_2,roc_Y_2,~,auc_2] = perfcurve(test_Y,pred_2,1);
    disp(['kNN AUC: ' num2str(auc_2)]);
    totauc_2 = totauc_2 + auc_2;
    
    % Naive Bayes
    Mdl = fitcnb(train_X,train_Y);
    pred_3 = predict(Mdl, test_X);
    
    err_3 = sum(test_Y ~= pred_3)/size(test_Y,1);
    disp(['Naive Bayes error rate: ', num2str(err_3)]);
    sum_acc3 = sum_acc3 + 1 - err_3;
    
    conMat_3 = confusionmat(test_Y, pred_3);
    spec_3 = conMat_3(2,2)/(conMat_3(2,1)+conMat_3(2,2));
    disp(['Naive Bayes specificity: ', num2str(spec_3)]);
    specificity_3 = specificity_3 + spec_3;
    
    [roc_X_3,roc_Y_3,~,auc_3] = perfcurve(test_Y,pred_3,1);
    disp(['Naive Bayes AUC: ' num2str(auc_2)]);
    totauc_3 = totauc_3 + auc_3;
    
    % Random Forest
    t = templateTree('MinLeafSize',5);
    tic
    rusTree = fitensemble(train_X,train_Y,'RUSBoost',500,t,'LearnRate',0.1);
    toc
%     Mdl = TreeBagger(100,train_X,train_Y);
    pred_4 = predict(rusTree, test_X);
%     pred_4 = str2num(cell2mat(pred_4));
    
    err_4 = sum(test_Y ~= pred_4)/size(test_Y,1);
    disp(['Random Forest error rate: ', num2str(err_4)]);
    sum_acc4 = sum_acc4 + 1 - err_4;
    
    conMat_4 = confusionmat(test_Y, pred_4);
    spec_4 = conMat_4(2,2)/(conMat_4(2,1)+conMat_4(2,2));
    disp(['Random Forest specificity: ', num2str(spec_4)]);
    specificity_4 = specificity_4 + spec_4;
    
    [roc_X_4,roc_Y_4,~,auc_4] = perfcurve(test_Y,pred_4,1);
    disp(['Random Forest AUC: ' num2str(auc_4)]);
    totauc_4 = totauc_4 + auc_4;
end

disp(['Average accuracy of SVM: ', num2str(sum_acc1 * 10), '%']);
disp(['Average Specificity of SVM: ', num2str(specificity_1 / 10)]);
disp(['Average auc of SVM: ', num2str(totauc_1 / 10)]);

disp(['Average accuracy of kNN: ', num2str(sum_acc2 * 10), '%']);
disp(['Average Specificity of kNN: ', num2str(specificity_2 / 10)]);
disp(['Average auc of kNN: ', num2str(totauc_2 / 10)]);

disp(['Average accuracy of Naive Bayes: ', num2str(sum_acc3 * 10), '%']);
disp(['Average Specificity of Naive Bayes: ', num2str(specificity_3 / 10)]);
disp(['Average auc of Naive Bayes: ', num2str(totauc_3 / 10)]);

disp(['Total accuracy of Random Forest: ', num2str(sum_acc4 * 10), '%']);
disp(['Average Specificity of Random Forest: ', num2str(specificity_4 / 10)]);
disp(['Average auc of Random Forest: ', num2str(totauc_4 / 10)]);
