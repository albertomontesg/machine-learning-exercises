%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Exercise Set 4- Problem 1
%
% Overfitting with non-linear SVMs
%
% Skeleton - This is a skeleton provided as a guide to solve problem 1 
% in exercise set 4 for Machine Learning-Autumn-Semester-2015
%
% IMPORTANT!!!!
%
% When %INSERT CODE HERE% appears there is missing code, to be filled
% by you, the student. ;)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1a) and 1b) Generate means
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% simulation parameters
n_train = 100;
n_test = 10000;  
mu_1   = [1 0]';
mu_2 = [0 1]';
I = eye(2);
rng(7); %we set seed so as not to be able to reproduce results

% we simulate 10 means for each class
ms_1   =  mvnrnd(mu_1, I, 10);
ms_2 =  mvnrnd(mu_2,I, 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1c), 1d) and 1e) Generate 100 from each class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we sample the index of the mean vectors m_k
% that we will use for each realization
indx_1_train   = randsample(10, n_train, true);
indx_2_train = randsample(10,n_train, true);
indx_1_test   = randsample(10, n_test, true);
indx_2_test = randsample(10,n_test, true);

% sample training and test realizations for classes 1 and 2
X_1_train   = normrnd( %INSERT CODE HERE%, 1/5);
X_2_train   = normrnd( %INSERT CODE HERE%, 1/5);
X_1_test    = normrnd( %INSERT CODE HERE%, 1/5);
X_2_test    = normrnd( %INSERT CODE HERE%, 1/5);

% construct the final training and test data, predictors and response
% variable
X_train = [X_1_train;X_2_train];
X_test = [X_1_test;X_2_test];
class_train = [ones(n_train,1)*2;ones(n_train,1)*3];
class_test = [ones(n_test,1)*2;ones(n_test,1)*3];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Fit Linear SVM to X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2a) Fit models for C=0.02,1 and 1,000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_box_C02   = fitcsvm(%INSERT CODE HERE%);
model_box_C1    = fitcsvm(%INSERT CODE HERE%);
model_box_C1000 = fitcsvm(%INSERT CODE HERE%);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2b) Visualize models. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
graphSVM(model_box_C02,X_train,class_train);
graphSVM(model_box_C1,X_train,class_train);
graphSVM(model_box_C1000,X_train,class_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2c) Estimate training and
% generalization error for
% different values of penalty term C.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create vector of Cs we will try
maxC = 1000
minC = 0.01
numCs = 50;
logCs = %INSERT CODE HERE%
Cs = exp(logCs);

%initialize error vectors
err_train_box = ones(numCs,1)*-1000;
err_pred_box = ones(numCs,1)*-1000;

%fit SVMs for different C values
for i =1:numCs
    % train SVM
    model_box = fitcsvm(%INSERT CODE HERE%);
    % predict for train and test data
    pred_train_box = predict(%INSERT CODE HERE%);
    pred_test_box = predict(%INSERT CODE HERE%);
    % calculate error for train and test data
    err_train_box(i) = %INSERT CODE HERE%;
    err_test_box(i) = %INSERT CODE HERE%;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2d) Graphh the training and generalization 
% error versus the penalty parameter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot training and test error against log-penalty
plot(logCs, err_train_box,'b.--',logCs,err_test_box,'rx--')
title('Error vs. penalty - linear kernel')
xlabel('log(C)') 
ylabel('error')
legend('training err.','gen. err. ')

%Can you observe any overfitting going on for different values of $C$?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) Fit polynomial-order-4-SVM to X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3a) Fit models for C=0.02,1 and 1,000
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_poly_C02 = fitcsvm(%INSERT CODE HERE%);
model_poly_C1 = fitcsvm(%INSERT CODE HERE%);
model_poly_C1000 = fitcsvm(%INSERT CODE HERE%);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3b) Visualize models. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
graphSVM(model_poly_C02,X_train,class_train);
graphSVM(model_poly_C1,X_train,class_train);
graphSVM(model_poly_C1000,X_train,class_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3c) Estimate training and
% generalization error for
% different values of penalty term C.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize error vectors
err_train_poly = ones(numCs,1)*-1000;
err_pred_poly = ones(numCs,1)*-1000;

%fit SVMs for different C values
for i =1:numCs
    % train SVM
    model_poly = fitcsvm(%INSERT CODE HERE%);
    pred_train_poly = predict(%INSERT CODE HERE%);
    pred_test_poly = predict(%INSERT CODE HERE%);
    % calculate error for train and test data
    err_train_poly(i) = %INSERT CODE HERE%;
    err_test_poly(i) = %INSERT CODE HERE%;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3d) Graphh the training and generalization 
% error versus the penalty parameter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot training and test error against log-penalty
plot(logCs, err_train_poly,'b.--',logCs,err_test_poly,'rx--')
title('Error vs. penalty - polynomial kernel')
xlabel('log(C)') 
ylabel('error')
legend('training err.','gen. err. ')

%Can you observe any overfitting going on for different values of $C$?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4) Fit RBF-kernel-SVM to X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_gauss_C02 = fitcsvm(%INSERT CODE HERE%);
model_gauss_C1 = fitcsvm(%INSERT CODE HERE%);
model_gauss_C1000 = fitcsvm(%INSERT CODE HERE%);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4b) Visualize models. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
graphSVM(model_gauss_C02,X_train,class_train);
graphSVM(model_gauss_C1,X_train,class_train);
graphSVM(model_gauss_C1000,X_train,class_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4c) Estimate training and
% generalization error for
% different values of penalty term C.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize error vectors
err_train_gauss = ones(numCs,1)*-1000;
err_pred_gauss = ones(numCs,1)*-1000;

%fit SVMs for different C values
for i =1:numCs
    % train SVM
    model_gauss = fitcsvm(%INSERT CODE HERE%);
    % predict for train and test data
    pred_train_gauss = predict(%INSERT CODE HERE%);
    pred_test_gauss = predict(%INSERT CODE HERE%);
    % calculate error for train and test data
    err_train_gauss(i) = %INSERT CODE HERE%;
    err_test_gauss(i) = %INSERT CODE HERE%;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4d) Graphh the training and generalization 
% error versus the penalty parameter.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot training and test error against log-penalty
plot(logCs, err_train_gauss,'b.--',logCs,err_test_gauss,'rx--')
title('Error vs. penalty - gaussian kernel')
xlabel('log(C)') 
ylabel('error')
legend('training err.','gen. err. ')


%Can you observe any overfitting going on for different values of $C$?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3 c. Comparison of behavior for 3 kernels as a function of penalty parameter C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot generalization errors on one graph

plot(logCs, err_test_box,'b.--',logCs,err_test_poly,'rx--', logCs, err_test_gauss, 'g*--')
title('Generalization error vs. penalty')
xlabel('log(C)') 
ylabel('error')
legend('linear','polynomial','RBF')




