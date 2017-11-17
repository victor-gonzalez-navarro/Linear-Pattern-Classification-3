function Predict_test = predict_parzen(X_train,Labels_train,N_classes,h,X_test)
%% Parzen classifier with gaussian window
% X_train: matrix, rows contain train vectors
% Labels_train: Assumed to be 0,2,....N_classes-1
% N_classes: Number of different classes
% h: Window width parameter
% X_test: matrix, rows contain vectors to be labeled
% Predict_test: Labels given to the test set
% MC March 2016
%% Intialitation parameters
Eps=0.001;           %Constant for matrix regularization
d=size(X_train,2);  % Number of features
N_test=size(X_test,1);  % Number of elements to be labeled in the test set.
%% Test Labels
Score_test=zeros(N_test,N_classes);  %Score matrix
K0=1/sqrt((2*pi)^d);
for i_class=0:N_classes-1
    index=find(Labels_train==i_class);
    C=cov(X_train(index,:),1);
    if cond(C)>1/Eps
        C=C+trace(C)*Eps*eye(d)/d;
    end
    n=length(index);
    hn=h/sqrt(n);
    K=K0/(hn^d);
    K=K/n;
    K=K/sqrt(det(C));
    C=0.5*inv(C);
    for i_test=1:N_test
        x1=X_test(i_test,:);
        for i_train=1:length(index)
            x2=X_train(index(i_train),:);  %Gaussian window center
            x3=(x1-x2)/hn;
            Score_test(i_test,i_class+1)=Score_test(i_test,i_class+1)+exp(-x3*C*x3');
        end
        Score_test(i_test,i_class+1)=K*Score_test(i_test,i_class+1);
    end
end
%% Scores are converted to labels
[~,Index]=max(Score_test,[],2);
Predict_test=Index-1;
end