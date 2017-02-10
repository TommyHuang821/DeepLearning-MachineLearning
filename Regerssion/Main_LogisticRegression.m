clc;clear
close all

% iris
load fisheriris
x = meas(51:end,1:4);        % iris data, 2 classes and 2 features
y = (1:100)'>50;             % versicolor=0, virginica=1
Label=[ones(50,1);ones(50,1)*2];

% 2-dimension data
% x= [normrnd(0.5,0.1,100,2);...
%            normrnd(1,0.1,100,2)];
% y=[zeros(100,1);ones(100,1)];
% Label=[ones(100,1);ones(100,1)*2];


option.maxiter=1000;
option.reg=1;
option.lamda=1;
[stat]=LogisticRegression(x,y,option);

figure
plot(stat.cost)
xlabel('iteration time')
ylabel('Cost value')
title('Learning Iteration')

figure
subplot(2,1,1)
hold on
plot(y,'r-*')
plot(stat.y_hat,'o')
acc=mean(y==stat.y_hat);
title (['Predict Result acc=' num2str(acc*100) '%, red: ground truth; blue: predicted result' ])
subplot(2,1,2)
[X,Y,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Label,stat.y_hat_probability,2);
plot(X,Y)
xlabel('False positive rate'); ylabel('True positive rate')
title(['ROC for classification by logistic regression, AUC=' num2str(AUC)])


