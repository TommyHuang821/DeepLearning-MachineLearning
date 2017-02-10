clc;clear
load fisheriris
x = meas;        % iris data, 2 classes and 2 features
y = [ones(50,1);ones(50,1)*2;ones(50,1)*3];             % versicolor=0, virginica=1


% [stat]=LogisticRegression(x,y);
[stat]=SoftmaxRegression(x,y);

figure
plot(stat.cost)
xlabel('iteration time')
ylabel('Cost value')
title('Learning Iteration')

figure
plot(y,'r-*')
plot(stat.y_hat,'*')
acc=mean(y'==stat.y_hat);
title (['Predict Result acc=' num2str(acc*100) '%' ])
figure
[X,Y,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(y,stat.y_hat,3);
plot(X,Y)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by logistic regression')
