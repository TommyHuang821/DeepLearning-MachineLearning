load fisheriris
x = meas(51:end,1:2);        % iris data, 2 classes and 2 features
y = (1:100)'>50;             % versicolor=0, virginica=1
% b = glmfit(x,y,'binomial');  % logistic regression
% p = glmval(b,x,'logit');     % get fitted probabilities for scores

[stat]=LogisticRegression(x,y);

Label=[ones(50,1);ones(50,1)*2];
[X,Y,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(Label,stat.y_hat_probability,2);
plot(X,Y)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by logistic regression')
