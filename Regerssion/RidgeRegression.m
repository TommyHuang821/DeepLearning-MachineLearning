function [stat]=RidgeRegression(x,y,lamda)
% Ridge Regression by least square approach
% Input:
% x : independent variable  n*dim
% y : dependent variable n*1
% lamda: if regularization is done, the parameter of lamda, default
%         is 1
% Output:
% stat.Beta: estimated coefficients of linear regression
% stat.y_hat: predicted y.
% stat.y: true y.
% stat.RMSE: Root mean square error
% stat.lamda: regularization parameter.



if  nargin < 3
    lamda=1;
end

m=length(y);
x=[ones(m,1),x]; % add bias term


dim=size(x,2);
R=eye(dim);
R(1,1)=0;
Beta_hat=pinv(x'*x+lamda*R)*x'*y; % Least square approach
    
Estimatedy=x*Beta_hat;
RMSE=mean((y-Estimatedy).^2);
stat.y_hat=Estimatedy;
stat.y=y;
stat.x=x;
stat.RMSE=RMSE;
stat.Beta=Beta_hat;
stat.lamda = lamda;