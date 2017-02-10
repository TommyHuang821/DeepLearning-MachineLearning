function [stat]=LinearRegression(x,y)
% Linear Regression by least square approach
% Input:
% x : independent variable  n*dim
% y : dependent variable n*1
% Output:
% stat.Beta: estimated coefficients of linear regression
% stat.y_hat: predicted y.
% stat.y: true y.
% stat.RMSE: Root mean square error
% stat.regularization: does this train implement the regularization.
% stat.lamda: regularization parameter.
 

m=length(y);
x=[ones(m,1),x];

 
Beta_hat=pinv(x'*x)*x'*y; % Least square approach

    
Estimatedy=x*Beta_hat;
RMSE=mean((y-Estimatedy).^2);
stat.y_hat=Estimatedy;
stat.y=y;
stat.x=x;
stat.RMSE=RMSE;
stat.Beta=Beta_hat;
