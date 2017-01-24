function [stat]=LinearRegression(x,y,option)
% Linear Regression by least square approach
% Input:
% x : independent variable  n*dim
% y : dependent variable n*1
% option.reg: regularization (1) or not (0) default:0
% option.lamda: if regularization is done, the parameter of lamda, default
%         is 1
% Output:
% stat.Beta: estimated coefficients of linear regression
% stat.y_hat: predicted y.
% stat.y: true y.
% stat.RMSE: Root mean square error
% stat.regularization: does this train implement the regularization.
% stat.lamda: regularization parameter.
 



if  nargin < 3
    option.reg=0;
end
if option.reg==1
    if isfield(option,'lamda')
        lamda=option.lamda;
    else
        lamda=1;
    end
end

m=length(y);
x=[ones(m,1),x];

% by check the covariance matrix is singular or not.
if det(x'*x)<=eps
    option.reg=1;
    lamda=1;
end

if option.reg==0;    
    Beta_hat=pinv(x'*x)*x'*y; % Least square approach
elseif option.reg==1;
    dim=size(x,2);
    R=eye(dim);
    R(1,1)=0;
    Beta_hat=pinv(x'*x+lamda*R)*x'*y;
end
    
Estimatedy=x*Beta_hat;
RMSE=mean((y-Estimatedy).^2);
stat.y_hat=Estimatedy;
stat.y=y;
stat.x=x;
stat.RMSE=RMSE;
stat.Beta=Beta_hat;
stat.regularization = option.reg;
if  option.reg==1
    stat.lamda= lamda;
end