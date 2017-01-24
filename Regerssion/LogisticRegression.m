function [stat]=LogisticRegression(x,y,option)
% Linear Regression by least square approach
% Input:
% x : independent variable  n*dim
% y : class [label: 0 or 1] n*1
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
    maxiter=100;
else
    if isfield(option,'maxiter')
        maxiter=option.maxiter;
    else
        maxiter=100;
    end
end
if option.reg==1
    if isfield(option,'lamda')
        lamda=option.lamda;
    else
        lamda=1;
    end
end

labelindex=unique(y);
m=length(y);
x=[ones(m,1),x];
g = inline('1.0 ./ (1.0 + exp(-z))'); 
% g1=@(z) (1.0 ./ (1.0 + exp(-z)));
dim=size(x(1,:),2);
Beta_hat = zeros(dim,1);
J_value=[];
iter=0;
while iter<=maxiter
    iter=iter+1;
    % Newton's optimal approach 
    z= x*Beta_hat;
    J_value(iter)= mean(-(y.* log(g(z))) - ((ones(m,1)-y).*log(1-g(z))));

    err=g(z)-y;
    invJ=x'*err/m;
    H=(repmat(g(z),1,dim).*x)'*(repmat(1-g(z),1,dim).*x);
    H=H/m;
    
    if option.reg==1; % regualrization
        J_value(iter)= J_value(iter)+(lamda/(2*m))*Beta_hat(2:end)'*Beta_hat(2:end);
        coe=Beta_hat.*ones(dim,1);
        coe(1)=0;
        invJ=invJ+(lamda/m).*coe;
        
        R=eye(dim);
        R(1,1)=0;
        H=H+(lamda/m)*R;
    end
    
    Beta_hat=Beta_hat-inv(H)*invJ;
   
    if (iter>=2 & (abs((J_value(iter)-J_value(iter-1)))<eps))
        break
    end
end
Estimatedy=x*Beta_hat;
P=(1+exp(-Estimatedy)).^(-1);
Estimatedylabel=ones(m,1);
Estimatedylabel(find(Estimatedy<=0.5))=0;

stat.y_hat_probability=P;
stat.y_hat=Estimatedylabel;
stat.cost=J_value;
stat.numiter=iter;
stat.y=y;
stat.x=x;
stat.Beta=Beta_hat;
stat.regularization = option.reg;
if  option.reg==1
    stat.lamda= lamda;
end