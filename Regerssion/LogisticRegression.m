function [stat]=LogisticRegression(x,y,option)
% Logistic Regression by Newton method
% Input:
% x : independent variable  n*dim
% y : class [label: 0 or 1] n*1
% option.maxiter: maximum learning iteration
% option.reg : regularization, 1:yes, default: 1
% option.lamda : regularization parameter, default: 1
% Output:
% stat.Beta: estimated coefficients of linear regression
% stat.y_hat: predicted y.
% stat.y: true y.
%  stat.y_hat_probability: Predict Probability for Binary classification
% stat.cost: learning cost
% stat.numiter: how many iteration find the solution of Beta.
% stat.g: sigmoid function



if  nargin < 3
    maxiter=1000;
    reg=1;
    lamda=1;
else
    if isfield(option,'maxiter'); maxiter=option.maxiter;else maxiter=1000;end
    if isfield(option,'reg'); reg=option.reg;else reg=1;end
    if isfield(option,'lamda'); lamda=option.lamda;else lamda=1;end
end


m=length(y);
x=[ones(m,1),x];
g = inline('1 ./ (1 + exp(-z))'); 
dim=size(x(1,:),2);
Beta_hat = zeros(dim,1);
J_value=zeros(maxiter,1);
iter=1;
while iter<maxiter
    
    z= x*Beta_hat;  
    h=g(z); 

   % % %     cost function/ loss function
    J_value(iter)= mean(-(y.* log(h)) - ((ones(m,1)-y).*log(1-h)));
    if reg==1
       J_value(iter)= J_value(iter)+(lamda/(2*m))*norm(Beta_hat(2:end))^2;
    end
    if (iter>=2 && ((abs((J_value(iter)-J_value(iter-1)))<eps)) || isnan( J_value(iter)))
         J_value=J_value(1:iter-1);
         iter=iter-1;
        break
    end
    
    err=h-y;
    invJ=x'*err/m;
    H=(repmat(h,1,dim).*x)'*(repmat(1-h,1,dim).*x);
    H=H/m;
    if reg==1
        G = (lamda/m).*Beta_hat; G(1) = 0; % extra term for gradient
        L = (lamda/m).*eye(dim); L(1) = 0;% extra term for Hessian
        invJ=invJ+G;
        H=H+L;
    end
    
% %     Newton's optimal approach 
    Beta_hat=Beta_hat-pinv(H)*invJ;
    iter=iter+1; 
end

J_value=J_value(1:iter-1);
Estimatedy=x*Beta_hat;
P=g(Estimatedy);
Estimatedylabel=ones(m,1);
Estimatedylabel((Estimatedy<=0.5))=0;

stat.y_hat_probability=P;
stat.y_hat=Estimatedylabel;
stat.cost=J_value;
stat.numiter=iter;
stat.y=y;
stat.x=x;
stat.Beta=Beta_hat;
stat.g=g;
