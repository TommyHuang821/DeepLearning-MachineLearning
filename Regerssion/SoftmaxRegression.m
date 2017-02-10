function [stat]=SoftmaxRegression(x,y,option)
% Softmax Regression by Gradient Descent
% Input:
% x : independent variable  n*dim
% y : (classlabel numbers for each pattern) n*1
%     classlabel MUST be consecutive from 1 to the number of classes.
% option.LearningRate: Learning rate for gradient descent, default:0.1
% option.maxiter: maximum learning iteration
% option.reg : regularization, 1:yes, default: 1
% option.lamda : regularization parameter, default: 1
% Output:
% stat.Beta: estimated coefficients of linear regression
% stat.y_hat: predicted y.
% stat.y: true y.
%  stat.y_hat_probability: Predict Probability for multiple classification
% stat.cost: learning cost
% stat.numiter: how many iteration find the solution of Beta.
% stat.g: sigmoid function
 %      

if  nargin < 3
    maxiter=1000;
    lr=0.1;
    reg=1;
    lamda=0.01;
else
    if isfield(option,'maxiter'); maxiter=option.maxiter;else maxiter=1000;end
    if isfield(option,'LearningRate'); lr=option.LearningRate;else lr=0.1;end
    if isfield(option,'reg'); reg=option.reg;else reg=1;end
    if isfield(option,'lamda'); lamda=option.lamda;else lamda=0.1;end
end

[m dim]=size(x);
labelindex=unique(y);
N_label=length(labelindex);
x=[ones(m,1),x];
dim=dim+1;
Beta_hat = rand(dim,N_label);
GD=zeros(dim,N_label);
J_value=zeros(maxiter,1);
iter=1;
while iter<maxiter+1
    z= x*Beta_hat;
    h=exp(z)./repmat(sum(exp(z),2),1,N_label);% softmax
    
    tmp=0;
    for il=1:N_label
        tmp=tmp+sum(log(h((y==il),il)));
    end
    tmp=-tmp/m;
    J_value(iter)=tmp;
     if reg==1
       J_value(iter)= J_value(iter)+(lamda/2)*norm(Beta_hat(2:end,:))^2;
    end
    if (iter>=2 && ((abs((J_value(iter)-J_value(iter-1)))<eps)) || isnan( J_value(iter)))
        break
    end

    for il=1:N_label
        pos=double(y==il);
        GD(:,il)=-sum(x.*repmat((pos-h(:,il)),1,dim))'/m;
        if reg==1
            G = lamda*Beta_hat(:,il); % extra term for gradient
            G(1)=0;
            GD(:,il)=GD(:,il)+G;
        end
    end
    Beta_hat=Beta_hat-lr*GD;
    iter=iter+1;
end

z= x*Beta_hat;
EstimatedSoftmax_Probability=exp(z)./repmat(sum(exp(z),2),1,N_label);% softmax
[~,Estimatedylabel]=max(EstimatedSoftmax_Probability');

stat.y_hat_probability=EstimatedSoftmax_Probability;
stat.y_hat=Estimatedylabel;
stat.cost=J_value;
stat.numiter=iter;
stat.y=y;
stat.x=x;
stat.Beta=Beta_hat;
