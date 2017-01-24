function [Center U J]=FuzzyCMean(data,cluster_n,option)
% cluster_n : number of cluster
% option(1): exponent for the matrix U (weight)    (default: 2.0)
% option(2): maximum number of iterations          (default: 100)
% option(3): minimum amount of improvement         (default: 1e-5)
% option(4): info display during iteration         (default: 1)
%   Chih-Sheng (Tommy) Huang, Date: Feb-01-2016

if nargin < 3
    m=2; % m: exponent for the matrix U (weight) 
    MaxNumberIter=100;
    mincost=1e-5;
    dispiter=1;
else
    if isempty(option(1))==1;m=2;else m=option(1);end
    if isempty(option(2))==2;MaxNumberIter=100;else MaxNumberIter=option(2);end
    if isempty(option(3))==1;mincost=1e-5;else mincost=option(3);end
    if isempty(option(4))==1;dispiter=1;else dispiter=option(4);end
end

[N Dim]=size(data);

% initital membership
u=rand(N,cluster_n);
u = bsxfun(@rdivide, u, sum(u,2));
th=1;
iter=0;
while th>=mincost
    iter=iter+1;
    % update center
    for c=1:cluster_n
        temp1 = bsxfun(@times, data, u(:,c).^m);
        temp1 = sum(temp1) ; 
        temp2 = sum(u(:,c).^m) ; 
        Center(c,:)=temp1/temp2;
    end
    % updata all data to center of each cluster
    dis_DatatoCenter=[];
    for c=1:cluster_n
        temp_center=Center(c,:);
        temp = bsxfun(@minus, data, temp_center);
        temp = sqrt(sum(temp.^2,2));
        dis_DatatoCenter(:,c)=temp;
    end
    % objective function
    J(iter)=sum(sum((u.^m).*(dis_DatatoCenter.^2)));
    % updata the membership
    for c=1:cluster_n
    %     dis_DatatoCenter./ repmat(dis_DatatoCenter(:,c),1,cluster_n);
        membership = bsxfun(@rdivide, dis_DatatoCenter, dis_DatatoCenter(:,c));
        membership=membership.^(2/(m-1));
        u=u+membership;
    end
    u=u.^(-1); % membership
    u = bsxfun(@rdivide, u, sum(u,2));
    if dispiter==1
        if iter==1
            disp(['Fuzzy C-means starting'])
        end
        disp(['Iternation : ' num2str(iter) ', cost fuction J = ' num2str(J(iter))]);
    end
    if iter>=2
        th=(J(iter)-J(iter-1))^2;
    end
    if iter >=MaxNumberIter
        break
    end
end
U=u;


