function [Center LabeltoCluster J]=kmeans_sheng(data,cluster_n,option)
% cluster_n : number of cluster
% option(1): maximum number of iterations          (default: 100)
% option(2): minimum amount of improvement         (default: 1e-5)
% option(3): info display during iteration         (default: 1)
%   Chih-Sheng (Tommy) Huang, Date: Feb-01-2016

if nargin < 3
    MaxNumberIter=100;
    mincost=1e-5;
    dispiter=1;
else
    if isempty(option(1))==2;MaxNumberIter=100;else MaxNumberIter=option(2);end
    if isempty(option(2))==1;mincost=1e-5;else mincost=option(3);end
    if isempty(option(3))==1;dispiter=1;else dispiter=option(4);end
end

[N Dim]=size(data);
randpos=randperm(N);
len=floor(N/cluster_n);
for c=1:cluster_n
    if c<cluster_n
        pos=randpos((c-1)*len+1:len*c);
    elseif c==cluster_n
        pos=randpos(pos(end)+1:N);
    end
    Center(c,:)=mean(data(pos,:));
end

th=1;
iter=0;
while th>=mincost
    iter=iter+1;
    % updata all data to center of each cluster
    dis_DatatoCenter=[];
    for c=1:cluster_n
        temp_center=Center(c,:);
        temp = bsxfun(@minus, data, temp_center);
        temp = sqrt(sum(temp.^2,2));
        dis_DatatoCenter(:,c)=temp;
    end
    [v LabeltoCluster]=min(dis_DatatoCenter');
    Center_hat=[];
    % update center
    for c=1:cluster_n
        pos=find(LabeltoCluster==c);
        Center_hat(c,:)=mean(data(pos,:));
    end
    J(iter)=sum(sqrt(sum((Center_hat-Center).^2,2)));
    Center=Center_hat;
    if dispiter==1
        if iter==1
            disp(['K-means starting'])
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

