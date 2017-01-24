function [Medoid LabeltoCluster J]=Kmedoid(data,cluster_n,option)
% k-mediod clustering
% data: input data, N*dim
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
Medoid=(data(randpos(1:cluster_n),:));


th=1;
iter=0;
while th>=mincost
    iter=iter+1;
    % updata all data to center of each cluster
    dis_DatatoMedoid=[];
    for c=1:cluster_n
        temp_medoid=Medoid(c,:);
        temp = bsxfun(@minus, data, temp_medoid);
        temp = sqrt(sum(temp.^2,2));
        dis_DatatoMedoid(:,c)=temp;
    end
    [v LabeltoCluster]=min(dis_DatatoMedoid');
        
    % update Medoid
    Medoid_hat=[];
    for c=1:cluster_n
        index=find(LabeltoCluster == c);
        mtemp = mean(data(index,:));
        temp = sum([(data - repmat(mtemp,N,1)).^2],2);
        inx=find(temp==min(temp));
        inx=min(inx); %if there are many points with the minimum distance  
        Medoid_hat(c,:)=data(inx,:);
    end
    
    J(iter)= mean(mean(dis_DatatoMedoid));
    Medoid=Medoid_hat;
    if dispiter==1
        if iter==1
            disp(['K-medoid starting'])
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
