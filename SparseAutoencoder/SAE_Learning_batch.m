function Net=SAE_Learning_batch(X,Net)

[Ntrain,~]=size(X);
if isfield(Net,'batchsize')==0
    batchsize=floor(Ntrain/10);
else
    batchsize=Net.batchsize;
end

CostValue=zeros(Net.maxIter,1); %% value of cost function 
CostChange=zeros(Net.maxIter,1); %% difference between CostValue of the i-th and (i-1)-th iteration
numbatches = Ntrain / batchsize;
lr=Net.LearningRate;

for iter=1:Net.maxIter  
   kk = randperm(Ntrain);
   errorvalue=0;
    for l = 1 : numbatches
        batch_x = X(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        [Net, cost] = SparseAutoencoderCost(batch_x,Net);
        
        % type of Gradient Descent 
        for ip=1:length(Net.Weight)
            if strcmp(Net.LearningApproach,'SGD')
                Net.Weight{ip}.W= Net.Weight{ip}.W-lr.*Net.dWeight{ip}.W;
                Net.Weight{ip}.b= Net.Weight{ip}.b-lr.*Net.dWeight{ip}.b;
            elseif strcmp(Net.LearningApproach,'Momentum')
                Net.Weight{ip}.W= Net.Weight{ip}.W*Net.m-lr.*Net.dWeight{ip}.W;
                Net.Weight{ip}.b= Net.Weight{ip}.b*Net.m-lr.*Net.dWeight{ip}.b;
            elseif strcmp(Net.LearningApproach,'AdaGrad')
                tmp=(Net.dWeight{ip}.W.^2);
                Net.Weight{ip}.W=Net.Weight{ip}.W-lr.*Net.dWeight{ip}.W./sqrt(tmp);
                tmp=(Net.dWeight{ip}.b.^2);
                Net.Weight{ip}.b=Net.Weight{ip}.b-lr.*Net.dWeight{ip}.b./sqrt(tmp);
            elseif strcmp(Net.LearningApproach,'RMSProp')
                tmp=(Net.dWeight{ip}.W.^2);
                tmp=Net.m*tmp+(1-Net.m)*tmp;
                Net.Weight{ip}.W=Net.Weight{ip}.W-lr.*Net.dWeight{ip}.W./sqrt(tmp);
                tmp=(Net.dWeight{ip}.b.^2);
                tmp=Net.m*tmp+(1-Net.m)*tmp;
                Net.Weight{ip}.b=Net.Weight{ip}.b-lr.*Net.dWeight{ip}.b./sqrt(tmp);
            end
        end

        errorvalue=errorvalue+cost;
    end
    CostValue(iter)=errorvalue/numbatches;
    if iter>=2
        CostChange(iter)=(CostValue(iter)-CostValue(iter-1))^2;
    end
    fprintf('Iter: %d/%d ,learning rate: %f ,Cost: %f, CostChange: %d\n',iter,Net.maxIter,Net.LearningRate,CostValue(iter),CostChange(iter));
    if (iter>=2) &&(( CostChange(iter)< eps*2) | CostValue(iter)< 0.1 )  % break, if converge
        fprintf('converged at epoch: %d\n',iter);
        break 
    end   
end
Net.CostValue=CostValue;
Net.CostChange=CostChange;
