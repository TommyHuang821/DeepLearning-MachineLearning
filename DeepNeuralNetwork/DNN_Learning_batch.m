function DNN_net=DNN_Learning_batch(traindata,train_out,DNN_net)

%%%% check inputdata
[Ntrain, dim]=size(traindata);
[Ntrain2, dim2]=size(train_out); %% train_out must be (n x class) or (n x 1)
if Ntrain~=Ntrain2 & Ntrain2==dim
   traindata=traindata';
   [Ntrain, dim]=size(traindata);
end
if Ntrain~=Ntrain2
    error ('Number of traindata and train_out is different.');
end 

%%% batch learning
batchsize=DNN_net.batchsize;
%%% normalization for each dimension
isnormalization=DNN_net.isnormalization;
%%% max of learning iteration (epoch), if not converge 
maxIter=DNN_net.maxIter;

%% 1. Zscore for each dimesnsion, standardise the data to mean=0 and standard deviation=1
% input training data
if isnormalization==1
    Normal_traindata=traindata;
    mu_train=zeros(dim,1);
    sigma_trian=zeros(dim,1);
    for i=1:dim
        mu_train(i,1) = mean(traindata(:,i));
        sigma_trian(i,1) = std(traindata(:,i));
        Normal_traindata(:,i) = (traindata(:,i) - mu_train(i,1)) / sigma_trian(i,1);
    end
    DNN_net.mu_train=mu_train;
    DNN_net.sigma_trian=sigma_trian;
    X=Normal_traindata; %% just simplify the code name
else
    X=traindata;
end

SizeOutputLayer=DNN_net.LayerDesign{end}.n_node;
% output data normalization for only regression case
if SizeOutputLayer>=2
    act=train_out;
    DNN_net.mu_output='';
    DNN_net.sigma_output='';
else
    act=train_out;
    mu_output=mean(act);
    sigma_output=std(act);
    act=(act-mu_output)./sigma_output;
    DNN_net.mu_output=mu_output;
    DNN_net.sigma_output=sigma_output;
end

%% 3. DNN Learning
CostValue=zeros(maxIter,1); %% value of cost function 
CostChange=zeros(maxIter,1); %% difference between CostValue of the i-th and (i-1)-th iteration

numbatches = Ntrain / batchsize;
for iter=1:maxIter  
   kk = randperm(Ntrain);
   errorvalue=0;
    for l = 1 : numbatches
%         pos=find(kk==l);
        batch_x = X(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_out = act(kk((l - 1) * batchsize + 1 : l * batchsize), :);
       
        DNN_net=DNN_feedforward(batch_x',batch_out',DNN_net);
        DNN_net=DNN_feedbackward(DNN_net);
        [DNN_net]=DNN_UpdateGradients(DNN_net);
         
        pred =DNN_net.LayerDesign{end}.a;
        if DNN_net.LayerDesign{end}.n_node~=size(pred,2)
            pred=pred';
        end
        
        if SizeOutputLayer>=2 % classification case
            for i=1:batchsize
                errorvalue=errorvalue-log10( pred(i,(batch_out(i,:)==1)));
            end
        else % regression case
            errorvalue=errorvalue+sqrt((pred-batch_out)'*(pred-batch_out));
        end
    end
%     toc
    CostValue(iter)=errorvalue/Ntrain;
    if iter>=2
        CostChange(iter)=abs(CostValue(iter)-CostValue(iter-1));
    end
    fprintf('Iter: %d/%d ,learning rate: %f ,Cost: %f, CostChange: %d\n',iter,maxIter,DNN_net.r,CostValue(iter),CostChange(iter));
    if (iter>=2) &&(( CostChange(iter)<= eps) || CostValue(iter)<= eps )  % break, if converge
        fprintf('converged at epoch: %d\n',iter);
        break 
    end   
end
DNN_net.CostValue=CostValue;
DNN_net.CostChange=CostChange;
DNN_net.Pred_TrainingData=pred;
