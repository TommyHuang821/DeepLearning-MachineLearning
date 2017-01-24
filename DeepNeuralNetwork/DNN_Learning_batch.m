function DNN_net=DNN_Learning_batch(traindata,train_out,DNN_net)


%%%%
[Ntrain dim]=size(traindata);
if Ntrain<=dim %% because traindata must be n x dim
   traindata=traindata';
   [Ntrain dim]=size(traindata);
end
if size(train_out,1)<size(train_out,2) %% because train_out must be n x 1
    train_out=train_out';
end
if size(train_out,1)~=Ntrain
    error ('Number of traindata and train_out is different.');
end 
if isfield(DNN_net,'batchsize')==0
    batchsize=floor(Ntrain/10);
else
    batchsize=DNN_net.batchsize;
end

if isfield(DNN_net,'maxIter')==0
    maxIter=10000;
else
    maxIter=DNN_net.maxIter; %% max of learning iteration, if not converge 
end


%% 1. Zscore for each dimesnsion, standardise the data to mean=0 and standard deviation=1
% input training data
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
% output data, if regression case
SizeOutputLayer=DNN_net.DesignDNNLayersize(end);
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
%% 2. add a bias as an input
bias = ones(Ntrain,1);
Normal_traindata = [Normal_traindata bias];
X=Normal_traindata; %% just simplify the code name
%% 3. DNN Learning
CostValue=zeros(maxIter,1); %% value of cost function 
CostChange=zeros(maxIter,1); %% difference between CostValue of the i-th and (i-1)-th iteration

numbatches = Ntrain / batchsize;
for iter=1:maxIter  
    
   kk = randperm(Ntrain);
   errorvalue=0;
    for l = 1 : numbatches
        batch_x = X(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        batch_out = act(kk((l - 1) * batchsize + 1 : l * batchsize), :);
%         tic
        [DNN_net]=DNN_Learning_ForwardBack(batch_x',batch_out',DNN_net);
%         toc
        pred = DNN_net.V{end};
        if SizeOutputLayer>=2 % classification case
            pred=softmax_Sheng(pred');
            for i=1:batchsize
                errorvalue=errorvalue-log10( pred(i,(batch_out(i,:)==1)));
            end
        else % regression case
            errorvalue=errorvalue+sqrt((pred-batch_out')*(pred-batch_out')');
        end
    end
    CostValue(iter)=errorvalue;
    if iter>=2
        CostChange(iter)=(CostValue(iter)-CostValue(iter-1))^2;
    end
    fprintf('Inter: %d ,learning rate: %f ,Cost: %f, CostChange: %d\n',iter,DNN_net.r,CostValue(iter),CostChange(iter));
    if (iter>=2) &&(( CostChange(iter)< eps*2) | CostValue(iter)< 0.1 )  % break, if converge
        fprintf('converged at epoch: %d\n',iter);
        break 
    end   
end
DNN_net.CostValue=CostValue;
DNN_net.CostChange=CostChange;
DNN_net.Pred_TrainingData=pred;
