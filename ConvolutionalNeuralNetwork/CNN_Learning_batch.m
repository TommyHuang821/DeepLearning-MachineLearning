function CNN_net=CNN_Learning_batch(traindata,train_out,CNN_net)
% traindata: image format (mapsize * mapsize * n_train), first two dimesnion represent a image, third dimension represents the number of training data.
% train_out: Label (n_label * n_train)

[Ntrain]=size(traindata,3);
if isfield(CNN_net.opt,'batchsize')==0
    batchsize=floor(Ntrain/100);
    CNN_net.opt.batchsize=batchsize;
else
    batchsize=CNN_net.opt.batchsize;
end
if isfield(CNN_net.opt,'numepochs')==0
    numepochs=30;
    CNN_net.opt.numepochs=numepochs;
else
    numepochs=CNN_net.opt.numepochs; %% max of learning iteration, if not converge 
end

numbatches = Ntrain / batchsize;
CostValue = zeros(1,numepochs);
CostChange = zeros(1,numepochs);
CNN_net.rL = [];
for iter=1:numepochs  
   kk = randperm(Ntrain);
   tic
   errorvalue=0;
   for l = 1 : numbatches
       batch_x = traindata(:,:,kk((l - 1) * batchsize + 1 : l * batchsize));
       batch_out = train_out(:,kk((l - 1) * batchsize + 1 : l * batchsize));
       
       CNN_net=CNN_feedforward(batch_x,CNN_net);
       CNN_net=CNN_feedbackward(batch_out,CNN_net);
       CNN_net=CNN_UpdateGradients(CNN_net);
       if isempty(CNN_net.rL)
            CNN_net.rL(1) = CNN_net.L;
        end
        CNN_net.rL(end + 1) = 0.99 * CNN_net.rL(end) + 0.01 * CNN_net.L;
        
        tmp=0;
        for i=1:batchsize
            errorvalue=errorvalue-log10( CNN_net.Output((batch_out(:,i)==1),i));
            tmp=tmp-log10( CNN_net.Output((batch_out(:,i)==1),i));
        end
%         CNN_net.rL(end + 1)=tmp/batchsize;
   end
    CostValue(iter)=errorvalue/Ntrain;
    implementtime=toc;
    if iter>=2
        CostChange(iter)=abs(CostValue(iter)-CostValue(iter-1));
    end
    fprintf('epoch: %d/%d ,learning rate: %f ,Cost: %f, CostChange: %d, Implement time: %f\n',iter,numepochs,CNN_net.opt.alpha,CostValue(iter),CostChange(iter), implementtime);
    if (iter>=2) &&(( CostChange(iter)<= eps) | CostValue(iter)< eps )  % break, if converge
        fprintf('converged at epoch: %d\n',iter);
        break 
    end   
end
CNN_net.CostValue=CostValue;
CNN_net.CostChange=CostChange;
