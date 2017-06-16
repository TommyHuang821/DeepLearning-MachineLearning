clear all;  clc; close all

load fisheriris
traindata = meas;        
train_out = [ones(50,1);ones(50,1)*2;ones(50,1)*3];  

%% for classification
plot_classification=1; %% if you classification problem is 2-dimension
Nclass=3;
[Ntrain, dim]=size(traindata);
if dim~=2
    plot_classification=0;
end

Train_Label=zeros(Ntrain,Nclass);
for i=1:Nclass
    pos=find(train_out==i);
    Train_Label(pos,i)=1;
end
[Ntrain, dim]=size(traindata);
SizeInputLayer=dim;
SizeOutputLayer=Nclass;
DNN_net.LayerDesign={
   struct('LayerType','Input','LayerName','IL','n_node',SizeInputLayer)                                            % input layer
   struct('LayerType','Hidden','LayerName','H1','n_node',100,'ActF','ELU') 
   struct('LayerType','Output','LayerName','OL','n_node',SizeOutputLayer,'ActF','linear')                                  % Pooling layer
};
DNN_net.option_ActFunction=0.1;
DNN_net.maxIter=100; %%default=100
DNN_net.r=0.01; %%default=0.1, learning rate
DNN_net.LearningApproach='Adam';%% default=SGD'
DNN_net.batchsize=100; %% default=100
DNN_net.isnormalization=1; %% default=0
DNN_net.dropoutFraction=0.1; %% default=0.5

% 1. initial DNN net
DNN_net=Initialization_Net_DNN(DNN_net);
% 2. DNN learning
DNN_net.batchsize=30;
tic 
DNN_net=DNN_Learning_batch(traindata,Train_Label,DNN_net);
toc
% 3. Testing
pred=DNN_Test(traindata,DNN_net);
[~,Label]=max(pred);
Perforamcne=VIndex(train_out,Label');
Perforamcne

figure
plot(DNN_net.CostValue)
title('DNN cost value')
xlabel('Iteration time')

