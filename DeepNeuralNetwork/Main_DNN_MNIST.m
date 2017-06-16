% 
clc;clear;
load mnist_uint8;
traindata = double(train_x) / 255;
testdata  = double(test_x)  / 255;
train_out = double(train_y);
test_out  = double(test_y);
% for only one out put
% 1. initial DNN net
[Ntrain, dim]=size(traindata);
SizeInputLayer=dim;
SizeOutputLayer=10;
DNN_net.LayerDesign={
   struct('LayerType','Input','LayerName','IL','n_node',SizeInputLayer)                                            % input layer
   struct('LayerType','Hidden','LayerName','H1','n_node',100,'ActF','sigmoid') 
   struct('LayerType','Output','LayerName','OL','n_node',SizeOutputLayer,'ActF','linear')                                  % Pooling layer
};
DNN_net.option_ActFunction=0.1;
DNN_net.maxIter=10; %%default=100
DNN_net.r=0.1; %%default=0.1, learning rate
DNN_net.LearningApproach='Adam';%% default=SGD'
DNN_net.batchsize=1000; %% default=100
DNN_net.isnormalization=0; %% default=0
DNN_net.dropoutFraction=0.5; %% default=0.5


DNN_net=Initialization_Net_DNN(DNN_net);
% 2. DNN learning
tic 

DNN_net=DNN_Learning_batch(traindata,train_out,DNN_net);
toc

figure(1)
subplot(2,1,1)
plot(DNN_net.CostValue)
title('Learning Cost function')
xlabel('Iteration')
ylabel('RMSE')

% 3. Testing
pred=DNN_Test(testdata,DNN_net);
[~,pos]=max(pred);
[~,pos2]=max(test_out');
[ Performance, C ]=VIndex(pos2,pos)