clc;clear
load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% Train a 6cSigm-2s-12cSigm-2s Convolutional neural network 
%% CNN structure Design 
CNN_net.LayerDesign={
   struct('LayerType','I','LayerName','IL')                                            % input layer
   struct('LayerType','C','LayerName','C1','n_map',6,'kernelsize',5,'ActF','sigmoid')  % convolution layer
   struct('LayerType','S','LayerName','S2','scale',2)                                  % Pooling layer
   struct('LayerType','C','LayerName','C3','n_map',12,'kernelsize',5,'ActF','sigmoid') % convolution layer
   struct('LayerType','S','LayerName','S4','scale',2)                                  % Pooling layer
   struct('LayerType','F','LayerName','F5','ActF','sigmoid')                           % FullConnection layer
};
mapsize = size(squeeze(train_x(:, :, 1)));
Outsize = size(train_y,1);
CNN_net=Initialization_Net_CNN(CNN_net,mapsize,Outsize);


CNN_net.opt.alpha=1;
CNN_net.opt.LearningApproach='SGD';
CNN_net.opt.batchsize = 50;
CNN_net.opt.numepochs = 100;


CNN_net=CNN_Learning_batch(train_x, train_y,CNN_net);
[OutputLabel Pred ] = CNN_test(CNN_net, test_x);
% 
% %plot mean squared error for each batch
figure; plot(CNN_net.rL);
% 

[~,OutputLabel] = max(Pred);
[~, TureLabel] = max(test_y);
accuracy = numel(find(OutputLabel == TureLabel)) / length(TureLabel);
