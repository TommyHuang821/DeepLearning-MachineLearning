clc;clear
load 100;
%%% It's just an example.
%%% I jsut try the ECG signal from MIT-BIH arrhythmia database (subject number:100) to test 1D CNN. 
%%% The label (train_y) doesn't meaningful

% sample rate is 360, take 10second as a trail/pattern as input data, and there are 50 patterns
S=Signal(1:3600*50,1); 
train_x = (reshape(S,[360*10,50]))';
train_y(1:25,1) = 0;
train_y(26:50,1) = 1;

Train_Label=zeros(50,2);
c=0;
for i=[0,1]
    c=c+1;
    pos=find(train_y==i);
    Train_Label(pos,c)=1;
end

%% Train a 6cSigm-2s-12cSigm-2s Convolutional neural network 
%% CNN structure Design 
CNN_net.LayerDesign={
   struct('LayerType','I','LayerName','IL')                                            % input layer
   struct('LayerType','C','LayerName','C1','n_map',10,'kernelsize',11,'ActF','sigmoid')  % convolution layer
   struct('LayerType','S','LayerName','S2','scale',5)                                  % Pooling layer
   struct('LayerType','C','LayerName','C3','n_map',10,'kernelsize',11,'ActF','sigmoid') % convolution layer
   struct('LayerType','S','LayerName','S4','scale',6)                                  % Pooling layer                               % Pooling layer
   struct('LayerType','F','LayerName','F5','ActF','sigmoid')                           % FullConnection layer
};
[Ntrain mapsize] = size(train_x);
Outsize = size(Train_Label,2);
% initial 
CNN_net=Initialization_Net_CNN_1D(CNN_net,mapsize,Outsize);


CNN_net.opt.alpha=1;
CNN_net.opt.LearningApproach='SGD';
CNN_net.opt.batchsize = 10;
CNN_net.opt.numepochs = 1000;


CNN_net=CNN_Learning_batch_1D(train_x, Train_Label,CNN_net);
[OutputLabel Pred ] = CNN_test(CNN_net, train_x);
% 
% %plot mean squared error for each batch
% figure; plot(CNN_net.rL);
% % 
% 
% [~,OutputLabel] = max(Pred);
% [~, TureLabel] = max(test_y);
% accuracy = numel(find(OutputLabel == TureLabel)) / length(TureLabel);
