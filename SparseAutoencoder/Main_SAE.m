clc;clear
load('sampledata')
X=traindata;
[Ntrain dim]=size(X);

% Sparse Autoencoder
DesignSAELayersize=[dim,5];
Option_SAE.ActivationFunction='sigmoid';
Option_SAE.LearningRate=0.05;
Option_SAE.maxIter=1000;
Option_SAE.lambda=0.0001; % weight decay parameter 
Option_Sparsity.beta=0.1;  % weight of sparsity penalty term 
Option_Sparsity.sparsityParam=0.01;  % desired average activation of the hidden units
LearningApproach='SGD';% 'SGD','Momentum','AdaGrad','RMSProp';

Net=Initialization_Net_SAE(DesignSAELayersize,Option_SAE,Option_Sparsity,LearningApproach);

Net.batchsize=10;
Net=SAE_Learning_batch(X,Net);