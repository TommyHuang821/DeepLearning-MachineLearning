clear all;  clc; close all

% XOR data
% Ntrain=100;
% Ntest=100;
% dim=2;
% genertae data
% traindata= [normrnd(0.5,0.1,Ntrain,dim);...
%            normrnd(-0.4,0.1,Ntrain,dim);...
%            normrnd(-0.2,0.1,Ntrain,1),normrnd(1,0.1,Ntrain,1); ...
%            normrnd(1,0.1,Ntrain,1),normrnd(-0.5,0.1,Ntrain,1)];
% train_out=[ones(Ntrain*2,1);ones(Ntrain*2,1)*2];
% 
% testdata= [normrnd(0.5,0.1,Ntest,dim);...
%            normrnd(-0.4,0.1,Ntest,dim);...
%            normrnd(-0.2,0.1,Ntest,1),normrnd(1,0.1,Ntest,1); ...
%            normrnd(1,0.1,Ntest,1),normrnd(-0.5,0.1,Ntest,1)];
% test_out=[ones(Ntest*2,1);ones(Ntest*2,1)*2];
load ('sampledata')

% for only one out put
[Ntrain dim]=size(traindata);
SizeInputLayer=dim+1;
SizeOutputLayer=1;
DesignDNNLayersize=[SizeInputLayer 20 SizeOutputLayer]; 
AactivationFunction={'sigmoid','linear'}; %% sigmoid,tanh ,ReLU, linear
maxIter=1000;
LearningRate=0.01;
LearningApproach='Adam';%% 'SGD','Momentum','AdaGrad','RMSProp','Adam'


% 1. initial DNN net
DNN_net=Initialization_Net_DNN(DesignDNNLayersize,LearningRate,AactivationFunction,maxIter,LearningApproach);
% 2. DNN learning
DNN_net.batchsize=10;
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
pred=pred*DNN_net.sigma_output+DNN_net.mu_output;
RMSforTest=sqrt(sum((test_out-pred').^2));
subplot(2,1,2)
plot(test_out,'b');
hold on;
plot(pred,'r')
title ([{'Testing Prediction Result'},{ ['RMS:' num2str(RMSforTest)]}])
xlabel('Testing Data')


fprintf('entry any keys to continuous the classification topic')
pause;
%% for classification
plot_classification=1; %% if you classification problem is 2-dimension
Nclass=2;
[Ntrain dim]=size(traindata);
if dim~=2
    plot_classification=0;
end

Train_Label=zeros(Ntrain,Nclass);
for i=1:Nclass
    pos=find(train_out==i);
    Train_Label(pos,i)=1;
end
[Ntest dim]=size(testdata);
Test_Label=zeros(Ntest,Nclass);
for i=1:Nclass
    pos=find(test_out==i);
    Test_Label(pos,i)=1;
end

SizeInputLayer=dim+1;
SizeOutputLayer=Nclass;
DesignDNNLayersize=[SizeInputLayer 20 SizeOutputLayer]; 
AactivationFunction={'sigmoid','linear'}; %% sigmoid,tanh ,ReLU, linear
maxIter=1000;
LearningRate=0.01;
LearningApproach='Adam';%% 'SGD','Momentum','AdaGrad','RMSProp','Adam'

% 1. initial DNN net
DNN_net=Initialization_Net_DNN(DesignDNNLayersize,LearningRate,AactivationFunction,maxIter,LearningApproach);
% 2. DNN learning
DNN_net.batchsize=10;
tic 
DNN_net=DNN_Learning_batch(traindata,Train_Label,DNN_net);
toc
% 3. Testing
pred=DNN_Test(testdata,DNN_net);
pred=softmax_Sheng(pred');
[~,Label]=max(pred');
Perforamcne=VIndex(test_out,Label);
Perforamcne
if plot_classification==1
    tmp=[-2:0.02:2.5]';
    c=0;
    linedata=[];
    for i=1:length(tmp)
        for j=1:length(tmp)
            c=c+1;
            linedata(c,1)=tmp(i);
            linedata(c,2)=tmp(j);
        end
    end
    Normal_testdata=[];
    for i=1:size(testdata,2)
        Normal_testdata(:,i)= (linedata(:,i)-DNN_net.mu_train(i,1)) /DNN_net.sigma_trian(i,1);
    end
    bias = ones(size(linedata,1),1);
    Normal_testdata = [Normal_testdata, bias];
    DNN_test=forwordpropagation_Sheng(Normal_testdata',DNN_net);
    pred=DNN_test.V{end};
     pred=softmax_Sheng(pred');
    [~,Label2]=max(pred');
    
    figure
    COL=jet(Nclass);
    for i=1:Nclass
        hold on
        pos=find(Label2==i);
        plot(linedata(pos,1),linedata(pos,2),'.','color',COL(i,:));
    end
    shapeplot={'o','s','^','x','+','*','v','>','<','p'};
    for i=1:Nclass
        hold on
        pos=find(test_out==i);
        plot(testdata(pos,1),testdata(pos,2),'*','color',COL(i,:));
        pos=find(Label==i);
        plot(testdata(pos,1),testdata(pos,2),['k' shapeplot{i}]);
    end
    title([{'Classification Result'},{['Test accuracy is ' num2str(100*Perforamcne(2,1)) '%']}])
end
