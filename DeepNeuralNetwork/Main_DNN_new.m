clear all;  clc; close all
load ('sampledata')

% for only one output
% 1. initial DNN net
[Ntrain, dim]=size(traindata);
SizeInputLayer=dim;
SizeOutputLayer=1;
DNN_net.LayerDesign={
   struct('LayerType','Input','LayerName','IL','n_node',SizeInputLayer) 
   struct('LayerType','Hidden','LayerName','H1','n_node',100,'ActF','ELU','option_ActFunction',0.1)  
   struct('LayerType','Hidden','LayerName','H2','n_node',100,'ActF','sigmoid')                                
   struct('LayerType','Output','LayerName','OL','n_node',SizeOutputLayer,'ActF','linear')                                 
};
DNN_net.maxIter=100; %%default=100
DNN_net.r=0.01; %%default=0.1, learning rate
DNN_net.LearningApproach='Adam';%% default=SGD'
DNN_net.batchsize=100; %% default=100
DNN_net.isnormalization=1; %% default=1
DNN_net.dropoutFraction=0; %% default=0.5
DNN_net=Initialization_Net_DNN(DNN_net);

DNN_net=DNN_Learning_batch(traindata,train_out,DNN_net);
% 3. Testing
pred=DNN_Test(testdata,DNN_net);
pred=pred*DNN_net.sigma_output+DNN_net.mu_output;
RMSforTest=sqrt(sum((test_out-pred').^2));


figure(1)
subplot(2,1,1)
plot(DNN_net.CostValue)
title('Learning Cost function')
xlabel('Iteration')
ylabel('RMSE')
subplot(2,1,2)
plot(test_out,'b');
hold on;
plot(pred,'r')
title ([{'Testing Prediction Result'},{ ['RMS:' num2str(RMSforTest)]}])
xlabel('Testing Data')



% 
fprintf('entry any keys to continuous the classification topic \n')
pause;
%% for classification
plot_classification=1; %% if you classification problem is 2-dimension
Nclass=2;
[Ntrain, dim]=size(traindata);
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

% 1. initial DNN net
[Ntrain, dim]=size(traindata);
SizeInputLayer=dim;
SizeOutputLayer=Nclass;
DNN_net.LayerDesign={
   struct('LayerType','Input','LayerName','IL','n_node',SizeInputLayer)                                            % input layer
   struct('LayerType','Hidden','LayerName','H1','n_node',100,'ActF','ELU')  
   struct('LayerType','Hidden','LayerName','H2','n_node',100,'ActF','ELU')                                  % Pooling layer
   struct('LayerType','Output','LayerName','OL','n_node',SizeOutputLayer,'ActF','linear')                                  % Pooling layer
};
DNN_net.maxIter=100; %%default=100
DNN_net.r=0.01; %%default=0.1, learning rate
DNN_net.LearningApproach='Adam';%% default=SGD'
DNN_net.batchsize=100; %% default=100
DNN_net.isnormalization=0; %% default=0
DNN_net.dropoutFraction=0; %% default=0.5
DNN_net=Initialization_Net_DNN(DNN_net);
% 2. DNN learning
tic 
DNN_net=DNN_Learning_batch(traindata,Train_Label,DNN_net);
toc
% 3. Testing
pred=DNN_Test(testdata,DNN_net);
[~,Label]=max(pred);
Perforamcne=VIndex(test_out,Label');



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
    Normal_testdata=linedata;
    for i=1:size(testdata,2)
        Normal_testdata(:,i)= (linedata(:,i)-DNN_net.mu_train(i,1)) /DNN_net.sigma_trian(i,1);
    end
    
    DNN_test=DNN_feedforward(Normal_testdata',[],DNN_net,1);
    pred=DNN_test.LayerDesign{end}.a;
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
