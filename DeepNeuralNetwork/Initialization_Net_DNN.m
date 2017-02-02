function DNN_net=Initialization_Net_DNN(DesignDNNLayersize,LearningRate,ActivationFunction,maxIter,LearningApproach)

if nargin<5
    LearningApproach='SGD';
end


L=length(DesignDNNLayersize);
W{L-1}={};
delta_W{L-1}={};
Wb{L-1}={};
delta_Wb{L-1}={};
%% forward
for i=1:L-1
    w = (randn(DesignDNNLayersize(i+1),DesignDNNLayersize(i)) - 0.5)/10;
    wb = zeros(DesignDNNLayersize(i+1),1);
    W{i}=w;
    Wb{i}=wb;
    delta_W{i}=zeros(DesignDNNLayersize(i+1),DesignDNNLayersize(i));
    delta_Wb{i}=zeros(DesignDNNLayersize(i+1),1);
end

 %% Aactivation Function
af{L-1}={};
daf{L-1}={};
if length(ActivationFunction)==1
    if strcmp(ActivationFunction,'sigmoid')
        for i=1:L-2
            af{i}=@(x) (1./(1+exp(-x))); % sigmoid function
            daf{i}=@(x) (1-x).*x; % deviated sigmoid function
        end
    elseif strcmp(ActivationFunction,'linear')
        for i=1:L-2
            af{i}=@(x) (x); % linear
            daf{i}=@(x) (1); % deviated linear
        end
    elseif strcmp(ActivationFunction,'tanh')
        for i=1:L-2
            af{i}=@(x) ((exp(x)-exp(-x))./(exp(x)+exp(-x))); % 
            daf{i}=@(x) (1-x.^2); % deviated tanh
        end
    elseif strcmp(ActivationFunction,'ReLU') % Rectified linear unit 
        for i=1:L-2
            af{i}=@(x) (double(x>=0).*x); 
            daf{i}=@(x) double(x>=0); % deviated
        end
    end
    % output must be a linear Activation function
    af{L-1}=@(x) (x); % linear
    daf{L-1}=@(x) (1); % deviated linear
elseif length(ActivationFunction) ~= (L-1)
    error('Number of Activation Function Seting is not equal to Layer number');
else
    for i=1:L-1 
        if strcmp(ActivationFunction{i},'sigmoid')
            af{i}=@(x) (1./(1+exp(-x))); % sigmoid function
            daf{i}=@(x) (1-x).*x; % deviated sigmoid function
        elseif strcmp(ActivationFunction{i},'linear')
            af{i}=@(x) (x); % linear
            daf{i}=@(x) (1); % deviated linear
        elseif strcmp(ActivationFunction{i},'tanh')
            af{i}=@(x) ((exp(x)-exp(-x))./(exp(x)+exp(-x))); % 
            daf{i}=@(x) (1-x.^2); % deviated tanh
        elseif strcmp(ActivationFunction{i},'ReLU') % Rectified linear unit 
            af{i}=@(x) (double(x>=0).*x); 
            daf{i}=@(x) double(x>=0); % deviated
        end
    end
end


DNN_net=[];
DNN_net.LearningApproach=LearningApproach;
if strcmp(LearningApproach,'Momentum')
    DNN_net.m=0.99;
elseif strcmp(LearningApproach,'RMSProp')
    DNN_net.m=0.9;
elseif strcmp(LearningApproach,'Adam')
    for i=1:L-1
        DNN_net.v{i}=zeros(DesignDNNLayersize(i+1),DesignDNNLayersize(i));
        DNN_net.vb{i}=zeros(DesignDNNLayersize(i+1),1);
    end
    DNN_net.b1=0.9;
    DNN_net.b2=0.999;
end


DNN_net.W=W;
DNN_net.Wb=Wb;
DNN_net.delta_W=delta_W;
DNN_net.delta_Wb=delta_Wb;
DNN_net.af=af;
DNN_net.daf=daf;
DNN_net.L=L;
DNN_net.r=LearningRate;
DNN_net.V{1}=0;
DNN_net.ActivationFunctionName=ActivationFunction;
DNN_net.maxIter=maxIter;
DNN_net.DesignDNNLayersize=DesignDNNLayersize;
