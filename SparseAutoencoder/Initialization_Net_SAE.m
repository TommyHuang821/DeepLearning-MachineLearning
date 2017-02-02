function Net=Initialization_Net_SAE(DesignSAELayersize,Option_SAE,Option_Sparsity,LearningApproach)

if nargin<4
    LearningApproach='SGD';
end

if nargin<3
    Option_Sparsity=[];
end
if nargin<2
    Option_SAE=[];
end

visibleSize=DesignSAELayersize(1);
hiddenSize=DesignSAELayersize(2);


if isfield(Option_SAE,'ActivationFunction');ActivationFunction=Option_SAE.ActivationFunction;else ActivationFunction='sigmoid';end
if isfield(Option_SAE,'LearningRate');LearningRate=Option_SAE.LearningRate;else LearningRate=0.01;end
if isfield(Option_SAE,'maxIter');maxIter=Option_SAE.maxIter;else maxIter=1000;end
% weight decay parameter 
if isfield(Option_SAE,'lambda');lambda=Option_SAE.lambda;else lambda=0.0001;end

% desired average activation of the hidden units.
% (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
%  in the lecture notes). 
if isfield(Option_Sparsity,'sparsityParam');sparsityParam=Option_Sparsity.sparsityParam;else sparsityParam=0.01;end
% weight of sparsity penalty term 
if isfield(Option_Sparsity,'beta');beta=Option_Sparsity.beta;else beta=0.1;end



W_encoder=rand(hiddenSize,visibleSize);
b_encoder=zeros(hiddenSize,1);
W_decoder=rand(visibleSize,hiddenSize);
b_decoder=zeros(visibleSize,1);

Weight{1}.W=W_encoder;
Weight{1}.b=b_encoder;

Weight{2}.W=W_decoder;
Weight{2}.b=b_decoder;

dWeight{1}.W = zeros(hiddenSize,visibleSize); 
dWeight{2}.W  = zeros(visibleSize,hiddenSize);
dWeight{1}.b  = zeros(hiddenSize,1); 
dWeight{2}.b = zeros(visibleSize,1);


if strcmp(ActivationFunction,'sigmoid')
    af{1}=@(x) (1./(1+exp(-x))); % sigmoid function
    daf{1}=@(x) (1-x).*x; % deviated sigmoid function
elseif strcmp(ActivationFunction,'linear')
    af{1}=@(x) (x); % linear
    daf{1}=@(x) (1); % deviated linear
elseif strcmp(ActivationFunction,'tanh')
    af{1}=@(x) ((exp(x)-exp(-x))./(exp(x)+exp(-x))); % 
    daf{1}=@(x) (1-x.^2); % deviated tanh
elseif strcmp(ActivationFunction,'ReLU') % Rectified linear unit 
    af{1}=@(x) (double(x>=0).*x); 
    daf{1}=@(x) double(x>=0); % deviated
end
% output must be a linear Activation Function
af{2}=@(x) (x); % linear
daf{2}=@(x) (1); % deviated linear


Net=[];
Net.LearningApproach=LearningApproach;
if strcmp(LearningApproach,'Momentum')
    Net.m=0.99;
elseif strcmp(LearningApproach,'RMSProp')
    Net.m=0.9;
end


Net.Weight=Weight;
Net.dWeight=dWeight;
Net.af=af;
Net.daf=daf;
Net.ActivationFunction=ActivationFunction;
Net.LearningRate=LearningRate;
Net.maxIter=maxIter;
Net.lambda=lambda;
Net.Sparsity.sparsityParam=sparsityParam;
Net.Sparsity.beta=beta;

