function [Net, cost] = SparseAutoencoderCost(data,Net)
                                         
% data: n x dim
% Net structure contains
%   Weight{1}.W: W_encoder: hiddensize x visibleSize 
%   Weight{2}.W: W_decoder: visibleSize x hiddensize
%   Weight{1}.b: b_encoder: visibleSize x 1  
%   Weight{2}.b: b_decoder: hiddensize x 1
%   af: Activation function
%   daf: derived Activation function
%   lambda: weight decay parameter
%   sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
%   beta: weight of sparsity penalty term

Weight=Net.Weight;
af=Net.af;
daf=Net.daf;
lambda=Net.lambda;
sparsityParam=Net.Sparsity.sparsityParam;
beta=Net.Sparsity.beta;

W_encoder=Weight{1}.W;
W_decoder=Weight{2}.W;
b_encoder=Weight{1}.b;
b_decoder=Weight{2}.b;

W_encoder_grad = Weight{1}.W; 
W_decoder_grad = Weight{2}.W;
b_encoder_grad = Weight{1}.b; 
b_decoder_grad = Weight{2}.b; 

[Ntrain,~]=size(data);
%% forward-feed
z2 = W_encoder*data'+repmat(b_encoder,1,Ntrain);
a2=af{1}(z2); % active 
z3 = W_decoder*a2+repmat(b_decoder,1,Ntrain);
a3 = af{2}(z3); % active 

%% cost
% error value
Jcost = (0.5/Ntrain)*sum(sum((a3-data').^2));
% penalty for weight
Jweight = (1/2)*(sum(sum(W_encoder.^2))+sum(sum(W_decoder.^2)));
% Sparse
rho= (1/Ntrain).*sum(a2,2);
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));
%cost fnuction
cost = Jcost+lambda*Jweight+beta*Jsparse;

%% backpropagation
d3=-(data'-a3).*daf{2}(z3);
% bacause sparse constraint, must include this term
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
d2 = (W_decoder'*d3+repmat(sterm,1,Ntrain)).*daf{1}(z2); 

%W_encoder_grad 
W_encoder_grad = W_encoder_grad+d2*data;
W_encoder_grad = (1/Ntrain)*W_encoder_grad+lambda*W_encoder;
%W_decoder_grad  
W_decoder_grad = W_decoder_grad+d3*a2';
W_decoder_grad = (1/Ntrain).*W_decoder_grad+lambda*W_decoder;
%b_encoder_grad
b_encoder_grad = b_encoder_grad+sum(d2,2);
b_encoder_grad = (1/Ntrain)*b_encoder_grad;
%b_decoder_grad 
b_decoder_grad = b_decoder_grad+sum(d3,2);
b_decoder_grad = (1/Ntrain)*b_decoder_grad;
dWeight{1}.W=W_encoder_grad;
dWeight{2}.W=W_decoder_grad;
dWeight{1}.b=b_encoder_grad;
dWeight{2}.b=b_decoder_grad;
Net.dWeight=dWeight;
