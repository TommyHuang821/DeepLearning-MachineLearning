function DNN_net=forwordpropagation_Sheng(this_pat,DNN_net)
% this_pat: input data, n x dim
% Weight: weight for all layer
% B: intercept for all layer
% NumhiddenLayer: number of hidden layer
% af: Aactivation function

%%%%%%%%%%%%%%
% input layer to hidden layer
NumconnectionLayer=numel(DNN_net.DesignDNNLayersize)-1; % how many connection
NumhiddenLayer=NumconnectionLayer-1;
Weight=DNN_net.W;
af=DNN_net.af;

for i=1:NumhiddenLayer
    saf=af{i};
    if i==1
        Z{i}=Weight{i}*this_pat; % input-hidden
    else
        Z{i}=Weight{i}*V{i-1}; % hidden-hidden
    end
    V{i}=saf(Z{i});
end
Z{NumconnectionLayer}=Weight{NumconnectionLayer}*V{NumconnectionLayer-1}; % hidden-output
V{NumconnectionLayer}=Z{NumconnectionLayer};
DNN_net.Z=Z;
DNN_net.V=V;


