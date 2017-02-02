function DNN_net=forwordpropagation(this_pat,DNN_net)
% this_pat: input data, dim x n
% Weight: weight for all layer
% Wb: intercept for all layer
% NumhiddenLayer: number of hidden layer
% af: Aactivation function

[~,N]=size(this_pat);
NumconnectionLayer=DNN_net.L-1; % how many connection
SizeOutputLayer=DNN_net.DesignDNNLayersize(end);
Weight=DNN_net.W;
Wb=DNN_net.Wb;
af=DNN_net.af;

for i=1:NumconnectionLayer
    saf=af{i};
    if i==1
        Z{i}=Weight{i}*this_pat; % input-hidden
    else
        Z{i}=Weight{i}*V{i-1}; % hidden-hidden
    end
    Z{i}=Z{i}+repmat(Wb{i},1,N);
    if (i==NumconnectionLayer) & (SizeOutputLayer>=2) % classification case
        tmp=softmax_Sheng(Z{i}');
        V{i}=tmp';
    else
        V{i}=saf(Z{i});
    end 
end
DNN_net.Z=Z;
DNN_net.V=V;


