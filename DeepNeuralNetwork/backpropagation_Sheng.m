function [DNN_net]=backpropagation_Sheng_2(this_pat,act,DNN_net)

V=DNN_net.V;
Z=DNN_net.Z;
L=DNN_net.L-1;
daf=DNN_net.daf;
W=DNN_net.W;
r=DNN_net.r;

NumconnectionLayer=numel(DNN_net.DesignDNNLayersize)-1; % how many connection
NumhiddenLayer=NumconnectionLayer-1;

 
% output-hidden
e{L}=V{L}-act;
delta_W{L}=r.*(V{L}-act)*V{L-1}';
%
W{L}=W{L}-delta_W{L};
e{L-1}=W{L}'*e{L};
for i=1:NumhiddenLayer
    tmp=(daf(V{L-i}).*e{L-i});
    if i~=NumhiddenLayer
        delta_W{L-i}=r.*tmp *V{L-i-1}';
        W{L-i}=W{L-i}-delta_W{L-i};
        e{L-i-1}=W{L-i}'* tmp;
    else  
        delta_W{L-i}=r.*tmp *this_pat';
        W{L-i}=W{L-i}-delta_W{L-i};
    end     
end
DNN_net.W=W;
    
