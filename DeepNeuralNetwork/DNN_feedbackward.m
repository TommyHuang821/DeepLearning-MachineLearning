function DNN_net=DNN_feedbackward(this_pat,act,DNN_net)
% this_pat: sizeofinputlayer x n
% act: sizeofoutlayer x n
L=DNN_net.L-1; % total layer
daf=DNN_net.daf; % formula of derived active function
NumconnectionLayer=numel(DNN_net.DesignDNNLayersize)-1; % how many connection
NHiddenLayer=NumconnectionLayer-1;
W=DNN_net.W;
Z=DNN_net.Z;
dW=DNN_net.dW;
dWb=DNN_net.dWb;
V=DNN_net.V;
% back-propagation
e{L}=V{L}-act;
for i=0:NHiddenLayer
    sdaf=daf{L-i};
    tmp=(sdaf(Z{L-i}).*e{L-i});
    if i~=NHiddenLayer
        dW{L-i}=tmp*V{L-i-1}';
        e{L-i-1}=W{L-i}'*tmp;
    else  
        dW{L-i}=tmp*this_pat';
    end     
    dWb{L-i}=mean(tmp,2);
end
DNN_net.dW=dW;
DNN_net.dWb=dWb;
