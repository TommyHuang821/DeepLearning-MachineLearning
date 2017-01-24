function [DNN_net]=DNN_Learning_ForwardBack(this_pat,act,DNN_net)


L=DNN_net.L-1; % total layer
af=DNN_net.af; % formula of active function
daf=DNN_net.daf; % formula of derived active function
NumconnectionLayer=numel(DNN_net.DesignDNNLayersize)-1; % how many connection
NHiddenLayer=NumconnectionLayer-1;
W=DNN_net.W;
delta_W=DNN_net.delta_W;
r=DNN_net.r;
LearningApproach=DNN_net.LearningApproach;
SizeOutputLayer=DNN_net.DesignDNNLayersize(end);


% forward
% input layer to hidden layer
for i=1:NumconnectionLayer
    saf=af{i};
    if i==1
        Z{i}=W{i}*this_pat; % input-hidden
    else
        Z{i}=W{i}*V{i-1}; % hidden-hidden & hidden-output
    end
    if (i==NumconnectionLayer) & (SizeOutputLayer>=2) % classification case
        tmp=softmax_Sheng(Z{i}');
        V{i}=tmp';
    else
        V{i}=saf(Z{i});
    end
        
        
end


% back-propagation
e{L}=V{L}-act;
for i=0:NHiddenLayer
    sdaf=daf{L-i};
    tmp=(sdaf(V{L-i}).*e{L-i});
    if i~=NHiddenLayer
        delta_W{L-i}=tmp*V{L-i-1}';
        e{L-i-1}=W{L-i}'*tmp;
    else  
        delta_W{L-i}=tmp*this_pat';
    end     
end

% update
%% https://www.youtube.com/watch?v=UlUGGB7akfE&t=79s&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=15
for i=1:length(delta_W)
    if strcmp(LearningApproach,'SGD')
        W{i}=W{i}-r.*delta_W{i};
    elseif strcmp(LearningApproach,'Momentum')
        W{i}=W{i}.*DNN_net.m-r.*delta_W{i};
    elseif strcmp(LearningApproach,'AdaGrad')
        tmp=(delta_W{i}.^2);
        W{i}=W{i}-r.*delta_W{i}./sqrt(tmp);
    elseif strcmp(LearningApproach,'RMSProp')
        tmp=(delta_W{i}.^2);
        tmp=DNN_net.m*tmp+(1-DNN_net.m)*tmp;
        W{i}=W{i}-r.*delta_W{i}./sqrt(tmp);
    elseif strcmp(LearningApproach,'Adam')
        DNN_net.delta_W{i}=DNN_net.b1*DNN_net.delta_W{i}+(1-DNN_net.b1)*delta_W{i};
        DNN_net.v{i}=DNN_net.b2*DNN_net.v{i}+(1-DNN_net.b2)*delta_W{i}.^2;
        W{i}=W{i}-r.* DNN_net.delta_W{i}./sqrt(DNN_net.v{i});
    end
end
DNN_net.W=W; % update Weight
DNN_net.V=V; % active value
DNN_net.Z=Z; % mapping value
