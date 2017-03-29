function [DNN_net]=DNN_UpdateGradients(DNN_net)

r=DNN_net.r;
LearningApproach=DNN_net.LearningApproach;

W=DNN_net.W;
Wb=DNN_net.Wb;
delta_W=DNN_net.delta_W;
dW=DNN_net.dW;
delta_Wb=DNN_net.delta_Wb;
dWb=DNN_net.dWb;

% update
%% https://www.youtube.com/watch?v=UlUGGB7akfE&t=79s&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=15
for i=1:length(delta_W)
    if strcmp(LearningApproach,'SGD')
        delta_W{i}=dW{i};
        delta_Wb{i}=dWb{i};
    elseif strcmp(LearningApproach,'Momentum')
        m=DNN_net.m;
        delta_W{i}= m *delta_W{i}+(1-m).*dW{i};
        delta_Wb{i}= m *delta_Wb{i}+(1-m).*dWb{i};
    elseif strcmp(LearningApproach,'AdaGrad')
        g2=(dW{i}.^2);
        delta_W{i}=dW{i}./sqrt(g2);
        g2b=(dWb{i}.^2);
        delta_Wb{i}=dWb{i}./sqrt(g2b);
    elseif strcmp(LearningApproach,'RMSProp')
        g2=(dW{i}.^2);
        DNN_net.v{i}=DNN_net.m *DNN_net.v{i}+(1-DNN_net.m)*g2;
        delta_W{i}=dW{i}./sqrt(DNN_net.v{i}+10^-8); 
        
        g2b=(dWb{i}.^2);
        DNN_net.vb{i}=DNN_net.m *DNN_net.vb{i}+(1-DNN_net.m)*g2b;
        delta_Wb{i}=dWb{i}./sqrt( DNN_net.vb{i}+10^-8);  
    elseif strcmp(LearningApproach,'Adam')
        b1=DNN_net.b1;
        b2=DNN_net.b2;
        
        g=dW{i};
        g2=(dW{i}.^2);
        DNN_net.mt{i}=b1 *DNN_net.mt{i}+(1-b1).*g;
        DNN_net.v{i}=b2 *DNN_net.v{i}+(1-b2)*g2;
        delta_W{i}= DNN_net.mt{i}./(sqrt(DNN_net.v{i})+10^-8);       
        
        gb=dWb{i};
        g2b=(dWb{i}.^2);
        DNN_net.mtb{i}=b1 *DNN_net.mtb{i}+(1-b1).*gb;
        DNN_net.vb{i}=b2 *DNN_net.vb{i}+(1-b2)*g2b;
        delta_Wb{i}= DNN_net.mtb{i}./(sqrt(DNN_net.vb{i})+10^-8);          
    end
    W{i}=W{i}-r.*delta_W{i};
    Wb{i}=Wb{i}-r.*delta_Wb{i};
end

DNN_net.delta_W=delta_W;
DNN_net.delta_Wb=delta_Wb;
DNN_net.W=W;
DNN_net.Wb=Wb;
DNN_net.dW=dW;
DNN_net.dWb=dWb;
