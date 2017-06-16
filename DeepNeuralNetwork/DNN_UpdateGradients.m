function [DNN_net]=DNN_UpdateGradients(DNN_net)

r=DNN_net.r;
LearningApproach=DNN_net.LearningApproach;

%% https://www.youtube.com/watch?v=UlUGGB7akfE&t=79s&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=15
for iL= 1 : numel(DNN_net.LayerDesign)   %  layer
    tmpLayerType=DNN_net.LayerDesign{iL}.LayerType;   
    if strcmp(tmpLayerType,'Hidden') || strcmp(tmpLayerType,'Output')
         if strcmp(LearningApproach,'SGD')
            DNN_net.LayerDesign{iL}.delta_W=DNN_net.LayerDesign{iL}.dW;
            DNN_net.LayerDesign{iL}.delta_Wb=DNN_net.LayerDesign{iL}.dWb;
        elseif strcmp(LearningApproach,'Momentum')
            m=DNN_net.optimal.m;    
            DNN_net.LayerDesign{iL}.delta_W= m *DNN_net.LayerDesign{iL}.delta_W+(1-m).*DNN_net.LayerDesign{iL}.dW;
            DNN_net.LayerDesign{iL}.delta_Wb= m *DNN_net.LayerDesign{iL}.delta_Wb+(1-m).*DNN_net.LayerDesign{iL}.dWb;
        elseif strcmp(LearningApproach,'AdaGrad')
            g2=(DNN_net.LayerDesign{iL}.dW.^2);
            DNN_net.LayerDesign{iL}.delta_W=DNN_net.LayerDesign{iL}.dW./sqrt(g2);
            g2b=(DNN_net.LayerDesign{iL}.dWb.^2);
            DNN_net.LayerDesign{iL}.delta_Wb=DNN_net.LayerDesign{iL}.dWb./sqrt(g2b);
        elseif strcmp(LearningApproach,'RMSProp')
            m=DNN_net.optimal.m;  
            g2=(DNN_net.LayerDesign{iL}.dW.^2);
            DNN_net.LayerDesign{iL}.v=m*DNN_net.LayerDesign{iL}.v+(1-m)*g2;
            DNN_net.LayerDesign{iL}.delta_W=DNN_net.LayerDesign{iL}.dW./sqrt(DNN_net.LayerDesign{iL}.v+10^-8); 

            g2b=(DNN_net.LayerDesign{iL}.dWb.^2);
            DNN_net.LayerDesign{iL}.vb=m *DNN_net.LayerDesign{iL}.vb+(1-m)*g2b;
            DNN_net.LayerDesign{iL}.delta_Wb=DNN_net.LayerDesign{iL}.dWb./sqrt( DNN_net.LayerDesign{iL}.vb+10^-8);  
        elseif strcmp(LearningApproach,'Adam')
            b1=DNN_net.optimal.b1;
            b2=DNN_net.optimal.b2;
            g=DNN_net.LayerDesign{iL}.dW;
            g2=(DNN_net.LayerDesign{iL}.dW.^2);
            DNN_net.LayerDesign{iL}.mt=b1 *DNN_net.LayerDesign{iL}.mt+(1-b1).*g;
            DNN_net.LayerDesign{iL}.v=b2 *DNN_net.LayerDesign{iL}.v+(1-b2)*g2;
            DNN_net.LayerDesign{iL}.delta_W = DNN_net.LayerDesign{iL}.mt./(sqrt(DNN_net.LayerDesign{iL}.v)+10^-8);       

            gb=DNN_net.LayerDesign{iL}.dWb;
            g2b=(DNN_net.LayerDesign{iL}.dWb.^2);
            DNN_net.LayerDesign{iL}.mtb=b1 *DNN_net.LayerDesign{iL}.mtb+(1-b1).*gb;
            DNN_net.LayerDesign{iL}.vb=b2 *DNN_net.LayerDesign{iL}.vb+(1-b2)*g2b;
            DNN_net.LayerDesign{iL}.delta_Wb = DNN_net.LayerDesign{iL}.mtb./(sqrt(DNN_net.LayerDesign{iL}.vb)+10^-8);                   
        end
        DNN_net.LayerDesign{iL}.W=DNN_net.LayerDesign{iL}.W-r.*DNN_net.LayerDesign{iL}.delta_W;
        DNN_net.LayerDesign{iL}.Wb=DNN_net.LayerDesign{iL}.Wb-r.*DNN_net.LayerDesign{iL}.delta_Wb;
    end
end






