function DNN_net=Initialization_Net_DNN(DNN_net)
%%% LearningApproach:'SGD','Momentum','AdaGrad','RMSProp','Adam'


%%% learning rate, default=0.1
if ~isfield(DNN_net,'r'); DNN_net.r=0.1; end
%%% max of learning iteration (epoch), if not converge 
if ~isfield(DNN_net,'maxIter'); DNN_net.maxIter=100; end
% droupout fraction
if ~isfield(DNN_net,'dropoutFraction');  DNN_net.dropoutFraction=0; end
%%% batch learning
if ~isfield(DNN_net,'batchsize'); DNN_net.batchsize=100; end
%%% normalization for each dimension
if ~isfield(DNN_net,'isnormalization'); DNN_net.isnormalization=0; end
%%% default optimalization is 'SGD'
if ~isfield(DNN_net,'LearningApproach'); DNN_net.LearningApproach='SGD'; end

for iL= 1 : numel(DNN_net.LayerDesign)   %  layer
    tmpLayerType=DNN_net.LayerDesign{iL}.LayerType;    
    if strcmp(tmpLayerType,'Hidden') || strcmp(tmpLayerType,'Output')
        tmpActF=DNN_net.LayerDesign{iL}.ActF;
        DNN_net.LayerDesign{iL}.W = randn(DNN_net.LayerDesign{iL}.n_node,DNN_net.LayerDesign{iL-1}.n_node)- 0.5/10;
        DNN_net.LayerDesign{iL}.Wb= zeros(DNN_net.LayerDesign{iL}.n_node,1);
        DNN_net.LayerDesign{iL}.delta_W=zeros(DNN_net.LayerDesign{iL}.n_node,DNN_net.LayerDesign{iL-1}.n_node);
        DNN_net.LayerDesign{iL}.delta_Wb=zeros(DNN_net.LayerDesign{iL}.n_node,1); 
        
        if strcmp(tmpActF,'PReLU') || strcmp(tmpActF,'ELU')
            if isfield(DNN_net.LayerDesign{iL},'option_ActFunction'); 
                option_ActFunction=DNN_net.LayerDesign{iL}.option_ActFunction; 
            else
                option_ActFunction=0.01;
            end
            [DNN_net.LayerDesign{iL}.af, DNN_net.LayerDesign{iL}.daf]=AactivationFunction(tmpActF,option_ActFunction);
        else
            [DNN_net.LayerDesign{iL}.af, DNN_net.LayerDesign{iL}.daf]=AactivationFunction(tmpActF);
        end
        
        if strcmp(DNN_net.LearningApproach,'Momentum')
            DNN_net.optimal.m=0.99;
        elseif strcmp(DNN_net.LearningApproach,'RMSProp')
            DNN_net.LayerDesign{iL}.v=zeros(DNN_net.LayerDesign{iL}.n_node,DNN_net.LayerDesign{iL-1}.n_node);
            DNN_net.LayerDesign{iL}.vb=zeros(DNN_net.LayerDesign{iL}.n_node,1);
            DNN_net.optimal.m=0.9;
        elseif strcmp(DNN_net.LearningApproach,'Adam')
            DNN_net.LayerDesign{iL}.v=zeros(DNN_net.LayerDesign{iL}.n_node,DNN_net.LayerDesign{iL-1}.n_node);
            DNN_net.LayerDesign{iL}.vb=zeros(DNN_net.LayerDesign{iL}.n_node,1);
            DNN_net.LayerDesign{iL}.mt=zeros(DNN_net.LayerDesign{iL}.n_node,DNN_net.LayerDesign{iL-1}.n_node);
            DNN_net.LayerDesign{iL}.mtb=zeros(DNN_net.LayerDesign{iL}.n_node,1);
            DNN_net.optimal.b1=0.9;
            DNN_net.optimal.b2=0.999;
        end 
    end
end

        

