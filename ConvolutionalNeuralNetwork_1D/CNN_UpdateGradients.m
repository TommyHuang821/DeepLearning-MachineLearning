function [CNN_net]=CNN_UpdateGradients(CNN_net)
    
Num_Layer=numel(CNN_net.LayerDesign);

if ~isfield(CNN_net.opt,'alpha')
    r=0.1;
    CNN_net.opt.alpha=0.1;
else
    r=CNN_net.opt.alpha;  
end

if ~isfield(CNN_net.opt,'LearningApproach')
    LearningApproach='SGD';
    CNN_net.opt.LearningApproach=LearningApproach;
else
    LearningApproach=CNN_net.opt.LearningApproach;
end

if strcmp(LearningApproach,'SGD')
    for iL = 2 : Num_Layer
        if strcmp(CNN_net.LayerDesign{iL}.LayerType, 'C')
            for j = 1 : numel(CNN_net.LayerDesign{iL}.a)
                for ii = 1 : numel(CNN_net.LayerDesign{iL - 1}.a)
                    CNN_net.LayerDesign{iL}.kernelmap{ii}{j} = CNN_net.LayerDesign{iL}.kernelmap{ii}{j} - r * CNN_net.LayerDesign{iL}.dkernelmap{ii}{j};
                end
                CNN_net.LayerDesign{iL}.b{j} = CNN_net.LayerDesign{iL}.b{j} - r * CNN_net.LayerDesign{iL}.db{j};
            end
        elseif strcmp(CNN_net.LayerDesign{iL}.LayerType, 'F')
            CNN_net.LayerDesign{iL}.fc_W = CNN_net.LayerDesign{iL}.fc_W - r *  CNN_net.LayerDesign{iL}.dfc_W;
            CNN_net.LayerDesign{iL}.fc_b = CNN_net.LayerDesign{iL}.fc_b - r * CNN_net.LayerDesign{iL}.dfc_b;
        end
    end
end

