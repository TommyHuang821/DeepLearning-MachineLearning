function [CNN_net]=CNN_feedforward(this_pat,CNN_net)

Num_Layer=numel(CNN_net.LayerDesign);
CNN_net.LayerDesign{1}.a{1}=this_pat;
n_inputmap=1;

for iL=2:Num_Layer
    if strcmp(CNN_net.LayerDesign{iL}.LayerType, 'C') % convolution
        af=CNN_net.LayerDesign{iL}.af;
        for j = 1 : CNN_net.LayerDesign{iL}.n_map   %  for each output map
            % output map (z)
            z = zeros(size(CNN_net.LayerDesign{iL-1}.a{1}) - [CNN_net.LayerDesign{iL}.kernelsize - 1 CNN_net.LayerDesign{iL}.kernelsize - 1 0]);
            for i = 1 : n_inputmap   %  for each input map
                %  convolution for corresponding kernel map and input image to temp output map
                z = z + convn(CNN_net.LayerDesign{iL-1}.a{i}, CNN_net.LayerDesign{iL}.kernelmap{i}{j}, 'valid');
            end
            %  add bias, pass through activation function
            CNN_net.LayerDesign{iL}.a{j} = af(z + CNN_net.LayerDesign{iL}.b{j});
            CNN_net.LayerDesign{iL}.z{j} = (z + CNN_net.LayerDesign{iL}.b{j});
        end
        n_inputmap = CNN_net.LayerDesign{iL}.n_map; %  set number of input maps to this layers number of outputmaps

    elseif strcmp(CNN_net.LayerDesign{iL}.LayerType, 'S') %  downsample
        for j = 1 : n_inputmap
            z = convn(CNN_net.LayerDesign{iL-1}.a{j}, ones(CNN_net.LayerDesign{iL}.scale) / (CNN_net.LayerDesign{iL}.scale ^ 2), 'valid');  
            CNN_net.LayerDesign{iL}.a{j} = z(1 : CNN_net.LayerDesign{iL}.scale : end, 1 : CNN_net.LayerDesign{iL}.scale : end, :);
            
            z = convn(CNN_net.LayerDesign{iL-1}.z{j}, ones(CNN_net.LayerDesign{iL}.scale) / (CNN_net.LayerDesign{iL}.scale ^ 2), 'valid');  
            CNN_net.LayerDesign{iL}.z{j} = z(1 : CNN_net.LayerDesign{iL}.scale : end, 1 : CNN_net.LayerDesign{iL}.scale : end, :);
        end
    elseif strcmp(CNN_net.LayerDesign{iL}.LayerType,'F') 
        % Layer: full connection 
        %  concatenate all end layer feature maps into vector
        CNN_net.LayerDesign{iL}.fv = [];
        for j = 1 : numel(CNN_net.LayerDesign{Num_Layer-1}.a)
            sa = size(CNN_net.LayerDesign{Num_Layer-1}.a{j});
            CNN_net.LayerDesign{iL}.fv = [CNN_net.LayerDesign{iL}.fv; reshape(CNN_net.LayerDesign{Num_Layer-1}.a{j}, sa(1) * sa(2), sa(3))];
        end
        %  feedforward into output perceptrons
        af=CNN_net.LayerDesign{iL}.af;
        CNN_net.Output = af(CNN_net.LayerDesign{iL}.fc_W * CNN_net.LayerDesign{iL}.fv + repmat(CNN_net.LayerDesign{iL}.fc_b, 1, size(CNN_net.LayerDesign{iL}.fv, 2)));   
        CNN_net.Output_noaf = (CNN_net.LayerDesign{iL}.fc_W * CNN_net.LayerDesign{iL}.fv + repmat(CNN_net.LayerDesign{iL}.fc_b, 1, size(CNN_net.LayerDesign{iL}.fv, 2)));   
%         pred=softmax_Sheng(CNN_net.Output);
%         CNN_net.Output =pred';
        
    end
end
    
