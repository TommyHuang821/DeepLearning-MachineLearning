function [CNN_net]=CNN_feedbackward_1D(batch_out,CNN_net)
    Num_Layer=numel(CNN_net.LayerDesign);
    CNN_net.LayerDesign{Num_Layer}.e = CNN_net.Output - batch_out; % error: output
    CNN_net.L = 1/2* sum(CNN_net.LayerDesign{Num_Layer}.e(:) .^ 2) / size(CNN_net.LayerDesign{Num_Layer}.e, 2); % loss function

    %%  backward propagation deltas
%     af=CNN_net.LayerDesign{Num_Layer}.af;
    daf=CNN_net.LayerDesign{Num_Layer}.daf;
    CNN_net.LayerDesign{Num_Layer}.od=CNN_net.LayerDesign{Num_Layer}.e .*daf( CNN_net.Output); % delta: output
%     CNN_net.LayerDesign{Num_Layer}.od=CNN_net.LayerDesign{Num_Layer}.e .*daf( CNN_net.Output_noaf); % delta: output
    CNN_net.LayerDesign{Num_Layer}.fvd = (CNN_net.LayerDesign{Num_Layer}.fc_W' * CNN_net.LayerDesign{Num_Layer}.od')';  % delta: feature vector    
    
    if strcmp(CNN_net.LayerDesign{Num_Layer}.LayerType, 'C')         %  only conv layers has sigm function
        CNN_net.LayerDesign{Num_Layer}.fvd = CNN_net.LayerDesign{Num_Layer}.fvd .* daf(CNN_net.LayerDesign{Num_Layer}.fv);
    end
    % reshape delta feature vector into output map size
    sa = size(CNN_net.LayerDesign{Num_Layer-1}.a{1});
    fvnum = sa(2);
    for j = 1 : numel(CNN_net.LayerDesign{Num_Layer-1}.a)
        CNN_net.LayerDesign{Num_Layer-1}.d{j} = CNN_net.LayerDesign{Num_Layer}.fvd(:,(((j - 1) * fvnum + 1) : j * fvnum));
    end

     for iL = (Num_Layer - 2) : -1 : 1
        if strcmp(CNN_net.LayerDesign{iL}.LayerType, 'C')
            daf=CNN_net.LayerDesign{iL}.daf;
            for j = 1 : numel(CNN_net.LayerDesign{iL}.a)
                tmp=CNN_net.LayerDesign{iL}.z{j};
                tmp2=(ExpandMatrix(CNN_net.LayerDesign{iL + 1}.d{j}, [1 CNN_net.LayerDesign{iL + 1}.scale ]) / CNN_net.LayerDesign{iL + 1}.scale );
                CNN_net.LayerDesign{iL}.d{j} = daf(tmp) .* tmp2;
            end        
        elseif strcmp(CNN_net.LayerDesign{iL}.LayerType, 'S')
            for i = 1 : numel(CNN_net.LayerDesign{iL}.a)
                z = zeros(size(CNN_net.LayerDesign{iL}.a{1}));
                for j = 1 : numel(CNN_net.LayerDesign{iL + 1}.a)
                     z = z + convn(CNN_net.LayerDesign{iL + 1}.d{j}, rot180(CNN_net.LayerDesign{iL + 1}.kernelmap{i}{j}), 'full');
                end
                CNN_net.LayerDesign{iL}.d{i} = z;
            end
        end
    end

    %% calculate gradients
    for iL = 2 : Num_Layer
        if strcmp(CNN_net.LayerDesign{iL}.LayerType, 'C')
            for j = 1 : numel(CNN_net.LayerDesign{iL}.a)
                for i = 1 : numel(CNN_net.LayerDesign{iL - 1}.a)
                    CNN_net.LayerDesign{iL}.dkernelmap{i}{j} = convn(flipall(CNN_net.LayerDesign{iL - 1}.a{i}), CNN_net.LayerDesign{iL}.d{j}, 'valid') / size(CNN_net.LayerDesign{iL}.d{j}, 2);
                end
                CNN_net.LayerDesign{iL}.db{j} = sum(CNN_net.LayerDesign{iL}.d{j}(:)) / size(CNN_net.LayerDesign{iL}.d{j}, 2);
            end
        elseif strcmp(CNN_net.LayerDesign{iL}.LayerType, 'F')
            CNN_net.LayerDesign{Num_Layer}.dfc_W= CNN_net.LayerDesign{Num_Layer}.od'  *  CNN_net.LayerDesign{Num_Layer}.fv/ size(CNN_net.LayerDesign{Num_Layer}.od , 1);
            CNN_net.LayerDesign{Num_Layer}.dfc_b=mean(CNN_net.LayerDesign{Num_Layer}.od, 1)';  
        end
    end

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end

    function X=flipall(X)
        for ii=1:ndims(X)
            X = flipdim(X,ii);
        end
    end
end
