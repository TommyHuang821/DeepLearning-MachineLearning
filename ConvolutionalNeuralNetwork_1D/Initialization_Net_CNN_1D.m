function CNN_net=Initialization_Net_CNN_1D(CNN_net,mapsize,Outsize)

n_inputmap = 1;
for iL= 1 : numel(CNN_net.LayerDesign)   %  layer
    tmpLayerType=CNN_net.LayerDesign{iL}.LayerType;
    if strcmp(tmpLayerType,'C')
        mapsize = mapsize - CNN_net.LayerDesign{iL}.kernelsize + 1; % because convolution, the map size would be changed.
        n_out = CNN_net.LayerDesign{iL}.n_map * CNN_net.LayerDesign{iL}.kernelsize ^ 2;

         for j = 1 : CNN_net.LayerDesign{iL}.n_map  % Number of output map
            n_in = n_inputmap * CNN_net.LayerDesign{iL}.kernelsize ^ 2;
            for i = 1 : n_inputmap  %  input map
                 CNN_net.LayerDesign{iL}.kernelmap{i}{j} = (rand(1,CNN_net.LayerDesign{iL}.kernelsize) - 0.5) * 2 * sqrt(6 / (n_in + n_out));
            end
            CNN_net.LayerDesign{iL}.b{j} = 0;
         end
        n_inputmap = CNN_net.LayerDesign{iL}.n_map;
        
        if isfield(CNN_net.LayerDesign{iL},'ActF')==0
            ActF='ReLU';
            fprintf(['Not define the aactivation function for ' CNN_net.LayerDesign{iL}.LayerName  ', using default: ReLU\n'])          
        else
            ActF=CNN_net.LayerDesign{iL}.ActF;
        end
        [af daf]=AactivationFunction(ActF);
        CNN_net.LayerDesign{iL}.af=af;
        CNN_net.LayerDesign{iL}.daf=daf;
        
    elseif strcmp(tmpLayerType,'S')
        mapsize = mapsize / CNN_net.LayerDesign{iL}.scale;  % because pooling, the map size would be changed.
        % if the mapsize is not a integer, error message
        assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(iL) ' size must be integer. Actual: ' num2str(mapsize)]);
        for j = 1 : n_inputmap
            CNN_net.LayerDesign{iL}.b{j} = 0;
        end
    elseif strcmp(tmpLayerType,'F')    
        %%% Last Layer
        %%% initialize for Full Connection
        fc_input_num = mapsize * n_inputmap; 
        fc_output_num = Outsize;
        CNN_net.LayerDesign{iL}.fc_b = zeros(fc_output_num, 1); % the biases of the output neurons.
        %the weights between the last layer and the output neurons.
        CNN_net.LayerDesign{iL}.fc_W = (rand(fc_output_num, fc_input_num) - 0.5) * 2 * sqrt(6 / (fc_output_num + fc_input_num)); 
        [af daf]=AactivationFunction(ActF);
        CNN_net.LayerDesign{iL}.af=af;
        CNN_net.LayerDesign{iL}.daf=daf;
    end 
end

