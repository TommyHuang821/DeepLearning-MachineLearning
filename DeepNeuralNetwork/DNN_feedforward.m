function DNN_net=DNN_feedforward(this_pat,this_y,DNN_net,IndexTestdata)
% this_pat: input data, dim x n
if nargin<4
    IndexTestdata=0;
end
[~,N]=size(this_pat);
for iL= 1 : numel(DNN_net.LayerDesign)   %  layer
    tmpLayerType=DNN_net.LayerDesign{iL}.LayerType;   
    if strcmp(tmpLayerType,'Input')
        DNN_net.LayerDesign{iL}.a=this_pat;
    elseif strcmp(tmpLayerType,'Hidden')
        saf=DNN_net.LayerDesign{iL}.af;
        DNN_net.LayerDesign{iL}.z=DNN_net.LayerDesign{iL}.W*DNN_net.LayerDesign{iL-1}.a;
        DNN_net.LayerDesign{iL}.z=DNN_net.LayerDesign{iL}.z+repmat(DNN_net.LayerDesign{iL}.Wb,1,N);
        DNN_net.LayerDesign{iL}.a=saf(DNN_net.LayerDesign{iL}.z);        
        %%% dropout for learning in the hidden layer
        if IndexTestdata~=1
            if (DNN_net.dropoutFraction > 0)
                DNN_net.LayerDesign{iL}.dropOutMask = (rand(size(DNN_net.LayerDesign{iL}.a))>DNN_net.dropoutFraction);
                DNN_net.LayerDesign{iL}.a = DNN_net.LayerDesign{iL}.a.*DNN_net.LayerDesign{iL}.dropOutMask;
            end   
        end
    elseif  strcmp(tmpLayerType,'Output')
        saf=DNN_net.LayerDesign{iL}.af;
        DNN_net.LayerDesign{iL}.z=DNN_net.LayerDesign{iL}.W*DNN_net.LayerDesign{iL-1}.a;
        DNN_net.LayerDesign{iL}.z=DNN_net.LayerDesign{iL}.z+repmat(DNN_net.LayerDesign{iL}.Wb,1,N);
        DNN_net.LayerDesign{iL}.a=saf(DNN_net.LayerDesign{iL}.z);  
        if (DNN_net.LayerDesign{iL}.n_node>=2)
            tmp=softmax_Sheng(DNN_net.LayerDesign{iL}.a');
            DNN_net.LayerDesign{iL}.a=tmp';
        end
        DNN_net.LayerDesign{iL}.y=this_y;
    end
end

