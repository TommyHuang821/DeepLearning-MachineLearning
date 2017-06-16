function DNN_net=DNN_feedbackward(DNN_net)

% back-propagation
for iL= numel(DNN_net.LayerDesign) : -1 : 2    %  layer
    tmpLayerType=DNN_net.LayerDesign{iL}.LayerType;   
    if strcmp(tmpLayerType,'Output')
        DNN_net.LayerDesign{iL}.e=DNN_net.LayerDesign{iL}.a-DNN_net.LayerDesign{iL}.y;
        sdaf=DNN_net.LayerDesign{iL}.daf;
        tmp=(sdaf(DNN_net.LayerDesign{iL}.z).*DNN_net.LayerDesign{iL}.e);
        DNN_net.LayerDesign{iL-1}.e=DNN_net.LayerDesign{iL}.W'*tmp;
        DNN_net.LayerDesign{iL}.dW=tmp*DNN_net.LayerDesign{iL-1}.a'/size(tmp, 1);
        DNN_net.LayerDesign{iL}.dWb=mean(tmp,2);
    elseif strcmp(tmpLayerType,'Hidden')
        sdaf=DNN_net.LayerDesign{iL}.daf;
        tmp=(sdaf(DNN_net.LayerDesign{iL}.z).*DNN_net.LayerDesign{iL}.e);
        DNN_net.LayerDesign{iL-1}.e=DNN_net.LayerDesign{iL}.W'*tmp;
        if(DNN_net.dropoutFraction>0) && iL>2
            DNN_net.LayerDesign{iL-1}.e = DNN_net.LayerDesign{iL-1}.e .* DNN_net.LayerDesign{iL-1}.dropOutMask;
        end  
        DNN_net.LayerDesign{iL}.dW=tmp*DNN_net.LayerDesign{iL-1}.a'/size(tmp, 1);
        DNN_net.LayerDesign{iL}.dWb=mean(tmp,2);
    end   
end




