function predictresult=DNN_Test(testdata,DNN_net)
IndexTestdata=1;
if DNN_net.isnormalization==1
    Normal_testdata=testdata;
    for i=1:size(testdata,2)
        Normal_testdata(:,i)= (testdata(:,i)-DNN_net.mu_train(i,1)) /DNN_net.sigma_trian(i,1);
    end
    DNN_test=DNN_feedforward(Normal_testdata',[],DNN_net,IndexTestdata);
    predictresult=DNN_test.LayerDesign{end}.a;
else
    DNN_test=DNN_feedforward(testdata',[],DNN_net,IndexTestdata);
    predictresult=DNN_test.LayerDesign{end}.a;
end
