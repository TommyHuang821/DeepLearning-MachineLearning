function predictresult=DNN_Test(testdata,DNN_net)
Normal_testdata=[];
for i=1:size(testdata,2)
    Normal_testdata(:,i)= (testdata(:,i)-DNN_net.mu_train(i,1)) /DNN_net.sigma_trian(i,1);
end
bias = ones(size(testdata,1),1);
Normal_testdata = [Normal_testdata, bias];
DNN_test=forwordpropagation_Sheng(Normal_testdata',DNN_net);
predictresult=DNN_test.V{end};
