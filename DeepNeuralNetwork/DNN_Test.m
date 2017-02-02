function predictresult=DNN_Test(testdata,DNN_net)
Normal_testdata=testdata;
for i=1:size(testdata,2)
    Normal_testdata(:,i)= (testdata(:,i)-DNN_net.mu_train(i,1)) /DNN_net.sigma_trian(i,1);
end
DNN_test=forwordpropagation(Normal_testdata',DNN_net);
predictresult=DNN_test.V{end};
