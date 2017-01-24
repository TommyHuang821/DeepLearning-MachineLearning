clc
clear 
close all

traindata= [normrnd(0.5,0.1,100,2);...
           normrnd(1,0.1,100,2)];

option.MaxInter=100;
option.isplot=1;
option.numNode=2;

W=SOM(traindata,option);

for i=1:option.numNode
    for j=1:option.numNode
        w(:,:)=W(i,j,:);
        M(i,j)=norm(w);
    end
end
% 


figure(2)
imagesc(M)


