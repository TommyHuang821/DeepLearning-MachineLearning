function [y]=softmax_Sheng(x)
[n nlabel]=size(x);
y=exp(x)./repmat(sum(exp(x),2),1,nlabel);



