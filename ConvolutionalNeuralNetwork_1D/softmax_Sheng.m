function [y]=softmax_Sheng(x)
[n nlabel]=size(x);
if n<nlabel
    x=x';
    nlabel=n;
end
y=exp(x)./repmat(sum(exp(x),2),1,nlabel);



