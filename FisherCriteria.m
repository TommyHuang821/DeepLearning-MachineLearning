function [FisherInformation J Sb Sw]=FisherCriteria(data,label)
%%%Fisher Criterion Calculation
% Input: 
%       data: n x dim matrix
%       label: corresponding label, class label must start at 1
% Output:
%       Sb= Between-class scatter matrix
%       Sw= Within-class scatter matrix
%       FisherInformation = inv(Sw)*Sb 
%       J: Fisher Criteria, (J1, J2, J3)
%           J1=norm(sb)/norm(sw);
%           J2=trace(sb)/trace(sw);
%           J3=trace(sb*pinv(sw));
%
%   This version is edited by Chih-Sheng (Tommy) Huang
%   Jan 28, 2014
%

% check data length and label length
[a b]=size(data);
len_label=length(label);
if a==len_label
    data=data;
elseif b==len_label
    data=data';
else a~=len_label & b~=len_label
    error('Length of inpit data is not equal to length of input label.')
end
%
max_label=max(label);
min_label=min(label);
num_label=0; % check number of class
for i=min_label:max_label
    if sum(find(label==i))~=0
        num_label=num_label+1;
    end
end
%
dim=size(data,2);
sb=zeros(dim); % set initial between scatter matrix
sw=zeros(dim); % set initial within scatter matrix
mu_i={};cov_i={};
c=0;
for la=min_label:max_label
    c=c+1;
    pos=find(label==la);
    pi=length(pos)/len_label;
    perdata{c}=data(pos,:);
    mu=mean(data);
    mu_i{c}=mean(perdata{la});
    cov_i{c}=cov(perdata{la});
    dif_mu=mu_i{c}-mu;
    sb=sb+pi*(dif_mu'*dif_mu);
    sw=sw+pi*cov_i{c};
end
Sb=sb/num_label;
Sw=sw/num_label;
FisherInformation=Sb*pinv(Sw);
J(1)=norm(Sb)/norm(Sw);
J(2)=trace(Sb)/trace(Sw);
J(3)=trace(sb*pinv(sw));

