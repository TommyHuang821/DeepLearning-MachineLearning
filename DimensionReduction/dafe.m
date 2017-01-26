function [DAFE_vect, DAFE_val, DAFE_dataRot ] = dafe(data,classlabel, features)
%   Discriminant Analysis Feature Extraction (also called Linear Discriminant Analysis)
% data: n * dim
% classlabel: n * 1 (classlabel numbers for each pattern).
%       classlabel MUST be consecutive from 1 to the number of classes.

[N dim]=size(data);
NL=length(classlabel);
if (NL~=N && dim~=N)
    error('data length is not equal to classlabel length');
elseif (NL~=N && dim==N)
    data=data';
    tmpN =dim;
    tmpdim=N;
    N=tmpN;
    dim=tmpdim;
end
nc=max(classlabel);
if (nc < 2),
   error('dafe.m: There must be at least 2 classes.')
end
Sigma_data = zeros(dim*nc,dim);
Mu_data=zeros(nc,dim);
Sw = zeros(dim);
for i = 1:nc
    tmp=data((classlabel==i),:);
    cindex = (i-1)*dim + 1 : i*dim ;
	Sigma_data(cindex,:) = cov(tmp);
    Sw = Sw + Sigma_data(cindex,:);
  
    [Mu_data(i,:),] = mean(data(classlabel==i,:));
end
Sw = Sw / nc;
Sw = 0.5 * Sw + 0.5 * diag(diag(Sw)); % regualrization for within-class scatter matrix
Sb = cov(Mu_data)*(nc-1)/nc; % between-calss scatter matrix

[DAFE_vect, DAFE_val] = svd(pinv(Sw)*Sb);
DAFE_vect=DAFE_vect(:,1:features)';
DAFE_val=DAFE_val(:,1:features)';
DAFE_val=diag(DAFE_val);
xRot = DAFE_vect* data';
DAFE_dataRot=xRot';