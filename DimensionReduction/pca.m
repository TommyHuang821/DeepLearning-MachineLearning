function [RotMatrix coe_PC xRot ProjectedData]=pca(data)
%  Implement PCA to obtain the rotation matrix (RotMatrix), which is the
%  eigenbasis of covariance of input.
% edit by Chih-Sheng (Tommy) Huang.
%
% Input: 
%       data: input data, n*dim
% Output:
%     RotMatrix: PCA transformation matrix
%     coe_PC: variance 
%     xRot: projected data
%     ProjectedData.xPCAwhite :PCA whitening   
%     ProjectedData.xZCAWhite :ZCA whitening

[N dim ]=size(data);
if  (dim> N)
    data=data';
    tmpN =dim;
    tmpdim=N;
    N=tmpN;
    dim=tmpdim;
end

avg = mean(data);
data = data - repmat(avg, N, 1);

% covariance matrix
sigma = data' * data / N;
% singular value decomposition (SVD)
[RotMatrix, S] = svd(sigma);
coe_PC=diag(S);
xRot = RotMatrix* data';



% PCA-whitening
epsilon=10^(-5);
xPCAwhite = diag(1./sqrt(coe_PC + epsilon)) * RotMatrix * data';

% ZCA-whitening
xZCAWhite = RotMatrix * diag(1./sqrt(coe_PC + epsilon)) * RotMatrix * data';

ProjectedData.xPCAwhite=xPCAwhite;
ProjectedData.xZCAWhite=xZCAWhite;